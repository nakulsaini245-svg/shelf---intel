"""Daily dynamics: calendar, pricing, demand, inventory. Fully vectorised.

One simulated day is bounded by 22:00 (day d runs from 22:00 of d-1 to 22:00 of d)
and has four periods ending at the snapshot times 06:00, 12:00, 18:00, 22:00.
Replenishment arrives at the start of period 1 (overnight), with an optional
emergency top-up at the start of period 3.

Order of operations within a day (all state is carried between days):
  prices -> demand rates (using YESTERDAY's share-of-voice) -> inventory/sales per period
  -> end-of-day replenishment orders. The search-visibility auction (observe.py) then runs
  on the day's realised availability, and its share-of-voice feeds tomorrow's demand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .events import DayModifiers
from .seeds import rng_for
from .world import World

RING = 8  # replenishment pipeline length in days (max lead 2 + delay 1 < 8)


@dataclass
class State:
    on_hand: np.ndarray       # [S] int64
    ewma: np.ndarray          # [S] planner forecast of daily sales
    pipe: np.ndarray          # [RING,S] inbound units by arrival day
    promo_left: np.ndarray    # [P,K]
    promo_depth: np.ndarray   # [P,K]
    prices_prev: np.ndarray   # [P,K] (nan = not listed)
    conv_ema: np.ndarray      # [C] availability history (drives organic rank)
    sov_prev: np.ndarray      # [NP,Q,B]
    sov_ref: np.ndarray | None
    last_avail: np.ndarray    # [C] bool, previous day's last snapshot (for stale scrapes)


@dataclass
class DayOut:
    day: int
    date: date
    price_pk: np.ndarray      # [P,K]
    promo_active: np.ndarray  # [P,K] bool
    price_cell: np.ndarray    # [C] int
    listed_today: np.ndarray  # [C] bool
    demand_cell: np.ndarray   # [C] int
    sold_cell: np.ndarray     # [C] int
    avail_snap: np.ndarray    # [4,C] bool, true availability incl. listing
    onhand_snap: np.ndarray   # [4,S]


class Engine:
    def __init__(self, world: World):
        self.w = w = world
        self.cfg = c = world.cfg
        self.seed = c.seed
        subs = [c.subcats[s] for s in w.subcats]
        self.el = np.array([s.elasticity for s in subs])[w.sku_sub]            # [K]
        self.amp = np.array([s.seasonal_amp for s in subs])
        self.peak = np.array([s.seasonal_peak_doy for s in subs])
        self.fests = [(date.fromisoformat(f.date), f.window_days,
                       np.array([f.mult.get(s, 1.0) for s in w.subcats])) for f in c.demand.festivals]
        plat_dem = np.array([p.demand_idx for p in c.platforms])
        self.lam0 = (c.demand.base_units_per_cell_day * w.sku_pop[w.cell_sku]
                     * w.pin_dem[w.cell_pin] * plat_dem[w.cell_plat])
        self.cell_q = w.sku_sub[w.cell_sku]
        self.cell_brand = w.sku_brand[w.cell_sku]
        self.cell_el = self.el[w.cell_sku]
        self.cell_ref = w.ref_price[w.cell_plat, w.cell_sku]
        self.gid = w.cell_pair * w.Q + self.cell_q
        self.cell_pop = w.sku_pop[w.cell_sku]
        self.w_g = np.bincount(self.gid, weights=self.cell_pop, minlength=w.n_pairs * w.Q)  # popularity mass
        rel = np.array([p.restock_reliability for p in c.platforms])
        self.s_rel = rel[w.series_plat]
        self.s_case = w.sku_case[w.series_sku]
        self.s_lead = w.store_lead[w.series_store]
        self.s_wo = np.minimum(0.5, c.inventory.writeoff_scale / w.sku_shelf[w.series_sku])
        self.p_start = np.clip(c.pricing.promo_start_prob * w.sku_promo_mult[None, :].repeat(w.P, 0), 0, 1)
        self.period_w = np.array(c.demand.period_weights)
        self.S = w.n_series

    # ------------------------------------------------------------------ state
    def init_state(self) -> State:
        w, c = self.w, self.cfg
        exp_daily = np.bincount(w.cell_series, weights=self.lam0, minlength=self.S)
        ewma = np.maximum(exp_daily, 0.5)
        on_hand = np.ceil(ewma * c.inventory.initial_cover_days).astype(np.int64)
        return State(
            on_hand=on_hand, ewma=ewma, pipe=np.zeros((RING, self.S), dtype=np.int64),
            promo_left=np.zeros((w.P, w.K), dtype=np.int64), promo_depth=np.zeros((w.P, w.K)),
            prices_prev=w.ref_price.copy(), conv_ema=np.full(w.n_cells, 0.95),
            sov_prev=np.full((w.n_pairs, w.Q, w.B), 1.0 / w.B), sov_ref=None,
            last_avail=np.ones(w.n_cells, dtype=bool))

    # --------------------------------------------------------------- calendar
    def calendar(self, day: int) -> tuple[date, float, np.ndarray]:
        d = self.cfg.start + timedelta(days=day)
        doy = d.timetuple().tm_yday
        seasonal = 1.0 + self.amp * np.cos(2 * math.pi * (doy - self.peak) / 365.0)
        fest = np.ones(self.w.Q)
        for fd, win, mult in self.fests:
            tri = max(0.0, 1.0 - abs((d - fd).days) / win)
            fest = fest * (1.0 + (mult - 1.0) * tri)
        return d, self.cfg.demand.dow_mult[d.weekday()], seasonal * fest

    # ---------------------------------------------------------------- pricing
    def update_prices(self, day: int, st: State) -> tuple[np.ndarray, np.ndarray]:
        w, pr = self.w, self.cfg.pricing
        rng = rng_for(self.seed, "platform_sku", day, "pricing")
        u_start = rng.random((w.P, w.K))
        lens = rng.integers(pr.promo_len_range[0], pr.promo_len_range[1] + 1, (w.P, w.K))
        depth = rng.uniform(pr.promo_depth_range[0], pr.promo_depth_range[1], (w.P, w.K))
        u_match = rng.random((w.P, w.K))
        start = (st.promo_left == 0) & w.listed & (u_start < self.p_start)
        st.promo_left = np.where(start, lens, st.promo_left)
        st.promo_depth = np.where(start, depth, st.promo_depth)
        active = st.promo_left > 0
        st.promo_left = np.maximum(st.promo_left - 1, 0)
        target = w.ref_price * (1.0 - np.where(active, st.promo_depth, 0.0))
        prev = st.prices_prev
        other_min = np.empty_like(prev)
        for p in range(w.P):
            other_min[p] = np.fmin.reduce(np.delete(prev, p, axis=0), axis=0)  # ignores NaN
        with np.errstate(invalid="ignore"):
            gap = prev > other_min * (1.0 + pr.match_tolerance)
        trigger = gap & (u_match < pr.match_prob) & w.listed
        matched = np.maximum(other_min, w.floor_price)
        price = np.where(trigger, np.minimum(target, matched), target)
        price = np.where(w.listed, np.clip(np.round(price), w.floor_price, w.sku_mrp[None, :]), np.nan)
        st.prices_prev = price
        return price, active

    def cell_prices(self, price_pk: np.ndarray, mods: DayModifiers) -> np.ndarray:
        w = self.w
        p = price_pk[w.cell_plat, w.cell_sku]
        if mods.price_mult is not None:
            p = p * mods.price_mult
        return np.clip(np.round(p), w.floor_price[w.cell_plat, w.cell_sku], w.sku_mrp[w.cell_sku]).astype(np.int64)

    # ----------------------------------------------------------------- demand
    def demand_rate(self, dow: float, season: np.ndarray, price_cell: np.ndarray,
                    promo_cell: np.ndarray, listed_today: np.ndarray, st: State,
                    mods: DayModifiers) -> np.ndarray:
        dm = self.cfg.demand
        ratio = price_cell / self.cell_ref
        own = ratio ** (-self.cell_el)
        wp = self.cell_pop
        den = self.w_g[self.gid] - wp
        s_g = np.bincount(self.gid, weights=ratio * wp, minlength=len(self.w_g))
        comp = np.where(den > 1e-9, (s_g[self.gid] - ratio * wp) / np.maximum(den, 1e-9), 1.0)
        cross = comp ** (dm.cross_elasticity_ratio * self.cell_el)
        lift = np.where(promo_cell, dm.promo_display_lift, 1.0)
        sov = np.ones(self.w.n_cells)
        if st.sov_ref is not None:
            cur = st.sov_prev[self.w.cell_pair, self.cell_q, self.cell_brand]
            ref = np.maximum(st.sov_ref[self.w.cell_pair, self.cell_q, self.cell_brand], 0.02)
            sov = 1.0 + dm.sov_eta * np.clip(cur / ref - 1.0, -1.0, 3.0)
        # The clip guards against runaway ORGANIC cross-effects only; explicit event multipliers
        # (including 0 for an outage) are applied after it and must never be floored.
        eff = np.clip(own * cross * lift * sov, dm.demand_mult_clip[0], dm.demand_mult_clip[1])
        if mods.demand_mult is not None:
            eff = eff * mods.demand_mult
        return self.lam0 * dow * season[self.cell_q] * eff * listed_today

    # -------------------------------------------------------------- inventory
    def run_day_inventory(self, day: int, lam: np.ndarray, listed_today: np.ndarray, st: State,
                          mods: DayModifiers) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance stock through the 4 periods. Returns demand, sold, avail snapshots, on-hand snapshots."""
        w, iv = self.w, self.cfg.inventory
        S, C = self.S, w.n_cells
        rng = rng_for(self.seed, "cell", day, "demand")
        r = self.cfg.demand.nb_dispersion
        lam_k = lam[None, :] * self.period_w[:, None]
        dem = _nb_inverse(rng.random((4, C)), lam_k, r)                        # [4,C] neg-binomial, CRN

        arrive = st.pipe[day % RING].copy()
        st.pipe[day % RING] = 0
        if mods.supply_fill_mult is not None:
            arrive = np.floor(arrive * mods.supply_fill_mult).astype(np.int64)
        st.on_hand = st.on_hand + arrive
        if mods.stock_destroy_frac is not None:
            st.on_hand = st.on_hand - np.floor(st.on_hand * mods.stock_destroy_frac).astype(np.int64)
        rs = rng_for(self.seed, "series", day, "stock")
        # shelf-life write-off: expectation with stochastic rounding (fixed uniforms => common random numbers)
        st.on_hand = st.on_hand - np.floor(st.on_hand * self.s_wo + rs.random(S)).astype(np.int64)
        u_top = rs.random(S)

        sold_total = np.zeros(C, dtype=np.int64)
        avail = np.zeros((4, C), dtype=bool)
        onhand_snap = np.zeros((4, S), dtype=np.int64)
        for k in range(4):
            if k == 2:  # emergency mid-day top-up for sold-out series
                mask = (st.on_hand <= 0) & (u_top < iv.topup_prob * self.s_rel)
                qty = self.s_case * np.ceil(np.maximum(st.ewma * 0.5, 1.0) / self.s_case)
                st.on_hand = st.on_hand + np.where(mask, qty, 0).astype(np.int64)
            sold = _ration(dem[k], w.cell_series, st.on_hand, S)
            st.on_hand = st.on_hand - np.bincount(w.cell_series, weights=sold, minlength=S).astype(np.int64)
            sold_total += sold
            onhand_snap[k] = st.on_hand
            avail[k] = (st.on_hand > 0)[w.cell_series] & listed_today
        return dem.sum(0), sold_total, avail, onhand_snap

    def place_orders(self, day: int, sold_cell: np.ndarray, st: State) -> None:
        iv = self.cfg.inventory
        S = self.S
        sales_s = np.bincount(self.w.cell_series, weights=sold_cell, minlength=S)
        st.ewma = (1 - iv.ewma_alpha) * st.ewma + iv.ewma_alpha * sales_s
        ro = rng_for(self.seed, "series", day, "orders")
        delay_p = np.clip(iv.delay_base * (1.0 - self.s_rel) / 0.08, 0.0, 0.5)
        delay = (ro.random(S) < delay_p).astype(np.int64)
        fill = np.clip(ro.normal(self.s_rel, iv.fill_sd), 0.4, 1.0)
        target = np.maximum(st.ewma * (self.s_lead + iv.cover_days), iv.min_presentation_units)
        need = np.maximum(target - (st.on_hand + st.pipe.sum(0)), 0.0)
        qty = np.ceil(need / self.s_case) * self.s_case
        filled = np.floor(qty * fill).astype(np.int64)
        st.pipe[(day + self.s_lead + delay) % RING, np.arange(S)] += filled


def _nb_inverse(u: np.ndarray, mean: np.ndarray, r: float) -> np.ndarray:
    """Negative-binomial(r, mean) quantile at u, by inversion.

    Uses exactly one uniform per draw regardless of the parameters, so two worlds that differ only in
    `mean` still share their random numbers (common random numbers). numpy's poisson/binomial samplers
    consume a parameter-dependent number of bits, which would desynchronise every later draw and make
    counterfactual differences noisy.
    """
    q = r / (r + mean)
    pmf = q ** r                                   # P(0)
    cdf = pmf.copy()
    k = np.zeros(u.shape, dtype=np.int64)
    mx = float(mean.max())
    for i in range(int(mx + 12.0 * np.sqrt(mx * (1.0 + mx / r))) + 12):
        todo = u > cdf
        if not todo.any():
            break
        k += todo
        pmf = pmf * ((i + r) / (i + 1.0)) * (1.0 - q)
        cdf = cdf + pmf
    return k


def _ration(dem: np.ndarray, series: np.ndarray, on_hand: np.ndarray, S: int) -> np.ndarray:
    """Cap demand at on-hand per series; split the cap across cells by largest remainder.

    Guarantees: 0 <= sold <= demand per cell, and per-series total <= on_hand.
    """
    d_s = np.bincount(series, weights=dem, minlength=S)
    cons = d_s > on_hand
    if not cons.any():
        return dem
    ratio = np.where(cons, on_hand / np.maximum(d_s, 1.0), 1.0)[series]
    x = dem * ratio
    fl = np.floor(x)
    left = np.rint(np.minimum(d_s, on_hand) - np.bincount(series, weights=fl, minlength=S)).astype(np.int64)
    order = np.lexsort((-(x - fl), series))
    ser_sorted = series[order]
    starts = np.searchsorted(ser_sorted, np.arange(S))
    rank = np.arange(len(order)) - starts[ser_sorted]
    add = np.zeros(len(order))
    add[order] = rank < np.maximum(left, 0)[ser_sorted]
    return np.minimum(fl + add, dem).astype(np.int64)
