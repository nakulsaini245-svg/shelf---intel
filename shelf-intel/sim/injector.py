"""Planted-incident injector: E1..E10 as a deterministic, conflict-free plan + per-day mechanics.

The plan is a pure function of (seed, config.events, world); it never depends on how many days are
simulated, so extending history cannot move an event (append-only determinism).

Mechanics are expressed only through `DayModifiers` (see events.py), so the baseline engine is untouched.
Counterfactual effects (the answer key) are computed in effects.py by re-running worlds with and without
the events.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .config import WorldConfig
from .events import DayModifiers
from .schema import LATENT, coerce
from .seeds import WORLD_DAY, rng_for
from .world import World

TYPES = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
HARD_NEGATIVE = {"E4", "E8", "E10"}   # real or apparent change that should NOT be raised as an incident


def tier_name(d: float) -> str:
    return "easy" if d <= 0.25 else "hard" if d >= 0.75 else "medium"


@dataclass
class PlannedEvent:
    type: str
    name: str
    start: int
    end: int
    first: int                    # first/last day the mechanics act (E4 promo, E10 pre-stock start earlier)
    last: int
    difficulty: float
    mech: str                     # supply|outage|undercut|promo_end|sov|phantom|blackout|drift|festival|none
    platform: int | None          # 0-based; None = all platforms
    zones: list[int]
    pincodes: np.ndarray
    stores: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    skus: np.ndarray | None = None
    brands: list[int] = field(default_factory=list)
    magnitude: float = 0.0
    unit: str = ""
    params: dict = field(default_factory=dict)
    cells: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    arr: dict = field(default_factory=dict)       # precomputed per-event arrays
    event_id: int = 0
    parent: int | None = None
    component: str | None = None

    @property
    def hard_negative(self) -> bool:
        return self.type in HARD_NEGATIVE


# ------------------------------------------------------------------------------------- planning helpers
def _kn(cfg: WorldConfig, t: str, name: str, d: float) -> float:
    lo, hi = cfg.events.knobs[t][name]
    return lo + (hi - lo) * d


def _ri(x: float) -> int:
    return max(1, int(round(x)))


def _pins(w: World, zones) -> np.ndarray:
    return np.where(np.isin(w.pin_zone, zones))[0]


def _cells(w: World, p: int | None, pins: np.ndarray, skus: np.ndarray | None) -> np.ndarray:
    m = np.isin(w.cell_pin, pins)
    if p is not None:
        m &= w.cell_plat == p
    if skus is not None:
        m &= np.isin(w.cell_sku, skus)
    return np.where(m)[0]


def _skus(w: World, p: int, q: int | None = None, brands=None, focal: bool | None = None) -> np.ndarray:
    m = w.listed[p].copy()
    if q is not None:
        m &= w.sku_sub == q
    if brands is not None:
        m &= np.isin(w.sku_brand, brands)
    if focal is not None:
        m &= w.sku_is_focal == focal
    return np.where(m)[0]


def _zlabel(w: World, z: int) -> str:
    return str(w.dim_location.loc[w.dim_location["zone_id"] == z, "zone_label"].iloc[0])


def _base(w, cfg, typ, name, start, end, d, mech, platform, zones, pins, **kw) -> PlannedEvent:
    return PlannedEvent(type=typ, name=name, start=start, end=end, first=kw.pop("first", start),
                        last=kw.pop("last", end), difficulty=d, mech=mech, platform=platform,
                        zones=[int(z) for z in zones], pincodes=np.asarray(pins), **kw)


# --------------------------------------------------------------------------------------- type builders
def _b_E1(rng, w, cfg, d, start):
    dur, fill = _ri(_kn(cfg, "E1", "duration", d)), _kn(cfg, "E1", "fill_mult", d)
    share, jit = _kn(cfg, "E1", "store_share", d), _kn(cfg, "E1", "store_jitter", d)
    p, z = int(rng.integers(w.P)), int(rng.integers(len(set(w.pin_zone))))
    zs = np.where((w.store_plat == p) & (w.store_zone == z))[0]
    if len(zs) == 0:
        return None
    aff = np.sort(rng.choice(zs, size=min(len(zs), _ri(share * len(zs))), replace=False))
    mult = np.clip(fill * rng.uniform(1 - jit, 1 + jit, len(aff)), 0.0, 1.0)   # variance inside the zone
    sm = np.zeros(w.D)
    sm[aff] = mult
    series = np.where(np.isin(w.series_store, aff))[0]
    ev = _base(w, cfg, "E1", f"Supply shock: {_zlabel(w, z)} ({cfg.platforms[p].id})", start, start + dur - 1, d,
               "supply", p, [z], _pins(w, [z]), stores=aff, series=series, magnitude=fill,
               unit="inbound_fill_fraction",
               params={"store_fill_mult": {int(s): round(float(m), 3) for s, m in zip(aff, mult)}})
    ev.arr["mult"] = sm[w.series_store[series]]
    return [ev]


def _b_E2(rng, w, cfg, d, start):
    dur, sev = _ri(_kn(cfg, "E2", "duration", d)), _kn(cfg, "E2", "severity", d)
    p = int(rng.integers(w.P))
    s = int(rng.choice(np.where(w.store_plat == p)[0]))
    pins = np.where(w.pair_store[p] == s)[0]
    if len(pins) == 0:
        return None
    z = int(w.store_zone[s])
    ev = _base(w, cfg, "E2", f"Dark-store outage: {w.dim_dark_store.loc[s, 'name']}", start, start + dur - 1, d,
               "outage", p, [z], pins, stores=np.array([s]), series=np.where(w.series_store == s)[0],
               magnitude=sev, unit="stock_loss_fraction")
    return [ev]


def _undercut_pick(rng, w, p, n_brands):
    q = int(rng.integers(w.Q))
    cand = [b for b in range(w.B) if not w.cfg.brands[b].is_focal and len(_skus(w, p, q, [b])) > 0]
    if not cand:
        return None
    br = [int(b) for b in rng.choice(cand, size=min(n_brands, len(cand)), replace=False)]
    return q, br


def _b_E3(rng, w, cfg, d, start):
    dur, cut = _ri(_kn(cfg, "E3", "duration", d)), _kn(cfg, "E3", "cut_pct", d)
    nb, nz, lag = _ri(_kn(cfg, "E3", "n_brands", d)), _ri(_kn(cfg, "E3", "n_zones", d)), _ri(_kn(cfg, "E3", "lag_days", d))
    p = int(rng.integers(w.P))
    zones = rng.choice(len(set(w.pin_zone)), size=nz, replace=False)
    pick = _undercut_pick(rng, w, p, nb)
    if pick is None:
        return None
    q, br = pick
    skus, pins = _skus(w, p, q, br), _pins(w, zones)
    ev = _base(w, cfg, "E3", f"Competitor undercut: {w.subcats[q]} ({cfg.platforms[p].id})", start, start + dur - 1,
               d, "undercut", p, zones, pins, skus=skus, brands=br, cells=_cells(w, p, pins, skus),
               magnitude=cut, unit="price_cut_fraction",
               params={"subcategory": w.subcats[q], "lag_days": lag, "brands": [cfg.brands[b].name for b in br]})
    return [ev]


def _b_E4(rng, w, cfg, d, start):
    pdays, depth = _ri(_kn(cfg, "E4", "promo_days", d)), _kn(cfg, "E4", "depth", d)
    dip, ddays, nz = _kn(cfg, "E4", "dip", d), _ri(_kn(cfg, "E4", "dip_days", d)), _ri(_kn(cfg, "E4", "n_zones", d))
    p, q = int(rng.integers(w.P)), int(rng.integers(w.Q))
    zones = rng.choice(len(set(w.pin_zone)), size=nz, replace=False)
    skus, pins = _skus(w, p, q, focal=True), _pins(w, zones)
    if len(skus) == 0 or start - pdays < cfg.events.min_start_day:
        return None
    ev = _base(w, cfg, "E4", f"Own promo ends: {w.subcats[q]} ({cfg.platforms[p].id})", start, start + ddays - 1, d,
               "promo_end", p, zones, pins, skus=skus, first=start - pdays, cells=_cells(w, p, pins, skus),
               magnitude=dip, unit="post_promo_demand_dip",
               params={"subcategory": w.subcats[q], "promo_start_day": start - pdays, "promo_end_day": start - 1,
                       "promo_depth": round(depth, 3), "explained_by_observed_promo_flag": True})
    ev.arr["depth"] = depth
    return [ev]


def _b_E5(rng, w, cfg, d, start):
    dur, mult, nz = _ri(_kn(cfg, "E5", "duration", d)), _kn(cfg, "E5", "bid_mult", d), _ri(_kn(cfg, "E5", "n_zones", d))
    p = int(rng.integers(w.P))
    zones = rng.choice(len(set(w.pin_zone)), size=nz, replace=False)
    cand = [i for i in range(w.B) if not cfg.brands[i].is_focal and len(_skus(w, p, brands=[i])) > 0]
    if not cand:
        return None
    b = int(rng.choice(cand))
    ev = _base(w, cfg, "E5", f"Ad-SoV takeover: {cfg.brands[b].name} ({cfg.platforms[p].id})", start, start + dur - 1, d,
               "sov", p, zones, _pins(w, zones), brands=[b], magnitude=mult, unit="sponsored_bid_multiplier")
    return [ev]


def _b_E6(rng, w, cfg, d, start):
    dur, cut = _ri(_kn(cfg, "E6", "duration", d)), _kn(cfg, "E6", "cut_pct", d)
    bid, off = _kn(cfg, "E6", "bid_mult", d), int(round(_kn(cfg, "E6", "sov_offset_days", d)))
    p, z = int(rng.integers(w.P)), int(rng.integers(len(set(w.pin_zone))))
    pick = _undercut_pick(rng, w, p, 1)
    if pick is None:
        return None
    q, br = pick
    skus, pins, end = _skus(w, p, q, br), _pins(w, [z]), start + dur - 1
    nm = f"{cfg.brands[br[0]].name}, {_zlabel(w, z)} ({cfg.platforms[p].id})"
    parent = _base(w, cfg, "E6", f"Compound undercut + SoV surge: {nm}", start, end, d, "none", p, [z], pins,
                   skus=skus, brands=br, magnitude=cut, unit="price_cut_fraction",
                   params={"subcategory": w.subcats[q], "bid_mult": round(bid, 2), "sov_offset_days": off})
    a = _base(w, cfg, "E6", f"E6a undercut: {nm}", start, end, d, "undercut", p, [z], pins, skus=skus, brands=br,
              cells=_cells(w, p, pins, skus), magnitude=cut, unit="price_cut_fraction",
              params={"subcategory": w.subcats[q], "lag_days": 2}, component="E6a")
    b = _base(w, cfg, "E6", f"E6b SoV surge: {nm}", start + off, end, d, "sov", p, [z], pins, brands=br,
              magnitude=bid, unit="sponsored_bid_multiplier", component="E6b")
    a.type, b.type = "E6a", "E6b"
    return [parent, a, b]


def _b_E7(rng, w, cfg, d, start):
    dur, share, vis = _ri(_kn(cfg, "E7", "duration", d)), _kn(cfg, "E7", "sku_share", d), _kn(cfg, "E7", "visible_frac", d)
    p, z = int(rng.integers(w.P)), int(rng.integers(len(set(w.pin_zone))))
    listed = np.where(w.listed[p])[0]
    skus = np.sort(rng.choice(listed, size=_ri(share * len(listed)), replace=False))
    pins = _pins(w, [z])
    cells = _cells(w, p, pins, skus)
    if len(cells) == 0:
        return None
    ev = _base(w, cfg, "E7", f"Phantom stockout: {_zlabel(w, z)} ({cfg.platforms[p].id})", start, start + dur - 1, d,
               "phantom", p, [z], pins, skus=skus, cells=cells, magnitude=share, unit="sku_share",
               stores=np.where((w.store_plat == p) & (w.store_zone == z))[0],
               params={"visible_frac": round(vis, 3)})
    ev.arr["visible"] = rng.random(len(cells)) < vis
    return [ev]


def _b_E8(rng, w, cfg, d, start):
    dur, share = _ri(_kn(cfg, "E8", "duration", d)), _kn(cfg, "E8", "pincode_share", d)
    p, z = int(rng.integers(w.P)), int(rng.integers(len(set(w.pin_zone))))
    zp = _pins(w, [z])
    pins = np.sort(rng.choice(zp, size=_ri(share * len(zp)), replace=False))
    return [_base(w, cfg, "E8", f"Scrape blackout: {_zlabel(w, z)} ({cfg.platforms[p].id})", start, start + dur - 1, d,
                  "blackout", p, [z], pins, magnitude=share, unit="pincode_share")]


def _b_E9(rng, w, cfg, d, start):
    dur, tot, nz = _ri(_kn(cfg, "E9", "duration", d)), _kn(cfg, "E9", "total_drift", d), _ri(_kn(cfg, "E9", "n_zones", d))
    p = int(rng.integers(w.P))
    zones = rng.choice(len(set(w.pin_zone)), size=nz, replace=False)
    q = int(rng.integers(w.Q))
    br = [b for b in range(w.B) if not cfg.brands[b].is_focal]
    skus, pins = _skus(w, p, q, br), _pins(w, zones)
    if len(skus) == 0:
        return None
    return [_base(w, cfg, "E9", f"Slow price drift: {w.subcats[q]} ({cfg.platforms[p].id})", start, start + dur - 1, d,
                  "drift", p, zones, pins, skus=skus, brands=br, cells=_cells(w, p, pins, skus), magnitude=tot,
                  unit="total_price_drift_fraction", params={"subcategory": w.subcats[q]})]


def _b_E10(rng, w, cfg, d, start):
    dur, surge = _ri(_kn(cfg, "E10", "duration", d)), _kn(cfg, "E10", "surge", d)
    nz, pre, ns = _ri(_kn(cfg, "E10", "n_zones", d)), _kn(cfg, "E10", "prestock_mult", d), _ri(_kn(cfg, "E10", "n_subcats", d))
    zones = rng.choice(len(set(w.pin_zone)), size=nz, replace=False)
    qs = np.sort(rng.choice(w.Q, size=min(ns, w.Q), replace=False))
    pins = _pins(w, zones)
    skus = np.where(np.isin(w.sku_sub, qs))[0]
    series = np.where(np.isin(w.store_zone[w.series_store], zones) & np.isin(w.sku_sub[w.series_sku], qs))[0]
    label = "/".join(_zlabel(w, z) for z in sorted(zones))
    return [_base(w, cfg, "E10", f"Regional festival ({label})", start, start + dur - 1, d, "festival", None, zones,
                  pins, skus=skus, first=start - 1, cells=_cells(w, None, pins, skus), series=series,
                  magnitude=surge, unit="peak_demand_surge",
                  params={"subcategories": [w.subcats[q] for q in qs], "prestock_mult": pre})]


_BUILD = {"E1": _b_E1, "E2": _b_E2, "E3": _b_E3, "E4": _b_E4, "E5": _b_E5, "E6": _b_E6, "E7": _b_E7,
          "E8": _b_E8, "E9": _b_E9, "E10": _b_E10}


def _conflict(a: PlannedEvent, b: PlannedEvent, pad: int) -> bool:
    if a.first > b.last + pad or b.first > a.last + pad:
        return False
    if a.platform is not None and b.platform is not None and a.platform != b.platform:
        return False
    return bool(set(a.zones) & set(b.zones))


def plan_events(w: World, cfg: WorldConfig) -> list[PlannedEvent]:
    ec = cfg.events
    rng = rng_for(cfg.seed, "events", WORLD_DAY, "plan")
    pad = ec.tail_days + ec.min_gap_days
    plan: list[PlannedEvent] = []
    nxt = 1
    for t in TYPES:
        n = ec.counts.get(t, 0)
        off = int(rng.integers(len(ec.tiers)))
        for i in range(n):
            d = ec.tiers[(i + off) % len(ec.tiers)]
            for _ in range(2000):
                start = int(rng.integers(ec.min_start_day, ec.horizon_days - ec.tail_days - 1))
                evs = _BUILD[t](rng, w, cfg, d, start)
                if not evs:
                    continue
                head = evs[0]
                if head.first < ec.min_start_day or head.last + ec.tail_days > ec.horizon_days - 1:
                    continue
                if any(_conflict(head, o, pad) for o in plan if o.parent is None):
                    continue
                for k, ev in enumerate(evs):
                    ev.event_id = nxt
                    nxt += 1
                    ev.parent = None if k == 0 else evs[0].event_id
                plan.extend(evs)
                break
            else:
                raise RuntimeError(f"could not place {t} (difficulty {d}); relax counts or widen the horizon")
    return plan


# ------------------------------------------------------------------------------------------ mechanics
class _Mods:
    """Lazily-allocated DayModifiers; events compose multiplicatively (flags OR together)."""

    def __init__(self, w: World):
        self.w, self.a = w, {}

    def _get(self, name, shape, fill, dtype=float):
        if name not in self.a:
            self.a[name] = np.full(shape, fill, dtype=dtype)
        return self.a[name]

    def demand(self): return self._get("demand_mult", self.w.n_cells, 1.0)
    def price(self): return self._get("price_mult", self.w.n_cells, 1.0)
    def supply(self): return self._get("supply_fill_mult", self.w.n_series, 1.0)
    def destroy(self): return self._get("stock_destroy_frac", self.w.n_series, 0.0)
    def ad(self): return self._get("ad_bid_mult", (self.w.P, self.w.L, self.w.B), 1.0)
    def suppress(self): return self._get("listing_suppress", self.w.n_cells, False, bool)
    def hide(self): return self._get("hide_available", self.w.n_cells, False, bool)
    def promo(self): return self._get("promo_flag", self.w.n_cells, False, bool)
    def blackout(self): return self._get("scrape_blackout", (self.w.P, self.w.L), False, bool)

    def out(self) -> DayModifiers:
        return DayModifiers(**self.a)


def _apply(ev: PlannedEvent, day: int, m: _Mods) -> None:
    t = ev.mech
    if t == "supply":
        m.supply()[ev.series] *= ev.arr["mult"]
    elif t == "outage":
        if day == ev.start:
            d = m.destroy()
            d[ev.series] = np.maximum(d[ev.series], ev.magnitude)
        m.supply()[ev.series] *= 1.0 - ev.magnitude
    elif t == "undercut":
        ramp = min(1.0, (day - ev.start + 1) / ev.params["lag_days"])
        m.price()[ev.cells] *= 1.0 - ev.magnitude * ramp
    elif t == "promo_end":
        if day < ev.start:
            m.price()[ev.cells] *= 1.0 - ev.arr["depth"]
            m.promo()[ev.cells] = True
        else:
            m.demand()[ev.cells] *= 1.0 - ev.magnitude
    elif t == "sov":
        m.ad()[ev.platform, ev.pincodes, ev.brands[0]] *= ev.magnitude
    elif t == "phantom":
        vis = ev.arr["visible"]
        m.suppress()[ev.cells[vis]] = True
        m.hide()[ev.cells[~vis]] = True
    elif t == "blackout":
        m.blackout()[ev.platform, ev.pincodes] = True
    elif t == "drift":
        frac = min(1.0, (day - ev.start + 1) / (ev.end - ev.start + 1))
        m.price()[ev.cells] *= 1.0 - ev.magnitude * frac
    elif t == "festival":
        if ev.start <= day <= ev.end:
            tri = 1.0 - abs(day - (ev.start + ev.end) / 2.0) / ((ev.end - ev.start + 1) / 2.0 + 0.5)
            m.demand()[ev.cells] *= 1.0 + ev.magnitude * tri
        m.supply()[ev.series] *= ev.params["prestock_mult"]


def _csv(a) -> str:
    return ",".join(str(int(x)) for x in a)


def _json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))


class EventInjector:
    """EventHook implementation. `include` restricts which events act (used for counterfactual runs)."""

    needs_counterfactuals = True

    def __init__(self, world: World, cfg: WorldConfig, plan: list[PlannedEvent] | None = None, include=None):
        self.w, self.cfg = world, cfg
        self.plan = plan if plan is not None else plan_events(world, cfg)
        self.include = include or (lambda ev: ev.mech != "none")
        self.n_days = cfg.n_days
        self._effects = pd.DataFrame()

    # ---- counterfactual plumbing
    def bind_horizon(self, n_days: int) -> None:
        self.n_days = n_days

    def restricted(self, tag: str) -> "EventInjector":
        return EventInjector(self.w, self.cfg, self.plan, include=lambda ev: ev.component == tag)

    def component_tags(self) -> list[str]:
        return sorted({e.component for e in self.plan if e.component})

    def compute_effects(self, full, base, comps) -> None:
        from .effects import compute_effects
        self._effects = compute_effects(self, full, base, comps)

    # ---- EventHook
    def modifiers(self, day: int, day_date: date, world: World) -> DayModifiers:
        m = _Mods(world)
        for ev in self.plan:
            if ev.first <= day <= ev.last and ev.mech != "none" and self.include(ev) and (
                    ev.mech != "sov" or day >= ev.start):
                _apply(ev, day, m)
        return m.out()

    def latent_events(self) -> pd.DataFrame:
        rows = []
        for e in self.plan:
            if e.start >= self.n_days:
                continue
            rows.append(dict(
                event_id=e.event_id, type=e.type, name=e.name, start_day=e.start, end_day=e.end,
                tier=tier_name(e.difficulty), difficulty=e.difficulty,
                scope_platform="ALL" if e.platform is None else self.cfg.platforms[e.platform].id,
                scope_zones=_csv(e.zones), scope_pincodes=_csv(e.pincodes), scope_stores=_csv(e.stores),
                scope_skus="ALL" if e.skus is None else _csv(e.skus), scope_brands=_csv(e.brands),
                magnitude=float(e.magnitude), magnitude_unit=e.unit, causal_parent_event_id=e.parent,
                is_hard_negative=e.hard_negative, should_alert=not e.hard_negative and e.type != "E6",
                params=_json(e.params)))
        if not rows:
            return NullHookEmpty()
        df = pd.DataFrame(rows)
        # the E6 parent is the compound incident that should alert; its children are mechanism rows
        df.loc[df["type"] == "E6", "should_alert"] = True
        return coerce(df, "latent_events")

    def latent_effects(self) -> pd.DataFrame:
        if self._effects.empty:
            return coerce(pd.DataFrame({c: pd.Series(dtype="object") for c in LATENT["latent_effects"]}),
                          "latent_effects")
        return coerce(self._effects, "latent_effects")

    def calendar_rows(self) -> pd.DataFrame:
        rows = []
        for e in self.plan:
            if e.mech == "festival":
                for day in range(e.start, e.end + 1):
                    for z in e.zones:
                        rows.append((pd.Timestamp(self.cfg.start) + pd.Timedelta(days=day), e.name, z))
        df = pd.DataFrame(rows, columns=["date", "festival_name", "scope_zone_id"])
        df["date"] = df["date"].astype("datetime64[ns]")
        df["festival_name"] = df["festival_name"].astype("string")
        df["scope_zone_id"] = df["scope_zone_id"].astype("Int8")
        return df


def NullHookEmpty() -> pd.DataFrame:
    return coerce(pd.DataFrame({c: pd.Series(dtype="object") for c in LATENT["latent_events"]}), "latent_events")
