"""Orchestrator: runs warm-up + the daily loop and assembles observed and latent tables.

Append-only determinism: every draw is keyed by (seed, scope, day, stream) and state is a pure
function of earlier days, so simulate(cfg, n_days=N+1) reproduces days 0..N-1 of simulate(cfg, n_days=N)
exactly. Extending history = re-running to N+1 (a few seconds at full scale) and verifying the prefix hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import WorldConfig
from .dynamics import Engine
from .events import DayModifiers, EventHook, NullHook
from .observe import REASON_NAMES, STATUS_NAMES, Visibility, scrape_status
from .schema import ALL_TABLES, DIMS, LATENT, OBSERVED, coerce
from .seeds import rng_for
from .world import World, build_world

_LISTING_STATUS = np.array(["listed", "unlisted", "unknown"], dtype=object)


@dataclass
class SimResult:
    cfg: WorldConfig
    world: World
    n_days: int
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def observed(self) -> dict[str, pd.DataFrame]:
        return {k: v for k, v in self.tables.items() if k in OBSERVED}

    def latent(self) -> dict[str, pd.DataFrame]:
        return {k: v for k, v in self.tables.items() if k in LATENT}

    def dims(self) -> dict[str, pd.DataFrame]:
        return {k: v for k, v in self.tables.items() if k in DIMS}


def _check_mods(w: World, m: DayModifiers) -> None:
    exp = {"demand_mult": (w.n_cells,), "price_mult": (w.n_cells,), "supply_fill_mult": (w.n_series,),
           "stock_destroy_frac": (w.n_series,), "ad_bid_mult": (w.P, w.L, w.B),
           "listing_suppress": (w.n_cells,), "hide_available": (w.n_cells,), "promo_flag": (w.n_cells,),
           "scrape_blackout": (w.P, w.L)}
    for name, shape in exp.items():
        arr = getattr(m, name)
        if arr is not None and np.shape(arr) != shape:
            raise ValueError(f"DayModifiers.{name} has shape {np.shape(arr)}, expected {shape}")


class _Acc:
    """Collects per-day numpy chunks, then builds typed DataFrames once."""

    def __init__(self) -> None:
        self.chunks: dict[str, list[dict[str, np.ndarray]]] = {}

    def add(self, table: str, **cols: np.ndarray) -> None:
        self.chunks.setdefault(table, []).append(cols)

    def cat(self, table: str, key: str) -> np.ndarray:
        return np.concatenate([c[key] for c in self.chunks[table]])


def build_hook(cfg: WorldConfig, w: World) -> EventHook:
    """Default hook for a config: the planted-event injector if cfg.events.enabled, else the baseline."""
    if cfg.events.enabled:
        from .injector import EventInjector  # local import: injector depends on world/config only
        return EventInjector(w, cfg)
    return NullHook()


def simulate(cfg: WorldConfig, n_days: int | None = None, hook: EventHook | None = None,
             counterfactual: bool = True) -> SimResult:
    """Run the world. With the event injector, also runs counterfactual worlds (baseline, and each E6
    component alone) to fill latent_effects. counterfactual=False skips them (answer key left empty)."""
    cfg.validate()
    n = cfg.n_days if n_days is None else n_days
    if n < 1:
        raise ValueError("n_days must be >= 1")
    w = build_world(cfg)
    hook = hook if hook is not None else build_hook(cfg, w)
    if hasattr(hook, "bind_horizon"):
        hook.bind_horizon(n)
    res = _run(cfg, w, n, hook)
    if counterfactual and getattr(hook, "needs_counterfactuals", False):
        base = _run(cfg, w, n, NullHook())
        comps = {tag: _run(cfg, w, n, hook.restricted(tag)) for tag in hook.component_tags()}
        hook.compute_effects(res, base, comps)
        res.tables["latent_events"] = coerce(hook.latent_events(), "latent_events")
        res.tables["latent_effects"] = coerce(hook.latent_effects(), "latent_effects")
    return res


def _run(cfg: WorldConfig, w: World, n: int, hook: EventHook) -> SimResult:
    eng, vis = Engine(w), Visibility(w)
    st = eng.init_state()
    acc = _Acc()
    focal_cells = np.where(w.sku_is_focal[w.cell_sku])[0]
    unserviced = np.argwhere(w.pair_of < 0)
    W = cfg.inventory.warmup_days
    sov_sum = np.zeros_like(st.sov_prev)
    null = DayModifiers()

    for day in range(-W, n):
        emit = day >= 0
        mods = hook.modifiers(day, cfg.start + timedelta(days=day), w) if emit else null
        _check_mods(w, mods)
        d, dow, season = eng.calendar(day)
        price_pk, promo_active = eng.update_prices(day, st)
        price_cell = eng.cell_prices(price_pk, mods)
        rng = rng_for(cfg.seed, "cell", day, "unlist")
        unlist = rng.random(w.n_cells) < cfg.demand.baseline_unlist_prob
        if mods.listing_suppress is not None:
            unlist = unlist | mods.listing_suppress
        listed_today = ~unlist
        buyable = listed_today if mods.hide_available is None else listed_today & ~mods.hide_available
        promo_cell = promo_active[w.cell_plat, w.cell_sku]
        if mods.promo_flag is not None:
            promo_cell = promo_cell | mods.promo_flag
        lam = eng.demand_rate(dow, season, price_cell, promo_cell, buyable, st, mods)
        dem, sold, avail, onhand = eng.run_day_inventory(day, lam, buyable, st, mods)
        eng.place_orders(day, sold, st)
        slots, sov = vis.run(day, avail, listed_today, price_cell, st.conv_ema, mods)
        prev_last = st.last_avail
        a = cfg.ads.conv_ema_alpha
        st.conv_ema = (1 - a) * st.conv_ema + a * avail.mean(0)
        st.last_avail = avail[3]
        st.sov_prev = sov
        if not emit:
            sov_sum += sov
            if day == -1:
                st.sov_ref = sov_sum / W
            continue

        # ------------------------------------------------------- observation layer
        status, stale = scrape_status(w, day, mods)
        stc = status[w.cell_pair].T                                   # [4,C]
        ok = stc == 0
        prev_true = np.vstack([prev_last[None, :], avail[:3]])
        obs = np.where(stale[w.cell_pair].T, prev_true, avail).astype(np.int8)
        any_ok = ok.any(0)
        code = np.where(~any_ok, 2, np.where(listed_today, 0, 1))
        mrp = w.sku_mrp[w.cell_sku]
        n_c = w.n_cells
        dcol = np.full(n_c, day, dtype=np.int16)
        acc.add("fact_listing_daily", day_index=dcol, platform_id=(w.cell_plat + 1).astype(np.int8),
                pincode_id=w.cell_pin.astype(np.int16), sku_id=w.cell_sku.astype(np.int16),
                obs=obs, ok=ok, price=price_cell.astype(np.int32), mrp=mrp.astype(np.int32),
                promo=promo_cell.astype(np.int8), code=code.astype(np.int8), any_ok=any_ok)

        pr_ = np.repeat(np.arange(w.n_pairs), 4)
        sn_ = np.tile(np.arange(4), w.n_pairs)
        st_flat = status[pr_, sn_]
        acc.add("scrape_log", day_index=np.full(len(pr_), day, np.int16),
                platform_id=(w.pair_plat[pr_] + 1).astype(np.int8), pincode_id=w.pair_pin[pr_].astype(np.int16),
                snapshot=(sn_ + 1).astype(np.int8),
                status=np.array([STATUS_NAMES[int(s)] for s in st_flat], dtype=object),
                reason=np.array([REASON_NAMES[int(s)] for s in st_flat], dtype=object))
        if len(unserviced):
            up = np.repeat(np.arange(len(unserviced)), 4)
            acc.add("scrape_log", day_index=np.full(len(up), day, np.int16),
                    platform_id=(unserviced[up, 0] + 1).astype(np.int8),
                    pincode_id=unserviced[up, 1].astype(np.int16), snapshot=(np.tile(np.arange(4), len(unserviced)) + 1).astype(np.int8),
                    status=np.full(len(up), "unserviceable", dtype=object),
                    reason=np.full(len(up), "not_covered", dtype=object))

        for snap, s in slots.items():
            keep = status[s["pair"], snap - 1] == 0
            m = int(keep.sum())
            acc.add("fact_search_visibility", day_index=np.full(m, day, np.int16),
                    snapshot=np.full(m, snap, np.int8), platform_id=(w.pair_plat[s["pair"][keep]] + 1).astype(np.int8),
                    pincode_id=w.pair_pin[s["pair"][keep]].astype(np.int16), query_id=s["query"][keep].astype(np.int8),
                    position=s["position"][keep].astype(np.int8), sku_id=s["sku"][keep].astype(np.int16),
                    is_sponsored=s["is_sponsored"][keep])

        fm = len(focal_cells)
        acc.add("fact_focal_sellout", day_index=np.full(fm, day, np.int16),
                platform_id=(w.cell_plat[focal_cells] + 1).astype(np.int8),
                pincode_id=w.cell_pin[focal_cells].astype(np.int16), sku_id=w.cell_sku[focal_cells].astype(np.int16),
                units=sold[focal_cells].astype(np.int32), gmv=(sold[focal_cells] * price_cell[focal_cells]).astype(np.int64))
        acc.add("latent_demand", day_index=dcol, platform_id=(w.cell_plat + 1).astype(np.int8),
                pincode_id=w.cell_pin.astype(np.int16), sku_id=w.cell_sku.astype(np.int16),
                true_demand_units=dem.astype(np.int32), sold_units=sold.astype(np.int32),
                lost_units=(dem - sold).astype(np.int32))
        for k in range(4):
            acc.add("latent_stock", day_index=np.full(w.n_series, day, np.int16),
                    snapshot=np.full(w.n_series, k + 1, np.int8), dark_store_id=w.series_store.astype(np.int16),
                    sku_id=w.series_sku.astype(np.int16), on_hand_units=onhand[k].astype(np.int32))
        sp, ss = np.where(stale)
        acc.add("latent_obs_flags", day_index=np.full(len(sp), day, np.int16),
                platform_id=(w.pair_plat[sp] + 1).astype(np.int8), pincode_id=w.pair_pin[sp].astype(np.int16),
                snapshot=(ss + 1).astype(np.int8), was_stale=np.ones(len(sp), dtype=bool))

    return SimResult(cfg=cfg, world=w, n_days=n, tables=_assemble(cfg, w, acc, hook))


def _dates(cfg: WorldConfig, day_index: np.ndarray) -> pd.Series:
    return pd.Series(pd.Timestamp(cfg.start) + pd.to_timedelta(day_index.astype(np.int64), unit="D"))


def _assemble(cfg: WorldConfig, w: World, acc: _Acc, hook: EventHook) -> dict[str, pd.DataFrame]:
    t: dict[str, pd.DataFrame] = {k: coerce(getattr(w, k).copy(), k) for k in DIMS if k != "dim_calendar"}
    # ---- fact_listing_daily (nullable availability: NA == scrape failed, never filled)
    c = acc.chunks["fact_listing_daily"]
    cat = lambda k, ax=0: np.concatenate([x[k] for x in c], axis=ax)  # noqa: E731
    obs, ok, any_ok = cat("obs", 1), cat("ok", 1), cat("any_ok")
    day = cat("day_index")
    df = pd.DataFrame({"day_index": day, "date": _dates(cfg, day), "platform_id": cat("platform_id"),
                       "pincode_id": cat("pincode_id"), "sku_id": cat("sku_id")})
    for k in range(4):
        df[f"snap{k+1}_avail"] = pd.arrays.IntegerArray(obs[k].astype(np.int8), ~ok[k])
    price, mrp = cat("price"), cat("mrp")
    df["price"] = pd.arrays.IntegerArray(price.astype(np.int32), ~any_ok)
    df["mrp"] = mrp
    disc = np.where(any_ok, np.round((mrp - price) / mrp * 100.0, 1), np.nan)
    df["discount_pct"] = disc.astype(np.float32)
    df["promo_flag"] = pd.arrays.IntegerArray(cat("promo"), ~any_ok)
    df["listing_status"] = _LISTING_STATUS[cat("code")]
    df["scrape_ok"] = any_ok
    t["fact_listing_daily"] = coerce(df, "fact_listing_daily")
    # ---- remaining fact/latent tables built directly from concatenated chunks
    for name in ["fact_search_visibility", "fact_focal_sellout", "scrape_log", "latent_demand",
                 "latent_stock", "latent_obs_flags"]:
        spec = ALL_TABLES[name]
        cols = {k: acc.cat(name, k) for k in spec if k not in ("date",)}
        d = pd.DataFrame(cols)
        d["date"] = _dates(cfg, d["day_index"].to_numpy())
        t[name] = coerce(d, name)
    t["dim_calendar"] = coerce(_calendar(cfg, hook), "dim_calendar")
    t["latent_events"] = coerce(hook.latent_events(), "latent_events")
    t["latent_effects"] = coerce(hook.latent_effects(), "latent_effects")
    return t


def _calendar(cfg: WorldConfig, hook: EventHook) -> pd.DataFrame:
    """Public festival calendar: baseline festivals (all zones) plus planned regional festivals."""
    rows = []
    for f in cfg.demand.festivals:
        fd = pd.Timestamp(f.date)
        for off in range(-f.window_days + 1, f.window_days):
            rows.append((fd + pd.Timedelta(days=off), f.name, pd.NA, "baseline"))
    base = pd.DataFrame(rows, columns=["date", "festival_name", "scope_zone_id", "kind"])
    reg = hook.calendar_rows().assign(kind="regional")
    out = pd.concat([base, reg[["date", "festival_name", "scope_zone_id", "kind"]]], ignore_index=True)
    out["scope_zone_id"] = out["scope_zone_id"].astype("Int8")
    return out.sort_values(["date", "festival_name", "scope_zone_id"], na_position="first").reset_index(drop=True)
