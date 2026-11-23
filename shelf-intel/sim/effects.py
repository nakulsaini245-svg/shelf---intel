"""Counterfactual answer key: true effect of each planted event.

Because every random draw is keyed by (seed, scope, day, stream) and events only change *inputs* of the
engine, a baseline run and an event run share all noise: their difference isolates the event's causal
effect (plus its knock-on dynamics, e.g. stockouts outlasting the cause). Effects are summed over the
event's platform x pincode scope for days [first, end + tail_days].

Worlds compared:   full (all events)  vs  base (no events)
                   E6 attribution:  comps["E6a"] / comps["E6b"] = baseline + only that compound component

Metrics (true_delta = event world - baseline world; positive = more):
    delta_sold_units_focal / delta_gmv_focal        focal-brand first-party sell-out
    delta_sold_units_all / delta_true_demand_units_all / delta_lost_units_all     all brands, latent
    delta_oos_snapshot_rate                         share of store x SKU snapshots out of stock, serving stores
    mean_price_delta_pct                            shelf price change on the cells the event manipulates
    delta_{focal,attacker}_sov_pp                   share of ALL top-N search slots (percentage points), SoV events
    delta_{focal,attacker}_sponsored_sov_pp         share of SPONSORED slots (the direct ad effect; all-slot share saturates
                                                    when a query has <= N candidates)
    delta_scrape_failure_rate_pp                    scrape failure share (percentage points), E8
    delta_sold_units_focal_{promo,cliff}_phase      E4 only
    attr_{a,b,interaction}_*                        E6 only; interaction = total - a - b
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .world import World


def _slice(df: pd.DataFrame, ws: int, we: int) -> slice:
    lo, hi = np.searchsorted(df["day_index"].to_numpy(), [ws, we + 1])
    return slice(int(lo), int(hi))


def _scope_mask(df: pd.DataFrame, sl: slice, plats: np.ndarray, pins: np.ndarray) -> np.ndarray:
    d = df.iloc[sl]
    return np.isin(d["platform_id"].to_numpy(), plats + 1) & np.isin(d["pincode_id"].to_numpy(), pins)


def _sum(df: pd.DataFrame, col: str, sl: slice, m: np.ndarray) -> float:
    return float(df.iloc[sl][col].to_numpy(dtype="float64")[m].sum())


def _aligned(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> None:
    for k in keys:
        if not np.array_equal(a[k].to_numpy(), b[k].to_numpy()):
            raise RuntimeError(f"counterfactual tables are not row-aligned on {k}")


def _share(vis: pd.DataFrame, w: World, ws: int, we: int, plats, pins, brand: int,
           sponsored_only: bool = False) -> float:
    sl = _slice(vis, ws, we)
    m = _scope_mask(vis, sl, plats, pins)
    if sponsored_only:
        m = m & vis.iloc[sl]["is_sponsored"].to_numpy()
    b = w.sku_brand[vis.iloc[sl]["sku_id"].to_numpy()][m]
    return float((b == brand).mean() * 100.0) if len(b) else float("nan")


def _std(ev, X, Y, w: World, ws: int, we: int, phases: bool = True) -> dict[str, float]:
    """Standard effect metrics of event `ev`: world X (with the effect) versus world Y (without)."""
    plats = np.arange(w.P) if ev.platform is None else np.array([ev.platform])
    pins = np.asarray(ev.pincodes)
    out: dict[str, float] = {}
    fx, fy = X.tables["fact_focal_sellout"], Y.tables["fact_focal_sellout"]
    _aligned(fx, fy, ["day_index", "platform_id", "pincode_id", "sku_id"])
    sl = _slice(fx, ws, we)
    m = _scope_mask(fx, sl, plats, pins)
    out["delta_sold_units_focal"] = _sum(fx, "units", sl, m) - _sum(fy, "units", sl, m)
    out["delta_gmv_focal"] = _sum(fx, "gmv", sl, m) - _sum(fy, "gmv", sl, m)
    if phases and ev.type == "E4":
        for name, (a, b) in {"promo": (ev.first, ev.start - 1), "cliff": (ev.start, ev.end)}.items():
            s2 = _slice(fx, a, b)
            m2 = _scope_mask(fx, s2, plats, pins)
            out[f"delta_sold_units_focal_{name}_phase"] = _sum(fx, "units", s2, m2) - _sum(fy, "units", s2, m2)
    dx, dy = X.tables["latent_demand"], Y.tables["latent_demand"]
    _aligned(dx, dy, ["day_index", "platform_id", "pincode_id", "sku_id"])
    sl = _slice(dx, ws, we)
    m = _scope_mask(dx, sl, plats, pins)
    for col, name in [("sold_units", "sold_units_all"), ("true_demand_units", "true_demand_units_all"),
                      ("lost_units", "lost_units_all")]:
        out[f"delta_{name}"] = _sum(dx, col, sl, m) - _sum(dy, col, sl, m)
    # stock: serving dark stores of the scope
    stores = np.unique(w.pair_store[plats][:, pins])
    stores = stores[stores >= 0]
    sx, sy = X.tables["latent_stock"], Y.tables["latent_stock"]
    sl = _slice(sx, ws, we)
    ms = np.isin(sx.iloc[sl]["dark_store_id"].to_numpy(), stores)
    ox = (sx.iloc[sl]["on_hand_units"].to_numpy()[ms] == 0).mean() if ms.any() else np.nan
    oy = (sy.iloc[sl]["on_hand_units"].to_numpy()[ms] == 0).mean() if ms.any() else np.nan
    out["delta_oos_snapshot_rate"] = float(ox - oy)
    if len(ev.cells):  # shelf price change on manipulated cells
        lx, ly = X.tables["fact_listing_daily"], Y.tables["fact_listing_daily"]
        days = np.arange(ws, we + 1)
        idx = (days[:, None] * w.n_cells + ev.cells[None, :]).ravel()
        px = lx["price"].to_numpy(dtype="float64", na_value=np.nan)[idx]
        py = ly["price"].to_numpy(dtype="float64", na_value=np.nan)[idx]
        ok = np.isfinite(px) & np.isfinite(py)
        out["mean_price_delta_pct"] = float(((px[ok] - py[ok]) / py[ok]).mean() * 100.0) if ok.any() else float("nan")
    if ev.mech == "sov" or ev.type == "E6":
        vx, vy = X.tables["fact_search_visibility"], Y.tables["fact_search_visibility"]
        focal = next(i for i, b in enumerate(X.cfg.brands) if b.is_focal)
        out["delta_focal_sov_pp"] = _share(vx, w, ws, we, plats, pins, focal) - _share(vy, w, ws, we, plats, pins, focal)
        out["delta_attacker_sov_pp"] = (_share(vx, w, ws, we, plats, pins, ev.brands[0])
                                        - _share(vy, w, ws, we, plats, pins, ev.brands[0]))
        for tag, br in [("focal", focal), ("attacker", ev.brands[0])]:   # direct ad effect: sponsored slots only
            out[f"delta_{tag}_sponsored_sov_pp"] = (_share(vx, w, ws, we, plats, pins, br, True)
                                                    - _share(vy, w, ws, we, plats, pins, br, True))
    if ev.mech == "blackout":
        gx, gy = X.tables["scrape_log"], Y.tables["scrape_log"]
        sl = _slice(gx, ws, we)
        mg = _scope_mask(gx, sl, plats, pins)
        fx_ = (gx.iloc[sl]["status"].to_numpy()[mg] == "failed").mean()
        fy_ = (gy.iloc[sl]["status"].to_numpy()[mg] == "failed").mean()
        out["delta_scrape_failure_rate_pp"] = float((fx_ - fy_) * 100.0)
    return out


def compute_effects(inj, full, base, comps) -> pd.DataFrame:
    w, tail = inj.w, inj.cfg.events.tail_days
    rows: list[tuple] = []
    for ev in inj.plan:
        if ev.first > full.n_days - 1:
            continue
        ws, we = ev.first, min(ev.end + tail, full.n_days - 1)

        def add(metrics: dict[str, float], prefix: str = "") -> None:
            for k, v in metrics.items():
                rows.append((ev.event_id, prefix + k, float(v), ws, we))

        if ev.type == "E6":                      # compound: total + attribution
            tot = _std(ev, full, base, w, ws, we)
            a = _std(ev, comps["E6a"], base, w, ws, we)
            b = _std(ev, comps["E6b"], base, w, ws, we)
            add(tot)
            for k in ["delta_sold_units_focal", "delta_gmv_focal", "delta_lost_units_all"]:
                add({k: a[k]}, "attr_a_")
                add({k: b[k]}, "attr_b_")
                add({k: tot[k] - a[k] - b[k]}, "attr_interaction_")
        elif ev.type == "E6a":                   # mechanism row: price effect of the undercut alone
            add({"mean_price_delta_pct": _std(ev, comps["E6a"], base, w, ws, we)["mean_price_delta_pct"]})
        elif ev.type == "E6b":
            s = _std(ev, comps["E6b"], base, w, ws, we)
            add({k: s[k] for k in ["delta_focal_sov_pp", "delta_attacker_sov_pp",
                                   "delta_focal_sponsored_sov_pp", "delta_attacker_sponsored_sov_pp"]})
        else:
            add(_std(ev, full, base, w, ws, we))
    return pd.DataFrame(rows, columns=["event_id", "metric", "true_delta", "window_start", "window_end"])
