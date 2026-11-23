"""Revenue-at-Risk & Action Prioritization Ranker.

Input: incidents (ml.incidents) + conformal bands of the focal first-party sell-out series + observed tables.
For each incident (platform x zone x span):
  1. lost units  = signed sum of (expected - actual) focal units over [first-(persist-1), decision_day], plus a projection
                   of the current daily shortfall over `horizon_days` more days. Interval from the conformal half-widths
                   (sqrt(n) accumulation, an approximation).
  2. lost Rs     = lost units x the focal brand's observed trailing price per unit in that platform x zone.
  3. action      = inferred from which signals fired (availability -> supply, price -> price-match, ad share -> ad boost).
  4. cost        = assumed intervention cost (ACTIONS below; ASSUMPTIONS, not measured). value = recoverable_fraction x lost Rs.
  5. priority    = net benefit = value - cost (ROI = value/cost). Net<=0 -> "monitor". `net_benefit_low` uses the lower bound.
Only observed/public data is used. Drill-down lists the SKUs and pincodes with the largest focal shortfall.
All Rs figures are synthetic-world numbers; cost and recoverability parameters are illustrative assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import Signals


@dataclass(frozen=True)
class Action:
    name: str
    fixed_cost: float            # Rs per incident
    daily_cost: float            # Rs per day of intervention
    gmv_cost_pct: float          # Rs cost as % of the zone's expected daily focal GMV per intervention day
    recoverable: float           # share of the projected loss the action is assumed to win back


ACTIONS = {
    "supply": Action("expedite replenishment / rebalance stock", 4000.0, 800.0, 0.0, 0.60),
    "price": Action("match competitor price on affected subcategory", 1000.0, 0.0, 8.0, 0.45),
    "ads": Action("raise sponsored bids on defended queries", 1500.0, 3000.0, 0.0, 0.40),
}
SIGNAL_TO_ACTION = [("avail", "supply"), ("price", "price"), ("share", "ads"), ("focal_units", "supply")]  # priority order


@dataclass
class RankerCfg:
    horizon_days: int = 4          # assumed remaining duration of an incident still running at decision time
    decision_lag: int = 1          # decide this many days after the first alarm
    persist: int = 2
    trailing_days: int = 14
    actions: dict = field(default_factory=lambda: dict(ACTIONS))


def infer_action(signals: str) -> str:
    s = set(signals.split(","))
    for sig, act in SIGNAL_TO_ACTION:
        if sig in s:
            return act
    return "supply"


def _focal_index(sig: Signals) -> dict[tuple[int, int], int]:
    m = sig.panels["focal_units"].meta
    return {(int(p), int(z)): i for i, (p, z) in enumerate(zip(m.platform_idx, m.zone_id))}


def _price_per_unit(obs: dict, loc: pd.DataFrame) -> pd.DataFrame:
    f = obs["fact_focal_sellout"].merge(loc[["pincode_id", "zone_id"]], on="pincode_id")
    g = f.groupby(["platform_id", "zone_id", "day_index"], as_index=False)[["units", "gmv"]].sum()
    g["platform_idx"] = g["platform_id"] - 1
    return g


def drilldown(f: pd.DataFrame, plat: int, zone: int, d0: int, d1: int, k: int = 3) -> tuple[str, str]:
    """Top-k SKUs and pincodes by focal shortfall vs the same days 7 and 14 days earlier (observed data only)."""
    sub = f[(f["platform_id"] == plat + 1) & (f["zone_id"] == zone)]
    cur = sub[sub["day_index"].between(d0, d1)]
    ref = sub[sub["day_index"].between(d0 - 7, d1 - 7)].assign(w=0.5)
    ref2 = sub[sub["day_index"].between(d0 - 14, d1 - 14)].assign(w=0.5)
    ref = pd.concat([ref, ref2])
    out = []
    for key in ("sku_id", "pincode_id"):
        a = cur.groupby(key)["units"].sum()
        b = (ref["units"] * ref["w"]).groupby(ref[key]).sum()
        short = (b.reindex(a.index.union(b.index)).fillna(0) - a.reindex(a.index.union(b.index)).fillna(0)).sort_values(ascending=False)
        out.append(";".join(f"{int(i)}:{v:.0f}" for i, v in short.head(k).items() if v > 0))
    return out[0], out[1]


def rank_incidents(inc: pd.DataFrame, bands: dict, sig: Signals, obs: dict, cfg: RankerCfg | None = None) -> pd.DataFrame:
    cfg = cfg or RankerCfg()
    E, q, _ = bands["focal_units"]
    x = sig.panels["focal_units"].values
    idx = _focal_index(sig)
    loc = obs["dim_location"]
    ppu = _price_per_unit(obs, loc)
    fz = obs["fact_focal_sellout"].merge(loc[["pincode_id", "zone_id"]], on="pincode_id")
    rows = []
    for r in inc.itertuples():
        i = idx.get((int(r.platform_idx), int(r.zone_id)))
        if i is None:
            continue
        T = x.shape[1]
        d0, dec = max(0, int(r.first_day) - (cfg.persist - 1)), min(T - 1, int(r.first_day) + cfg.decision_lag)
        dec = max(dec, int(r.last_day)) if int(r.last_day) <= dec else dec          # online: nothing after decision day
        days = np.arange(d0, dec + 1)
        short = E[i, days] - x[i, days]
        qq = np.nan_to_num(q[i, days], nan=0.0)
        ok = ~np.isnan(short)
        n = max(int(ok.sum()), 1)
        realized = max(0.0, float(np.nansum(short)))
        spread = float(np.sqrt(np.nansum(qq[ok] ** 2)))
        rate = max(0.0, float(np.nanmean(short[-2:]))) if ok.any() else 0.0
        rate_hi = max(0.0, float(np.nanmean(short[-2:] + qq[-2:]))) if ok.any() else 0.0
        units = realized + rate * cfg.horizon_days
        units_lo = max(0.0, realized - spread)
        units_hi = realized + spread + rate_hi * cfg.horizon_days
        g = ppu[(ppu.platform_idx == r.platform_idx) & (ppu.zone_id == r.zone_id)
                & ppu.day_index.between(d0 - cfg.trailing_days, d0 - 1)]
        price = float(g.gmv.sum() / g.units.sum()) if g.units.sum() > 0 else float("nan")
        daily_gmv = float(g.gmv.sum() / max(1, g.day_index.nunique())) if len(g) else 0.0
        act = cfg.actions[infer_action(r.signals)]
        days_act = cfg.horizon_days + cfg.decision_lag
        cost = act.fixed_cost + act.daily_cost * days_act + act.gmv_cost_pct / 100.0 * daily_gmv * days_act
        loss_rs, lo_rs, hi_rs = (units * price, units_lo * price, units_hi * price) if price == price else (0.0, 0.0, 0.0)
        sk, pc = drilldown(fz, int(r.platform_idx), int(r.zone_id), d0, dec)
        rows.append(dict(incident_id=int(r.incident_id), platform_idx=int(r.platform_idx), zone_id=int(r.zone_id),
                         first_day=int(r.first_day), last_day=int(r.last_day), decision_day=int(dec), signals=r.signals,
                         action=infer_action(r.signals), peak_strength=float(r.peak_strength),
                         est_lost_units=units, est_lost_units_low=units_lo, est_lost_units_high=units_hi,
                         est_lost_rs=loss_rs, est_lost_rs_low=lo_rs, est_lost_rs_high=hi_rs,
                         intervention_cost_rs=cost, recoverable_rs=act.recoverable * loss_rs,
                         net_benefit_rs=act.recoverable * loss_rs - cost, net_benefit_low_rs=act.recoverable * lo_rs - cost,
                         roi=(act.recoverable * loss_rs) / cost if cost else float("nan"), top_skus=sk, top_pincodes=pc))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["recommendation"] = np.where(df["net_benefit_rs"] > 0, "act", "monitor")
    df = df.sort_values("net_benefit_rs", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
