"""Score detector alarms against the planted-event answer key (latent_events / latent_effects).

Matching rule (deliberately simple and symmetric across detectors): an alarm belongs to an event if its
platform is in the event scope (ALL = any), its zone is in the event's zones, and its day lies in the event's
answer-key window [start_day, window_end] (window_end = end_day + tail_days from latent_effects).
Alarm channels: business (availability, price, focal sell-out, ad share) vs dq (scrape-failure rate).

Per event (should_alert=True, parents only; E6a/E6b are mechanism rows):
    detected  = >=1 business alarm in scope in window;  delay = first alarm day - start_day (can't be < 0)
    above_floor = max(|delta_sold_units_focal|, |delta_sold_units_all|, |delta_lost_units_all|) >= noise_floor
Hard negatives (E4, E10): false_alert = >=1 business alarm in their window.
E8 (blackout, no real-world change): business alarm in window = false_alert; dq alarm = correct "data problem".
Background false alarms: business alarms matching NO event window, per 1000 series-days of monitored panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import ALARM_COLS
from .features import Signals

NOISE_FLOOR = 15.0
MECH_TYPES = {"E6a", "E6b"}
SIZE_METRICS = ("delta_sold_units_focal", "delta_sold_units_all", "delta_lost_units_all")


def _in_scope(alarms: pd.DataFrame, ev: pd.Series, lo: int, hi: int, platform_names: list[str]) -> pd.Series:
    zones = {int(z) for z in str(ev["scope_zones"]).split(",")}
    plat_ok = (pd.Series(True, index=alarms.index) if ev["scope_platform"] == "ALL"
               else alarms["platform_idx"] == platform_names.index(ev["scope_platform"]))
    return plat_ok & alarms["zone_id"].isin(zones) & alarms["day"].between(lo, hi)


def event_table(events: pd.DataFrame, effects: pd.DataFrame, noise_floor: float = NOISE_FLOOR) -> pd.DataFrame:
    ev = events[~events["type"].isin(MECH_TYPES)].copy()
    win = effects.groupby("event_id").agg(window_start=("window_start", "min"), window_end=("window_end", "max"))
    ev = ev.merge(win, left_on="event_id", right_index=True, how="left")
    piv = effects[effects["metric"].isin(SIZE_METRICS)].pivot_table(index="event_id", columns="metric",
                                                                    values="true_delta", aggfunc="first")
    ev["effect_size"] = ev["event_id"].map(piv.abs().max(axis=1))
    ev["above_floor"] = ev["effect_size"] >= noise_floor
    return ev


def score_detector(alarms: pd.DataFrame, events: pd.DataFrame, effects: pd.DataFrame, sig: Signals,
                   platform_names: list[str], warmup_day: int = 12,
                   noise_floor: float = NOISE_FLOOR) -> tuple[pd.DataFrame, dict]:
    ev = event_table(events, effects, noise_floor)
    biz = alarms[alarms["channel"] == "business"]
    dq = alarms[alarms["channel"] == "dq"]
    rows, claimed = [], pd.Series(False, index=biz.index)
    for _, e in ev.iterrows():
        lo, hi = int(e["start_day"]), int(e["window_end"])
        if e["type"] == "E4":
            lo = int(e["window_start"])
        m_b = _in_scope(biz, e, lo, hi, platform_names)
        claimed |= m_b
        m_d = _in_scope(dq, e, lo, hi, platform_names)
        first_b = int(biz.loc[m_b, "day"].min()) if m_b.any() else None
        first_d = int(dq.loc[m_d, "day"].min()) if m_d.any() else None
        rows.append(dict(event_id=int(e["event_id"]), type=e["type"], tier=e["tier"], start_day=int(e["start_day"]),
                         should_alert=bool(e["should_alert"]), hard_negative=bool(e["is_hard_negative"]),
                         effect_size=float(e["effect_size"]), above_floor=bool(e["above_floor"]),
                         warmup_limited=bool(e["start_day"] < warmup_day),
                         n_business_alarms=int(m_b.sum()), first_business_day=first_b,
                         delay_days=None if first_b is None else max(0, first_b - int(e["start_day"])),
                         signals=",".join(sorted(biz.loc[m_b, "signal"].unique())),
                         n_dq_alarms=int(m_d.sum()), dq_delay_days=None if first_d is None else max(0, first_d - int(e["start_day"]))))
    per = pd.DataFrame(rows)
    per["detected"] = per["n_business_alarms"] > 0
    per["false_alert"] = (~per["should_alert"]) & per["detected"]
    per["dq_detected"] = per["n_dq_alarms"] > 0
    # background: business alarms outside every event window (hard negatives included) => true false alarms
    bg_alarms = int((~claimed).sum())
    bg_by_signal = biz.loc[~claimed].groupby("signal").size().to_dict()
    sdays = sum(p.values.shape[0] * (p.values.shape[1] - warmup_day) for k, p in sig.panels.items()
                if p.channel == "business")
    sa = per[per.should_alert]
    af = sa[sa.above_floor]
    bf = sa[~sa.above_floor]
    det = lambda d: d[d.detected]
    summary = dict(
        n_should_alert=len(sa), recall_all=float(sa.detected.mean()),
        n_above_floor=len(af), recall_above_floor=float(af.detected.mean()) if len(af) else float("nan"),
        n_below_floor=len(bf), recall_below_floor=float(bf.detected.mean()) if len(bf) else float("nan"),
        median_delay_above_floor=float(det(af).delay_days.median()) if len(det(af)) else float("nan"),
        mean_delay_above_floor=float(det(af).delay_days.mean()) if len(det(af)) else float("nan"),
        recall_above_floor_post_warmup=float(af[~af.warmup_limited].detected.mean()) if (~af.warmup_limited).any() else float("nan"),
        business_alarms=int(len(biz)), alarm_precision=float(1 - bg_alarms / len(biz)) if len(biz) else float('nan'),
        background_false_alarms=bg_alarms, series_days=int(sdays),
        false_alarms_per_1000_series_days=1000.0 * bg_alarms / sdays, background_by_signal=bg_by_signal,
        false_alert_rate_E4=_fa(per, "E4"), false_alert_rate_E8=_fa(per, "E8"), false_alert_rate_E10=_fa(per, "E10"),
        e8_dq_recall=float(per[per.type == "E8"].dq_detected.mean()),
        e8_dq_median_delay=float(per[(per.type == "E8") & per.dq_detected].dq_delay_days.median())
        if per[(per.type == "E8") & per.dq_detected].shape[0] else float("nan"),
    )
    return per, summary


def _fa(per: pd.DataFrame, t: str) -> float:
    d = per[per.type == t]
    return float(d.false_alert.mean()) if len(d) else float("nan")
