"""Fuse per-signal alarms into incidents: alarms in the same platform x zone within `gap` days form one incident.

An incident carries the evidence needed downstream: which signals fired (availability, price, focal sell-out, ad share),
its day span, and its peak strength (alarm score = residual / conformal half-width). `corroborated` incidents have
>= 2 distinct business signals or a single very strong signal (`strong`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_incidents(alarms: pd.DataFrame, gap: int = 3, strong: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (alarms with incident_id, incidents). Only business-channel alarms form incidents."""
    a = alarms[alarms["channel"] == "business"].sort_values(["platform_idx", "zone_id", "day"]).copy()
    if a.empty:
        a["incident_id"] = pd.Series(dtype=int)
        return a, pd.DataFrame(columns=["incident_id", "platform_idx", "zone_id", "first_day", "last_day", "signals",
                                        "n_signals", "peak_strength", "corroborated"])
    new = ((a["platform_idx"] != a["platform_idx"].shift()) | (a["zone_id"] != a["zone_id"].shift())
           | (a["day"] - a["day"].shift() > gap))
    a["incident_id"] = new.cumsum().astype(int) - 1
    g = a.groupby("incident_id")
    inc = g.agg(platform_idx=("platform_idx", "first"), zone_id=("zone_id", "first"), first_day=("day", "min"),
                last_day=("day", "max"), n_signals=("signal", "nunique"), peak_strength=("score", "max"),
                signals=("signal", lambda s: ",".join(sorted(set(s))))).reset_index()
    inc["corroborated"] = (inc["n_signals"] >= 2) | (inc["peak_strength"] >= strong)
    return a, inc


def keep_corroborated(alarms: pd.DataFrame, gap: int = 3, strong: float = 2.0) -> pd.DataFrame:
    """Alarm table with uncorroborated business incidents removed (DQ alarms pass through)."""
    a, inc = build_incidents(alarms, gap, strong)
    ok = set(inc.loc[inc["corroborated"], "incident_id"])
    kept = a[a["incident_id"].isin(ok)].drop(columns="incident_id")
    return pd.concat([kept, alarms[alarms["channel"] != "business"]], ignore_index=True)
