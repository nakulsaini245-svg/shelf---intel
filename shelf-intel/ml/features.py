"""Signal panels for anomaly detection, built ONLY from observed and public tables.

A panel is a [n_series, n_days] matrix (NaN = no data that day) plus series metadata. Series are keyed by
platform x zone (x subcategory for price, x brand for sponsored share). Latent tables are never touched here.

Signals (business unless noted):
    avail         share of available SKU-snapshots per platform x zone                      (down)
    focal_units   focal-brand first-party sell-out units per platform x zone               (both)
    price_*       mean shelf discount % per platform x zone x subcategory                  (up)
                  price_all uses every row; price_nopromo drops rows carrying the promo flag
    share         non-focal brand's share of sponsored slots per platform x zone x brand   (up)
    scrape (dq)   scrape failure rate per platform x zone                                  (up)
    focal_promo   share of focal listing rows flagged promo (auxiliary explanation signal)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sim.io import read_table

OBSERVED_NEEDED = ["fact_listing_daily", "fact_search_visibility", "fact_focal_sellout", "scrape_log",
                   "dim_location", "dim_sku", "dim_brand", "dim_calendar"]


@dataclass
class Panel:
    name: str
    values: np.ndarray            # [S, T]
    meta: pd.DataFrame            # platform_idx, zone_id, key (subcat / brand id, or -1)
    direction: str                # "up" | "down" | "both"
    channel: str                  # "business" | "dq" | "aux"
    floor: float                  # minimum scale (absolute units of the signal)


@dataclass
class Signals:
    panels: dict[str, Panel]
    festival: np.ndarray          # [Z, T] public-calendar festival day (baseline: all zones; regional: its zones)
    n_days: int
    n_platforms: int
    n_zones: int
    focal_brand: int
    festival_regional: np.ndarray | None = None   # [Z, T] subset of `festival` announced as regional


def load_observed(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the observed + public tables (never latent_*) from a generated dataset (Parquet or csv.gz)."""
    return {n: read_table(data_dir, n) for n in OBSERVED_NEEDED}


def _panel(name: str, df: pd.DataFrame, keys: list[str], val: str, n_days: int, direction: str, channel: str,
           floor: float) -> Panel:
    g = df.groupby(keys + ["day_index"])[val].first().unstack("day_index").reindex(columns=range(n_days))
    meta = g.index.to_frame(index=False)
    meta.columns = ["platform_idx", "zone_id"] + (["key"] if len(keys) == 3 else [])
    if "key" not in meta:
        meta["key"] = -1
    return Panel(name, g.to_numpy(dtype="float64"), meta, direction, channel, floor)


def build_signals(T: dict[str, pd.DataFrame], na_policy: str = "exclude") -> Signals:
    """na_policy: "exclude" drops failed scrapes (correct); "zero" reads NA availability as unavailable (naive)."""
    L, V, F, S = (T["fact_listing_daily"], T["fact_search_visibility"], T["fact_focal_sellout"], T["scrape_log"])
    loc, sku = T["dim_location"], T["dim_sku"]
    n_days = int(L["day_index"].max()) + 1
    zone_of = loc.set_index("pincode_id")["zone_id"]
    n_zones, n_plat = int(loc["zone_id"].max()) + 1, int(L["platform_id"].max())
    focal = int(T["dim_brand"].loc[T["dim_brand"]["is_focal"], "brand_id"].iloc[0])
    sub_code = {s: i for i, s in enumerate(sorted(sku["subcategory"].unique()))}
    sku_sub = sku.set_index("sku_id")["subcategory"].map(sub_code)
    sku_brand = sku.set_index("sku_id")["brand_id"]
    panels: dict[str, Panel] = {}

    base = pd.DataFrame({"platform_idx": L["platform_id"].to_numpy() - 1,
                         "zone_id": zone_of.loc[L["pincode_id"]].to_numpy(), "day_index": L["day_index"].to_numpy()})
    # ---- availability
    snaps = L[[f"snap{k}_avail" for k in range(1, 5)]].to_numpy(dtype="float64", na_value=np.nan)
    if na_policy == "zero":
        row = np.nan_to_num(snaps, nan=0.0).mean(1)
    else:
        cnt = (~np.isnan(snaps)).sum(1)
        row = np.where(cnt > 0, np.nansum(snaps, 1) / np.maximum(cnt, 1), np.nan)
    a = base.assign(v=row).dropna(subset=["v"]).groupby(["platform_idx", "zone_id", "day_index"], as_index=False)["v"].mean()
    panels["avail"] = _panel("avail", a, ["platform_idx", "zone_id"], "v", n_days, "down", "business", 0.01)
    # ---- price (discount %), with and without promo-flagged rows
    ok = L["price"].notna().to_numpy()
    pr = base[ok].assign(key=sku_sub.loc[L.loc[ok, "sku_id"]].to_numpy(), v=L.loc[ok, "discount_pct"].to_numpy(float),
                         promo=L.loc[ok, "promo_flag"].to_numpy(dtype="float64", na_value=0.0))
    for nm, d in {"price_all": pr, "price_nopromo": pr[pr["promo"] == 0]}.items():
        g = d.groupby(["platform_idx", "zone_id", "key", "day_index"], as_index=False)["v"].mean()
        panels[nm] = _panel(nm, g, ["platform_idx", "zone_id", "key"], "v", n_days, "up", "business", 0.3)
    # ---- focal sell-out (first party)
    f = pd.DataFrame({"platform_idx": F["platform_id"].to_numpy() - 1, "zone_id": zone_of.loc[F["pincode_id"]].to_numpy(),
                      "day_index": F["day_index"].to_numpy(), "v": F["units"].to_numpy(float)})
    f = f.groupby(["platform_idx", "zone_id", "day_index"], as_index=False)["v"].sum()
    panels["focal_units"] = _panel("focal_units", f, ["platform_idx", "zone_id"], "v", n_days, "both", "business", 1.0)
    # ---- sponsored share per non-focal brand
    sp = V[V["is_sponsored"]]
    s = pd.DataFrame({"platform_idx": sp["platform_id"].to_numpy() - 1, "zone_id": zone_of.loc[sp["pincode_id"]].to_numpy(),
                      "day_index": sp["day_index"].to_numpy(), "key": sku_brand.loc[sp["sku_id"]].to_numpy()})
    cnt = s.groupby(["platform_idx", "zone_id", "day_index", "key"]).size().unstack("key", fill_value=0)
    share = cnt.div(cnt.sum(axis=1), axis=0).drop(columns=[focal], errors="ignore")
    long = share.stack().rename("v").reset_index()
    panels["share"] = _panel("share", long, ["platform_idx", "zone_id", "key"], "v", n_days, "up", "business", 0.02)
    # ---- scrape failure rate (data-quality channel)
    sv = S[S["status"] != "unserviceable"]
    q = pd.DataFrame({"platform_idx": sv["platform_id"].to_numpy() - 1, "zone_id": zone_of.loc[sv["pincode_id"]].to_numpy(),
                      "day_index": sv["day_index"].to_numpy(), "v": (sv["status"] == "failed").to_numpy(float)})
    q = q.groupby(["platform_idx", "zone_id", "day_index"], as_index=False)["v"].mean()
    panels["scrape"] = _panel("scrape", q, ["platform_idx", "zone_id"], "v", n_days, "up", "dq", 0.02)
    # ---- focal promo share (auxiliary)
    fm = ok & (sku_brand.loc[L["sku_id"]].to_numpy() == focal)
    fp = base[fm].assign(v=L.loc[fm, "promo_flag"].to_numpy(dtype="float64", na_value=0.0))
    fp = fp.groupby(["platform_idx", "zone_id", "day_index"], as_index=False)["v"].mean()
    panels["focal_promo"] = _panel("focal_promo", fp, ["platform_idx", "zone_id"], "v", n_days, "up", "aux", 0.03)
    # ---- public festival calendar
    cal, d0 = T["dim_calendar"], L["date"].min()
    fest = np.zeros((n_zones, n_days), dtype=bool)
    reg = np.zeros((n_zones, n_days), dtype=bool)
    for _, r in cal.iterrows():
        d = int((r["date"] - d0).days)
        if 0 <= d < n_days:
            if pd.isna(r["scope_zone_id"]):
                fest[:, d] = True
            else:
                fest[int(r["scope_zone_id"]), d] = True
                reg[int(r["scope_zone_id"]), d] = True
    return Signals(panels, fest, n_days, n_plat, n_zones, focal, reg)
