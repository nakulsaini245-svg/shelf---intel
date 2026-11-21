"""Dataset validation: schema, hard invariants, missing-data handling, and calibration targets.

`invariants()` must ALL pass for any config. `target_checks()` compares marginals to the ranges in
config.targets (assumed plausibility bands, not empirical benchmarks) and is used to calibrate the world.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import ALL_TABLES, DIMS, LATENT, LATENT_COLUMNS, OBSERVED
from .simulate import SimResult


@dataclass
class Check:
    name: str
    value: float
    lo: float
    hi: float
    ok: bool

    def __str__(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.value:.4g} (expected {self.lo:g}..{self.hi:g})"


def _flag(name: str, cond: bool) -> Check:
    return Check(name, float(cond), 1.0, 1.0, bool(cond))


def _key(day, plat, pin) -> np.ndarray:
    return day.astype(np.int64) * 10**6 + plat.astype(np.int64) * 10**3 + pin.astype(np.int64)


def invariants(res: SimResult) -> list[Check]:
    T, cfg, w = res.tables, res.cfg, res.world
    out: list[Check] = []
    # schema
    for name, spec in ALL_TABLES.items():
        out.append(_flag(f"schema:{name}", list(T[name].columns) == list(spec)
                         and all(str(T[name][c].dtype) == d for c, d in spec.items())))
    # latent leak guard
    for name in list(OBSERVED) + list(DIMS):
        out.append(_flag(f"no_latent_columns:{name}", not (set(T[name].columns) & LATENT_COLUMNS)))
    L, ld, ls = T["fact_listing_daily"], T["latent_demand"], T["latent_stock"]
    v, sv, sc = T["fact_search_visibility"], T["fact_focal_sellout"], T["scrape_log"]
    # price / discount bounds
    pr = L["price"].dropna().astype("int64")
    mrp = L.loc[L["price"].notna(), "mrp"].astype("int64")
    out.append(_flag("price<=mrp", bool((pr.to_numpy() <= mrp.to_numpy()).all())))
    out.append(_flag("price>=1", bool((pr >= 1).all())))
    floor_ok = (pr.to_numpy() >= np.ceil(mrp.to_numpy() * (1 - cfg.pricing.max_discount)) - 1e-9).all()
    out.append(_flag("price>=floor", bool(floor_ok)))
    dp = L["discount_pct"].dropna()
    out.append(_flag("discount_in_range", bool(((dp >= 0) & (dp <= cfg.pricing.max_discount * 100 + 0.1)).all())))
    # availability values
    for k in range(1, 5):
        out.append(_flag(f"avail_binary:snap{k}", bool(L[f"snap{k}_avail"].dropna().isin([0, 1]).all())))
    # stock and sales
    out.append(_flag("stock_non_negative", bool((ls["on_hand_units"] >= 0).all())))
    out.append(_flag("sold<=demand", bool((ld["sold_units"] <= ld["true_demand_units"]).all())))
    out.append(_flag("lost>=0", bool((ld["lost_units"] >= 0).all())))
    out.append(_flag("focal_units_non_negative", bool((sv["units"] >= 0).all() and (sv["gmv"] >= 0).all())))
    # sold units never exceed what the dark store could have held: checked via stock non-negativity above
    # visibility structure
    out.append(_flag("position_range", bool(v["position"].between(1, cfg.ads.n_slots).all())))
    gk = ["day_index", "snapshot", "platform_id", "pincode_id", "query_id"]
    out.append(_flag("unique_position_per_group", not v.duplicated(gk + ["position"]).any()))
    out.append(_flag("unique_sku_per_group", not v.duplicated(gk + ["sku_id"]).any()))
    sp_pos = set(cfg.ads.sponsored_positions)
    out.append(_flag("sponsored_only_at_sponsored_positions",
                     bool(v.loc[v["is_sponsored"], "position"].isin(sp_pos).all())))
    # scrape log completeness: every platform x pincode x snapshot x day is logged exactly once
    out.append(_flag("scrape_log_complete",
                     len(sc) == res.n_days * w.P * w.L * 4 and not sc.duplicated(
                         ["day_index", "platform_id", "pincode_id", "snapshot"]).any()))
    # MISSING-DATA HANDLING: NA availability <=> failed scrape, never silently filled
    first = L.drop_duplicates(["day_index", "platform_id", "pincode_id"])
    consistent = True
    for k in range(1, 5):
        na_keys = _key(first.loc[first[f"snap{k}_avail"].isna(), "day_index"].to_numpy(),
                       first.loc[first[f"snap{k}_avail"].isna(), "platform_id"].to_numpy(),
                       first.loc[first[f"snap{k}_avail"].isna(), "pincode_id"].to_numpy())
        f = sc[(sc["snapshot"] == k) & (sc["status"] == "failed")]
        fail_keys = _key(f["day_index"].to_numpy(), f["platform_id"].to_numpy(), f["pincode_id"].to_numpy())
        consistent &= bool(np.array_equal(np.sort(na_keys), np.sort(fail_keys)))
    out.append(_flag("na_avail_iff_failed_scrape", consistent))
    unk = L["listing_status"] == "unknown"
    out.append(_flag("unknown_iff_scrape_not_ok", bool((unk == ~L["scrape_ok"]).all())))
    out.append(_flag("price_na_iff_scrape_not_ok", bool((L["price"].isna() == ~L["scrape_ok"]).all())))
    all_na = L[[f"snap{k}_avail" for k in range(1, 5)]].isna().all(axis=1)
    out.append(_flag("all_snaps_na_iff_scrape_not_ok", bool((all_na == ~L["scrape_ok"]).all())))
    # An unlisted cell must read "unavailable" EXCEPT on snapshots the scraper silently served stale
    # (the stale response replays the previous snapshot's value; truth is in latent_obs_flags).
    unl = L[L["listing_status"] == "unlisted"]
    fl = T["latent_obs_flags"]
    stale_keys = set(_key(fl["day_index"].to_numpy(), fl["platform_id"].to_numpy(), fl["pincode_id"].to_numpy())
                     * 10 + fl["snapshot"].to_numpy())
    unexplained = 0
    for k in range(1, 5):
        bad = unl[unl[f"snap{k}_avail"].fillna(0) == 1]
        keys = _key(bad["day_index"].to_numpy(), bad["platform_id"].to_numpy(), bad["pincode_id"].to_numpy()) * 10 + k
        unexplained += sum(int(x) not in stale_keys for x in keys)
    out.append(_flag("unlisted_rows_unavailable_except_stale", unexplained == 0))
    return out


def _metrics(res: SimResult) -> dict[str, float]:
    T, w = res.tables, res.world
    L, ld, ls, v, sc = (T["fact_listing_daily"], T["latent_demand"], T["latent_stock"],
                        T["fact_search_visibility"], T["scrape_log"])
    sku_brand = w.sku_brand
    sp = v[v["is_sponsored"]]
    share = pd.Series(sku_brand[sp["sku_id"].to_numpy()]).value_counts(normalize=True)
    focal = w.sku_is_focal[ld["sku_id"].to_numpy()]
    served = sc[sc["status"] != "unserviceable"]
    return {
        "oos_rate_snapshot": float((ls["on_hand_units"] == 0).mean()),
        "mean_discount_pct": float(L["discount_pct"].dropna().mean()),
        "promo_share": float(L["promo_flag"].dropna().mean()),
        "scrape_fail_rate": float((served["status"] == "failed").mean()),
        "unserviced_share": float((w.pair_of < 0).mean()),
        "top3_brand_sponsored_share": float(share.head(3).sum()) if len(share) else float("nan"),
        "focal_fill_rate": float(ld.loc[focal, "sold_units"].sum() / max(1, ld.loc[focal, "true_demand_units"].sum())),
        "mean_units_per_cell_day": float(ld["sold_units"].mean()),
    }


def target_checks(res: SimResult) -> list[Check]:
    m = _metrics(res)
    out = []
    for name, (lo, hi) in res.cfg.targets.items():
        val = m.get(name, float("nan"))
        out.append(Check(f"target:{name}", val, lo, hi, bool(lo <= val <= hi)))
    return out


def validate(res: SimResult, include_targets: bool = True) -> list[Check]:
    return invariants(res) + (target_checks(res) if include_targets else [])
