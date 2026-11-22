"""Week 1 tests: determinism, append-only invariance, bounds, missing-scrape handling, hook seam, IO."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.config import config_from_dict, load_config
from sim.events import DayModifiers, NullHook
from sim.io import frame_hash, read_table, table_hashes, write_dataset
from sim.schema import DAY_INDEXED, LATENT_COLUMNS, OBSERVED
from sim.seeds import rng_for, seed_words
from sim.simulate import simulate
from sim.validate import invariants, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_cfg():
    """Small world (fast) with the same mechanics as v1."""
    raw = yaml.safe_load((ROOT / "configs/world_v1.yaml").read_text())
    raw["scale"].update(n_pincodes=16, n_zones=4, zone_sizes=None, stores_per_platform=6, n_skus=45)
    raw["scale"]["unserviced_share"] = 0.25  # tiny world has few pairs; guarantee some coverage gaps
    raw["inventory"]["warmup_days"] = 4
    raw["observation"].update(p_timeout=0.08, evening_extra=0.06, p_stale=0.03)  # more failures to exercise
    raw["targets"] = {}
    raw["events"] = {"enabled": False}  # Week 1 tests exercise the baseline world only
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def res8(tiny_cfg):
    return simulate(tiny_cfg, n_days=8)


# ---------------------------------------------------------------- seeds
def test_seed_words_golden_and_stable():
    assert seed_words(42, "cell", 0, "demand") == [494660989, 4026487039, 250096019, 4025008317]


def test_rng_keys_are_independent():
    a = rng_for(1, "cell", 0, "x").random(4)
    assert np.array_equal(a, rng_for(1, "cell", 0, "x").random(4))
    for other in [(2, "cell", 0, "x"), (1, "series", 0, "x"), (1, "cell", 1, "x"), (1, "cell", 0, "y")]:
        assert not np.array_equal(a, rng_for(*other).random(4))


# ---------------------------------------------------------------- determinism
def test_rerun_is_identical(tiny_cfg, res8):
    again = simulate(tiny_cfg, n_days=8)
    assert table_hashes(res8.tables) == table_hashes(again.tables)


def test_different_seed_changes_data(tiny_cfg, res8):
    other = simulate(replace(tiny_cfg, seed=tiny_cfg.seed + 1), n_days=8)
    assert table_hashes(res8.tables)["fact_listing_daily"] != table_hashes(other.tables)["fact_listing_daily"]


def test_append_day_leaves_history_untouched(tiny_cfg, res8):
    res9 = simulate(tiny_cfg, n_days=9)
    assert table_hashes(res8.tables, up_to_day=8) == table_hashes(res9.tables, up_to_day=8)
    for t in DAY_INDEXED:  # the new day really exists
        assert res9.tables[t]["day_index"].max() == 8


# ---------------------------------------------------------------- bounds / schema
def test_all_invariants_pass(res8):
    failed = [str(c) for c in invariants(res8) if not c.ok]
    assert not failed, failed


def test_price_and_stock_bounds_explicit(res8):
    L = res8.tables["fact_listing_daily"]
    ok = L[L["price"].notna()]
    assert (ok["price"].astype(int) <= ok["mrp"]).all() and (ok["price"] >= 1).all()
    assert (res8.tables["latent_stock"]["on_hand_units"] >= 0).all()
    d = res8.tables["latent_demand"]
    assert (d["sold_units"] <= d["true_demand_units"]).all()


def test_stockouts_and_promos_actually_occur(res8):
    assert (res8.tables["latent_stock"]["on_hand_units"] == 0).any()
    assert res8.tables["fact_listing_daily"]["promo_flag"].dropna().sum() > 0


def test_observed_tables_leak_no_latent_columns(res8):
    for name in OBSERVED:
        assert not (set(res8.tables[name].columns) & LATENT_COLUMNS)


# ---------------------------------------------------------------- missing scrape handling
def test_failed_scrapes_become_na_not_filled(res8):
    sc, L = res8.tables["scrape_log"], res8.tables["fact_listing_daily"]
    assert (sc["status"] == "failed").sum() > 0
    assert {"timeout", "serviceability_error"} <= set(sc.loc[sc["status"] == "failed", "reason"].dropna())
    assert L["snap1_avail"].isna().any()
    names = {c.name: c for c in invariants(res8)}
    for n in ["na_avail_iff_failed_scrape", "unknown_iff_scrape_not_ok", "price_na_iff_scrape_not_ok",
              "all_snaps_na_iff_scrape_not_ok"]:
        assert names[n].ok, n


def test_evening_failures_exceed_morning(res8):
    sc = res8.tables["scrape_log"]
    f = sc[sc["status"] == "failed"].groupby("snapshot").size()
    assert f.get(3, 0) + f.get(4, 0) > f.get(1, 0) + f.get(2, 0)  # MNAR evening skew


def test_unserviced_pairs_logged_distinctly(res8):
    sc = res8.tables["scrape_log"]
    assert (sc["status"] == "unserviceable").any()
    assert set(sc.loc[sc["status"] == "unserviceable", "reason"]) == {"not_covered"}


def test_stale_scrapes_are_invisible_in_observed_data(res8):
    flags = res8.tables["latent_obs_flags"]
    assert len(flags) > 0 and flags["was_stale"].all()
    sc = res8.tables["scrape_log"]
    key = ["day_index", "platform_id", "pincode_id", "snapshot"]
    m = flags.merge(sc, on=key)
    assert (m["status"] == "ok").all()  # the scraper believes these were fine


# ---------------------------------------------------------------- hook seam (Week 2 events)
@dataclass
class _Hook(NullHook):
    day: int = 3
    pin: int = 0
    mode: str = "demand0"

    def modifiers(self, day, day_date, world):
        m = DayModifiers()
        if day != self.day:
            return m
        mask_cells = world.cell_pin == self.pin
        if self.mode == "demand0":
            m.demand_mult = np.where(mask_cells, 0.0, 1.0)
        elif self.mode == "blackout":
            b = np.zeros((world.P, world.L), dtype=bool)
            b[:, self.pin] = True
            m.scrape_blackout = b
        elif self.mode == "suppress":
            m.listing_suppress = mask_cells.copy()
        return m


def test_null_hook_equals_no_hook(tiny_cfg, res8):
    assert table_hashes(simulate(tiny_cfg, n_days=8, hook=NullHook()).tables) == table_hashes(res8.tables)


def test_hook_demand_override_is_applied_and_scoped(tiny_cfg, res8):
    r = simulate(tiny_cfg, n_days=6, hook=_Hook(mode="demand0"))
    d = r.tables["latent_demand"]
    hit = d[(d["day_index"] == 3) & (d["pincode_id"] == 0)]
    assert len(hit) > 0 and (hit["true_demand_units"] == 0).all()
    base = res8.tables["latent_demand"]
    assert base[(base["day_index"] == 3) & (base["pincode_id"] == 0)]["true_demand_units"].sum() > 0
    # days before the event are byte-identical to baseline
    assert table_hashes(r.tables, up_to_day=3)["latent_demand"] == table_hashes(res8.tables, up_to_day=3)["latent_demand"]


def test_hook_blackout_yields_failed_scrapes_and_na(tiny_cfg):
    r = simulate(tiny_cfg, n_days=5, hook=_Hook(mode="blackout"))
    sc, L = r.tables["scrape_log"], r.tables["fact_listing_daily"]
    day3 = sc[(sc["day_index"] == 3) & (sc["pincode_id"] == 0) & (sc["status"] != "unserviceable")]
    assert (day3["status"] == "failed").all() and (day3["reason"] == "http_5xx").all()
    row = L[(L["day_index"] == 3) & (L["pincode_id"] == 0)]
    assert row["snap1_avail"].isna().all() and (row["listing_status"] == "unknown").all()
    assert all(c.ok for c in invariants(r) if "na_" in c.name or "unknown" in c.name)


def test_hook_phantom_suppress_shows_unlisted_and_no_sales(tiny_cfg):
    r = simulate(tiny_cfg, n_days=5, hook=_Hook(mode="suppress"))
    L = r.tables["fact_listing_daily"]
    row = L[(L["day_index"] == 3) & (L["pincode_id"] == 0) & L["scrape_ok"]]
    assert (row["listing_status"] == "unlisted").all()
    assert all(c.ok for c in invariants(r) if c.name == "unlisted_rows_unavailable_except_stale")
    d = r.tables["latent_demand"]
    assert (d[(d["day_index"] == 3) & (d["pincode_id"] == 0)]["sold_units"] == 0).all()


def test_bad_modifier_shape_is_rejected(tiny_cfg):
    @dataclass
    class Bad(NullHook):
        def modifiers(self, day, day_date, world):
            return DayModifiers(demand_mult=np.ones(3))
    with pytest.raises(ValueError, match="demand_mult"):
        simulate(tiny_cfg, n_days=2, hook=Bad())


# ---------------------------------------------------------------- IO
def test_write_read_roundtrip_preserves_content(tmp_path, res8):
    fmt = "parquet"
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        fmt = "csv.gz"
    m = write_dataset(res8, tmp_path, fmt=fmt)
    assert m["tables"]["fact_listing_daily"]["rows"] == len(res8.tables["fact_listing_daily"])
    for name in ["fact_listing_daily", "scrape_log", "fact_focal_sellout"]:
        assert frame_hash(read_table(tmp_path, name)) == frame_hash(res8.tables[name]), name


def test_no_latent_option_omits_truth(tmp_path, res8):
    m = write_dataset(res8, tmp_path, fmt="csv.gz", include_latent=False)
    assert not any(n.startswith("latent_") for n in m["tables"])


# ---------------------------------------------------------------- full scale (opt-in)
@pytest.mark.slow
def test_full_scale_marginals_in_target_ranges():
    cfg = load_config(ROOT / "configs/world_v1.yaml")
    res = simulate(cfg, n_days=30)
    bad = [str(c) for c in validate(res) if not c.ok]
    assert not bad, bad
