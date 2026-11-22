"""Step 2A tests: planted events E1..E10 and their counterfactual answer key."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.config import config_from_dict, load_config
from sim.events import NullHook
from sim.injector import HARD_NEGATIVE, plan_events
from sim.io import table_hashes
from sim.schema import OBSERVED
from sim.simulate import simulate
from sim.validate import invariants
from sim.world import build_world

ROOT = Path(__file__).resolve().parents[1]
ALL1 = {f"E{i}": 1 for i in range(1, 11)}


def make_cfg(counts, tiers=(0.0,), horizon=70):
    raw = yaml.safe_load((ROOT / "configs/world_v1.yaml").read_text())
    raw["scale"].update(n_pincodes=16, n_zones=4, zone_sizes=None, stores_per_platform=6, n_skus=45,
                        unserviced_share=0.25)
    raw["inventory"]["warmup_days"] = 4
    raw["events"].update(enabled=True, horizon_days=horizon, min_start_day=6, tail_days=3,
                         tiers=list(tiers), counts=counts)
    raw["targets"] = {}
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def res():
    return simulate(make_cfg(ALL1), n_days=70)


def eff(res, etype, metric, comp=None):
    ev = res.tables["latent_events"]
    eid = ev[(ev["type"] == (comp or etype))]["event_id"].iloc[0]
    ef = res.tables["latent_effects"]
    return float(ef[(ef["event_id"] == eid) & (ef["metric"] == metric)]["true_delta"].iloc[0])


# ---------------------------------------------------------------- the plan (full scale, cheap)
def test_v1_plan_structure():
    cfg = load_config(ROOT / "configs/world_v1.yaml")
    w = build_world(cfg)
    plan = plan_events(w, cfg)
    parents = [e for e in plan if e.parent is None]
    assert len(parents) == sum(cfg.events.counts.values()) == 29
    for t, n in cfg.events.counts.items():
        assert sum(e.type == t for e in parents) == n
    assert len({e.event_id for e in plan}) == len(plan)
    assert {e.type for e in plan if e.parent is not None} == {"E6a", "E6b"}
    for e in parents:
        assert e.first >= cfg.events.min_start_day and e.last + cfg.events.tail_days <= cfg.events.horizon_days - 1
    pad = cfg.events.tail_days + cfg.events.min_gap_days
    for i, a in enumerate(parents):
        for b in parents[i + 1:]:
            overlap = not (a.first > b.last + pad or b.first > a.last + pad)
            same_p = a.platform is None or b.platform is None or a.platform == b.platform
            assert not (overlap and same_p and set(a.zones) & set(b.zones)), (a.name, b.name)
    tiers = {e.difficulty for e in parents}
    assert tiers == set(cfg.events.tiers)
    assert {e.type for e in parents if e.hard_negative} == HARD_NEGATIVE


def test_plan_is_deterministic_and_independent_of_days_simulated():
    cfg = load_config(ROOT / "configs/world_v1.yaml")
    w = build_world(cfg)
    a, b = plan_events(w, cfg), plan_events(w, cfg)
    assert [(e.type, e.start, e.end, e.zones) for e in a] == [(e.type, e.start, e.end, e.zones) for e in b]
    short = simulate(cfg, n_days=30, counterfactual=False).tables["latent_events"]
    full = simulate(cfg, n_days=90, counterfactual=False).tables["latent_events"]
    assert set(short["event_id"]) <= set(full["event_id"]) and len(short) < len(full)
    j = short.merge(full, on="event_id", suffixes=("_s", "_f"))
    assert (j["start_day_s"] == j["start_day_f"]).all() and (j["params_s"] == j["params_f"]).all()


# ---------------------------------------------------------------- determinism / seam
def test_event_run_is_deterministic(res):
    again = simulate(make_cfg(ALL1), n_days=70)
    assert table_hashes(res.tables) == table_hashes(again.tables)


def test_history_before_first_event_equals_baseline(res):
    first = int(res.tables["latent_effects"]["window_start"].min())   # earliest day any mechanic acts
    base = simulate(res.cfg, n_days=70, hook=NullHook())
    keep = lambda t: {k: v for k, v in t.items() if k != "dim_calendar"}   # public calendar differs by design  # noqa: E731
    assert table_hashes(keep(res.tables), up_to_day=first) == table_hashes(keep(base.tables), up_to_day=first)


def test_append_only_holds_with_events():
    cfg = make_cfg(ALL1)
    a, b = simulate(cfg, n_days=30, counterfactual=False), simulate(cfg, n_days=31, counterfactual=False)
    assert table_hashes(a.tables, up_to_day=30) == table_hashes(b.tables, up_to_day=30)


def test_invariants_hold_with_events(res):
    assert not [str(c) for c in invariants(res) if not c.ok]


# ---------------------------------------------------------------- answer key correctness
def test_hard_negative_flags_and_parent_links(res):
    ev = res.tables["latent_events"].set_index("event_id")
    assert set(ev[ev["is_hard_negative"]]["type"]) == HARD_NEGATIVE
    assert not ev[ev["is_hard_negative"]]["should_alert"].any()
    kids = ev[ev["causal_parent_event_id"].notna()]
    assert set(kids["type"]) == {"E6a", "E6b"} and (ev.loc[kids["causal_parent_event_id"], "type"] == "E6").all()
    assert ev[ev["type"] == "E6"]["should_alert"].all()


def test_pure_observation_event_has_exactly_zero_world_effect():
    """E8 only changes what the scraper sees. With common random numbers the counterfactual must be exact."""
    r = simulate(make_cfg({"E8": 1}), n_days=70)
    for m in ["delta_sold_units_focal", "delta_gmv_focal", "delta_lost_units_all", "delta_true_demand_units_all",
              "delta_oos_snapshot_rate"]:
        assert eff(r, "E8", m) == 0.0, m
    assert eff(r, "E8", "delta_scrape_failure_rate_pp") > 20.0


def test_mechanism_signatures(res):
    assert eff(res, "E1", "delta_oos_snapshot_rate") > 0 and eff(res, "E1", "delta_lost_units_all") > 0
    assert eff(res, "E2", "delta_oos_snapshot_rate") > 0 and eff(res, "E2", "delta_lost_units_all") > 0
    assert eff(res, "E3", "mean_price_delta_pct") < -5
    assert eff(res, "E4", "delta_sold_units_focal_promo_phase") > 0
    assert eff(res, "E4", "delta_sold_units_focal_cliff_phase") < eff(res, "E4", "delta_sold_units_focal_promo_phase")
    assert eff(res, "E5", "delta_attacker_sponsored_sov_pp") > 0 and eff(res, "E5", "delta_focal_sponsored_sov_pp") < 0
    assert eff(res, "E7", "delta_sold_units_focal") < 0
    assert eff(res, "E9", "mean_price_delta_pct") < 0
    assert eff(res, "E10", "delta_true_demand_units_all") > 0


def test_e6_attribution_identity_and_children(res):
    tot = eff(res, "E6", "delta_sold_units_focal")
    parts = sum(eff(res, "E6", f"attr_{k}_delta_sold_units_focal") for k in ["a", "b", "interaction"])
    assert abs(tot - parts) < 1e-6
    assert eff(res, "E6", "mean_price_delta_pct", comp="E6a") < 0
    assert eff(res, "E6", "delta_attacker_sponsored_sov_pp", comp="E6b") > 0


def test_e1_has_store_level_variance_inside_the_zone():
    r = simulate(make_cfg({"E1": 1}, tiers=(0.5,)), n_days=70, counterfactual=False)
    import json
    mult = json.loads(r.tables["latent_events"].iloc[0]["params"])["store_fill_mult"]
    assert len(mult) >= 1 and all(0.0 <= v <= 1.0 for v in mult.values())


def test_observable_traces_of_events(res):
    L, ev = res.tables["fact_listing_daily"], res.tables["latent_events"]
    e7 = ev[ev["type"] == "E7"].iloc[0]  # easy tier: all affected cells visibly delisted
    pins = [int(x) for x in e7["scope_pincodes"].split(",")]
    d = L[(L["day_index"].between(e7["start_day"], e7["end_day"])) & L["pincode_id"].isin(pins)]
    assert (d["listing_status"] == "unlisted").any()
    e4 = ev[ev["type"] == "E4"].iloc[0]
    pr = L[(L["day_index"] >= e4["start_day"] - 5) & (L["day_index"] < e4["start_day"])]
    assert pr["promo_flag"].dropna().sum() > 0


def test_public_calendar_exposes_festivals_not_incidents(res):
    cal, ev = res.tables["dim_calendar"], res.tables["latent_events"]
    reg = cal[cal["kind"] == "regional"]
    e10 = ev[ev["type"] == "E10"].iloc[0]
    assert len(reg) == (e10["end_day"] - e10["start_day"] + 1) * len(e10["scope_zones"].split(","))
    assert reg["festival_name"].str.startswith("Regional festival").all()
    assert not any(c in OBSERVED for c in ["latent_events", "latent_effects"])


def test_counterfactual_can_be_skipped():
    r = simulate(make_cfg(ALL1), n_days=70, counterfactual=False)
    assert len(r.tables["latent_events"]) > 0 and len(r.tables["latent_effects"]) == 0


@pytest.mark.slow
def test_v1_full_scale_events_and_targets():
    from sim.validate import validate
    r = simulate(load_config(ROOT / "configs/world_v1.yaml"))
    assert len(r.tables["latent_events"]) == 35 and len(r.tables["latent_effects"]) > 200
    assert not [str(c) for c in validate(r) if not c.ok]
