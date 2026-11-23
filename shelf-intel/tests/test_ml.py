"""Fast tests for the conformal scorer, incident fusion and the revenue-at-risk ranker (+ one eval on world_v1)."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ml.ranker as ranker_mod
from ml.conformal import ConformalDetector, conformal_bands, expected_series
from ml.incidents import build_incidents, keep_corroborated
from ml.ranker import ACTIONS, RankerCfg, infer_action, rank_incidents
from tests.test_baselines import _signals, weekly

DATA = Path("data/world_v1")


# ---------------------------------------------------------------- conformal bands
def test_conformal_coverage_close_to_nominal_on_noise():
    x = weekly(T=120, S=40, amp=4.0, noise=1.0, seed=3)
    _, q, s = conformal_bands(x, floor=0.1, alpha=0.05)
    cov = (np.abs(s[~np.isnan(s)]) <= 1.0).mean()
    assert 0.90 <= cov <= 0.995


def test_conformal_is_online_and_needs_history():
    x = weekly(T=80, S=5)
    E1, q1, s1 = conformal_bands(x, 0.1)
    x2 = x.copy()
    x2[:, 60:] *= 5
    _, q2, s2 = conformal_bands(x2, 0.1)
    assert np.allclose(s1[:, :60], s2[:, :60], equal_nan=True)         # future cannot change the past
    assert np.isnan(s1[:, :12]).all()                                   # no score before enough calibration data


def test_gap_keeps_ongoing_incident_out_of_its_own_interval():
    x = weekly(T=80, S=10, amp=0.0, noise=0.5, seed=1)
    x[:, 45:] -= 6.0
    _, q_gap, _ = conformal_bands(x, 0.1, gap=3)
    _, q_nogap, _ = conformal_bands(x, 0.1, gap=0)
    assert np.nanmean(q_nogap[:, 48:52]) >= np.nanmean(q_gap[:, 48:52])
    assert np.nanmean(q_gap[:, 46:48]) < 2.0                            # still tight two days into the shift


def test_expected_series_matches_weekly_pattern_and_status():
    x = weekly(T=60, S=2, amp=5.0, noise=0.01)
    E = expected_series(x)
    assert np.nanmax(np.abs(E[:, 28:] - x[:, 28:])) < 0.5
    st = np.zeros((2, 60), bool)
    st[:, 40:] = True
    assert np.isfinite(expected_series(x, st)[:, 45]).all()             # falls back when no matched reference exists


def test_detector_alarms_on_drop_and_stays_quiet_on_noise():
    T = 80
    x = weekly(T=T, S=6, amp=0.0, noise=0.4, seed=5)
    x[2, 55:60] -= 8.0
    det = ConformalDetector()
    al = det.run(_signals(x))
    hit = al[(al.channel == "business") & (al.zone_id == 2) & al.day.between(55, 61)]
    assert len(hit) > 0 and hit.day.min() <= 57
    assert len(al[(al.channel == "business") & (al.zone_id != 2)]) <= 3          # few chance alarms elsewhere
    assert set(det.bands) >= {"avail", "focal_units", "price", "share", "scrape"}


def test_regional_festival_suppresses_conformal_alarm():
    T = 80
    x = weekly(T=T, S=3, amp=0.0, noise=0.4, seed=2)
    x[0, 55:60] -= 10.0                                                  # a drop: the (up-only) scrape panel stays quiet
    reg = np.zeros((3, T), bool)
    reg[0, 54:61] = True
    no_cal = ConformalDetector().run(_signals(x))
    cal = ConformalDetector().run(_signals(x, fest=reg, reg=reg))
    inwin = lambda a: ((a.zone_id == 0) & a.day.between(55, 62) & (a.channel == "business")).sum()
    assert inwin(no_cal) > 0 and inwin(cal) == 0


# ---------------------------------------------------------------- incidents
def _al(day, zone, sig="avail", plat=0, score=1.5, ch="business"):
    return dict(detector="t", signal=sig, channel=ch, platform_idx=plat, zone_id=zone, key=-1, day=day, direction="down", score=score)


def test_incident_clustering_and_corroboration():
    al = pd.DataFrame([_al(10, 0), _al(12, 0, "price"), _al(30, 0),            # zone 0: one 2-signal incident + a later single
                       _al(10, 1, plat=1), _al(11, 0, ch="dq", sig="scrape")])  # DQ alarms never form incidents
    a, inc = build_incidents(al, gap=3, strong=2.0)
    assert len(inc) == 3 and set(a["channel"]) == {"business"}
    two = inc[inc.n_signals == 2].iloc[0]
    assert two.first_day == 10 and two.last_day == 12 and two.corroborated
    assert not inc[(inc.first_day == 30)].corroborated.iloc[0]
    kept = keep_corroborated(al, gap=3, strong=2.0)
    assert (kept[kept.channel == "business"].day.isin([10, 12])).all() and (kept.channel == "dq").sum() == 1


# ---------------------------------------------------------------- ranker
def _obs(P=1, Z=2, T=60, units=50.0, price=100.0):
    loc = pd.DataFrame({"pincode_id": np.arange(Z, dtype="int16"), "zone_id": np.arange(Z, dtype="int8")})
    rows = [(d, p + 1, z, 7, int(units), int(units * price)) for d in range(T) for p in range(P) for z in range(Z)]
    f = pd.DataFrame(rows, columns=["day_index", "platform_id", "pincode_id", "sku_id", "units", "gmv"])
    return {"dim_location": loc, "fact_focal_sellout": f}


def _rank_setup(drops, signals=("avail", "avail"), T=60, first=40):
    """Zones with a given daily focal drop starting at `first`; returns (ranked, truth_units_per_zone)."""
    x = weekly(T=T, S=len(drops), base=50.0, amp=0.0, noise=0.3, seed=9)
    for z, dr in enumerate(drops):
        x[z, first:] -= dr
    sig = _signals(x)
    det = ConformalDetector()
    al = det.run(sig)
    _, inc = build_incidents(al)
    inc = inc.assign(signals=lambda d: d.zone_id.map(lambda z: signals[z % len(signals)]))
    obs = _obs(Z=len(drops), T=T)
    return rank_incidents(inc, det.bands, sig, obs), inc


def test_ranker_orders_by_size_and_has_ordered_intervals():
    r, _ = _rank_setup([25.0, 12.0, 0.0])
    assert list(r.zone_id[:2]) == [0, 1] and (r["rank"] == np.arange(1, len(r) + 1)).all()
    assert (r.est_lost_units_low <= r.est_lost_units + 1e-9).all() and (r.est_lost_units <= r.est_lost_units_high + 1e-9).all()
    assert (r.est_lost_rs > 0).all() and abs(r.iloc[0].est_lost_rs / r.iloc[0].est_lost_units - 100.0) < 1.0   # Rs per unit from observed gmv/units


def test_ranker_estimate_is_online_and_near_true_shortfall():
    r, _ = _rank_setup([25.0, 0.0, 0.0])
    top = r.iloc[0]
    realized_truth = 25.0 * (top.decision_day - 40 + 1)
    assert top.decision_day <= top.first_day + RankerCfg().decision_lag + 1
    assert 0.6 * realized_truth <= top.est_lost_units <= 2.5 * realized_truth      # realized part + 4-day projection


def test_cost_aware_decision_monitors_small_incidents():
    r, _ = _rank_setup([40.0, 4.0, 0.0])
    big, small = r[r.zone_id == 0].iloc[0], r[r.zone_id == 1]
    assert big.recommendation == "act" and big.net_benefit_rs > 0
    assert (small.recommendation == "monitor").all()                               # loss below intervention cost
    assert (r.sort_values("net_benefit_rs", ascending=False).index == r.index).all()


def test_action_inference_and_cost_model():
    assert infer_action("avail") == "supply" and infer_action("price,share") == "price" and infer_action("share") == "ads"
    assert infer_action("focal_units") == "supply" and infer_action("avail,price") == "supply"
    r_lo, _ = _rank_setup([30.0], signals=("price",))
    cheap = {k: ranker_mod.Action(v.name, 0.0, 0.0, 0.0, v.recoverable) for k, v in ACTIONS.items()}
    sig_x = weekly(T=60, S=1, base=50.0, amp=0.0, noise=0.3, seed=9)
    sig_x[0, 40:] -= 30.0
    sig = _signals(sig_x)
    det = ConformalDetector()
    _, inc = build_incidents(det.run(sig))
    r_free = rank_incidents(inc.assign(signals="avail"), det.bands, sig, _obs(Z=1), RankerCfg(actions=cheap))
    assert (r_free.intervention_cost_rs == 0).all() and (r_free.net_benefit_rs >= r_lo.net_benefit_rs.min()).all()


def test_ranker_uses_only_observed_tables():
    src = inspect.getsource(ranker_mod)
    assert "latent_" not in src
    import ml.conformal as c
    import ml.incidents as i
    assert "latent_" not in inspect.getsource(c) + inspect.getsource(i)


# ---------------------------------------------------------------- eval on world_v1 (held out from design choices)
@pytest.fixture(scope="module")
def mev():
    if not (DATA / "manifest.json").exists():
        pytest.skip("data/world_v1 not generated")
    if not json.loads((DATA / "manifest.json").read_text()).get("latent_included"):
        pytest.skip("no latent tables")
    import tempfile
    from ml.model_eval import run
    with tempfile.TemporaryDirectory() as td:
        return run(str(DATA), td, docs=None)


def test_eval_conformal_hard_negatives_and_calibration(mev):
    c = mev["detection"]["conformal"]
    assert c["false_alert_rate_E4"] == 0.0 and c["false_alert_rate_E8"] == 0.0
    assert c["recall_above_floor"] >= 0.7 and c["false_alarms_per_1000_series_days"] < 2.0
    for r in mev["coverage"]:
        assert 0.92 <= r["coverage"] <= 0.995, r


def test_eval_ranker_beats_week2_ordering_and_policy_beats_acting_on_everything(mev):
    rk, pol = mev["ranking"], mev["policy"]
    ml, w2 = rk["ML: net benefit (Rs)"], rk["Week 2 incidents by z strength"]
    assert ml["capture_at_k"] > w2["capture_at_k"] and ml["spearman_vs_true"] > w2["spearman_vs_true"]
    assert ml["realized_net_rs_at_k"] > w2["realized_net_rs_at_k"]
    assert pol["ml_net_benefit_gt0"]["precision"] >= 0.6
    assert pol["ml_net_benefit_gt0"]["realized_net_rs"] > pol["act_on_all_incidents"]["realized_net_rs"]
    assert mev["impact"]["spearman_units"] > 0.2
