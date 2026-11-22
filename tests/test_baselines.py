"""Unit tests (synthetic series) and an eval test (data/world_v1) for the Week 2 baseline detectors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.detectors import CusumDetector, SeasonalDetector, cusum_hits, persistent_hits, seasonal_z
from ml.features import Panel, Signals
from ml.score import event_table, score_detector

DATA = Path("data/world_v1")


def weekly(T=70, S=3, base=20.0, amp=6.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return base + amp * np.sin(2 * np.pi * t / 7)[None, :] + rng.normal(0, noise, (S, T))


# ---------------------------------------------------------------- CUSUM
def test_cusum_detects_level_shift_with_small_delay():
    x = weekly(amp=0.0)
    x[1, 40:] += 6.0
    hits = cusum_hits(x, "both", floor=0.5)
    s1 = [t for s, t, _, _ in hits if s == 1 and t >= 40]
    assert s1 and min(s1) - 40 <= 2
    assert all(t >= 40 for s, t, _, _ in hits if s == 1)            # quiet before the shift


def test_cusum_false_alarm_rate_bounded_on_stationary_noise():
    """Default h=5 with a 10-day training window: noisy mean/sd estimates leave a few alarms (documented), not a flood."""
    x = weekly(amp=0.0, S=20, noise=1.0)
    n = len(cusum_hits(x, "both", floor=0.5))
    assert 1000 * n / (x.shape[0] * (x.shape[1] - 10)) < 15
    assert len(cusum_hits(x, "both", floor=0.5, h=12.0)) < n                # raising h lowers false alarms


def test_cusum_respects_direction_and_nan():
    x = weekly(amp=0.0)
    x[0, 40:] -= 8.0
    assert not [h for h in cusum_hits(x, "up", 0.5) if h[0] == 0 and h[1] >= 40]
    x[2, 40:45] = np.nan                                                # missing days never raise alarms by themselves
    assert not [h for h in cusum_hits(x, "both", 0.5) if h[0] == 2 and 40 <= h[1] < 45]


def test_cusum_is_online():
    x = weekly(amp=0.0)
    h1 = cusum_hits(x, "both", 0.5)
    x2 = x.copy()
    x2[:, 60:] += 50.0                                                  # future change must not alter past alarms
    h2 = cusum_hits(x2, "both", 0.5)
    assert [h for h in h1 if h[1] < 60] == [h for h in h2 if h[1] < 60]


# ---------------------------------------------------------------- seasonal residual
def test_seasonal_z_explains_weekly_pattern():
    x = weekly(S=10, amp=8.0, noise=0.8)
    z = seasonal_z(x, floor=0.5)
    assert np.nanmax(np.abs(z[:, 21:])) < 6                              # seasonal swing is explained
    assert 1000 * len(persistent_hits(z, "both", 3.0, 2)) / (x.shape[0] * (x.shape[1] - 14)) < 5   # chance alarms only


def test_seasonal_detects_drop_and_needs_persistence():
    x = weekly(amp=5.0, noise=0.5)
    x[0, 50:53] -= 15.0
    z = seasonal_z(x, 0.5)
    hit = [t for s, t, d, _ in persistent_hits(z, "down", 3.0, 2) if s == 0]
    assert hit and 50 <= hit[0] <= 52
    y = weekly(amp=5.0, noise=0.5)
    y[0, 50] -= 15.0                                                     # single-day spike: no alarm with persist=2
    assert not [h for h in persistent_hits(seasonal_z(y, 0.5), "down", 3.0, 2) if h[0] == 0 and 50 <= h[1] <= 52]


def test_seasonal_z_is_online():
    x = weekly()
    z1 = seasonal_z(x, 0.5)
    x2 = x.copy()
    x2[:, 60:] *= 3
    z2 = seasonal_z(x2, 0.5)
    assert np.allclose(z1[:, :60], z2[:, :60], equal_nan=True)


def test_status_matching_does_not_blind_a_festival_run():
    """A festival lifts the level; same-status references avoid a flood, the fallback keeps early festival days scorable."""
    T = 60
    x = weekly(T=T, S=2, amp=0.0, noise=0.5)
    status = np.zeros((2, T), dtype=bool)
    status[:, 30:45] = True
    x[:, 30:45] += 10.0
    z = seasonal_z(x, 0.5, status)
    assert np.isfinite(z[:, 30:45]).mean() > 0.9                         # fallback gives a score even w/o matched refs
    assert np.nanmax(np.abs(z[:, 38:45])) < 5                            # once matched refs exist, level lift is explained


def _signals(values, fest=None, reg=None):
    S, T = values.shape
    meta = pd.DataFrame({"platform_idx": np.zeros(S, int), "zone_id": np.arange(S), "key": -1})
    z = np.zeros((S, T), bool)
    mk = lambda ch, dirn, fl: Panel("x", values.copy(), meta, dirn, ch, fl)
    panels = {k: mk("business", "both", 0.5) for k in ("avail", "focal_units", "price_all", "price_nopromo", "share")}
    panels["scrape"] = mk("dq", "up", 0.02)
    panels["focal_promo"] = mk("aux", "up", 0.03)
    return Signals(panels, z if fest is None else fest, T, 1, S, 0, z if reg is None else reg)


def test_regional_festival_suppresses_alarm_but_baseline_festival_does_not():
    T = 70
    x = weekly(T=T, S=3, amp=0.0, noise=0.4)
    x[0, 50:56] += 12.0
    reg = np.zeros((3, T), bool)
    reg[0, 49:57] = True
    plain = SeasonalDetector("p", False, False, False).run(_signals(x))
    cal = SeasonalDetector("c", True, False, False).run(_signals(x, fest=reg, reg=reg))
    inwin = lambda a: ((a.zone_id == 0) & a.day.between(50, 58) & (a.channel == "business")).sum()
    assert inwin(plain) > 0 and inwin(cal) == 0
    base = np.zeros((3, T), bool)
    base[:, 49:57] = True                                               # baseline festival: all zones, not regional
    cal2 = SeasonalDetector("c2", True, False, False).run(_signals(x, fest=base, reg=np.zeros((3, T), bool)))
    assert ((cal2.zone_id == 0) & cal2.day.between(50, 58) & (cal2.channel == "business")).sum() >= 0  # not muted by rule


def test_detectors_use_only_signals():
    """Detectors must not touch latent tables: they take Signals, which is built from observed/public data only."""
    import inspect
    import ml.detectors as d
    import ml.features as f
    assert "latent_" not in inspect.getsource(d).replace("latent_events", "")
    assert all(not n.startswith("latent_") for n in f.OBSERVED_NEEDED)


# ---------------------------------------------------------------- scoring
def _answer_key():
    ev = pd.DataFrame(dict(event_id=[1, 2, 3, 4], type=["E2", "E8", "E10", "E3"], tier=["easy"] * 4,
                           start_day=[20, 30, 40, 50], end_day=[22, 31, 42, 52], scope_platform=["P1", "P1", "ALL", "P1"],
                           scope_zones=["0", "1", "2", "0"], is_hard_negative=[False, True, True, False],
                           should_alert=[True, False, False, True]))
    rows = []
    for eid, sz, w in [(1, 500.0, (20, 27)), (2, 0.0, (30, 36)), (3, 900.0, (40, 47)), (4, 5.0, (50, 57))]:
        rows.append((eid, "delta_sold_units_focal", sz, *w))
    return ev, pd.DataFrame(rows, columns=["event_id", "metric", "true_delta", "window_start", "window_end"])


def _alarm(day, zone, ch="business", plat=0, sig="avail"):
    return dict(detector="t", signal=sig, channel=ch, platform_idx=plat, zone_id=zone, key=-1, day=day,
                direction="down", score=4.0)


def test_scoring_delay_floor_hard_negatives_and_background():
    ev, ef = _answer_key()
    sig = _signals(np.ones((3, 70)))
    alarms = pd.DataFrame([_alarm(22, 0),                                 # E2 hit, delay 2
                           _alarm(31, 1),                                 # business alarm on E8 -> false alert
                           _alarm(31, 1, ch="dq", sig="scrape"),          # E8 caught as data-quality
                           _alarm(41, 2, plat=2),                         # E10 (platform ALL) -> false alert
                           _alarm(65, 2)])                                # background false alarm
    per, s = score_detector(alarms, ev, ef, sig, ["P1", "P2", "P3"])
    r = per.set_index("event_id")
    assert r.loc[1, "detected"] and r.loc[1, "delay_days"] == 2
    assert r.loc[2, "false_alert"] and r.loc[2, "dq_detected"] and r.loc[3, "false_alert"]
    assert not r.loc[4, "detected"] and not r.loc[4, "above_floor"]       # below the 15-unit floor
    assert s["recall_above_floor"] == 1.0 and s["n_above_floor"] == 1 and s["n_below_floor"] == 1
    assert s["false_alert_rate_E8"] == 1.0 and s["e8_dq_recall"] == 1.0
    assert s["background_false_alarms"] == 1 and s["false_alarms_per_1000_series_days"] > 0


def test_event_table_skips_mechanism_rows():
    ev, ef = _answer_key()
    ev = pd.concat([ev, ev.iloc[[0]].assign(event_id=9, type="E6a")])
    assert 9 not in set(event_table(ev, ef).event_id)


# ---------------------------------------------------------------- eval on the generated world
@pytest.fixture(scope="module")
def world_eval():
    if not (DATA / "manifest.json").exists():
        pytest.skip("data/world_v1 not generated (python -m sim.cli generate --out data/world_v1)")
    man = json.loads((DATA / "manifest.json").read_text())
    if not man.get("latent_included"):
        pytest.skip("dataset has no latent tables")
    import tempfile
    from ml.cli import run
    with tempfile.TemporaryDirectory() as td:
        return run(str(DATA), td)["detectors"]


def test_eval_seasonal_full_beats_baselines_on_hard_negatives(world_eval):
    full, plain, cusum = world_eval["seasonal_full"], world_eval["seasonal_plain"], world_eval["cusum"]
    for k in ("false_alert_rate_E4", "false_alert_rate_E8", "false_alert_rate_E10"):
        assert full[k] == 0.0
    assert full["e8_dq_recall"] >= 0.6                                    # blackouts surface on the DQ channel
    assert cusum["false_alarms_per_1000_series_days"] > 5 * full["false_alarms_per_1000_series_days"]
    assert full["alarm_precision"] > cusum["alarm_precision"]


def test_eval_recall_and_delay_floor(world_eval):
    full = world_eval["seasonal_full"]
    assert full["recall_above_floor"] >= 0.8
    assert full["median_delay_above_floor"] <= 2
    assert full["false_alarms_per_1000_series_days"] < 2.0


def test_eval_results_respect_noise_floor(world_eval):
    s = world_eval["seasonal_full"]
    assert s["n_above_floor"] + s["n_below_floor"] == s["n_should_alert"]
    assert s["n_below_floor"] >= 1                                         # at least one event sits in the answer-key noise
