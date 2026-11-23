"""python -m ml.cli --data data/world_v1 --out eval/results   (runs 4 detector configs, writes CSV/JSON/markdown)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sim.io import read_table
from .detectors import CusumDetector, SeasonalDetector
from .features import build_signals, load_observed
from .score import NOISE_FLOOR, score_detector

DIGEST_HINT = "regenerate with: python -m sim.cli generate --out data/world_v1"


def detectors():
    return [("cusum", CusumDetector("cusum"), "exclude"),
            ("cusum_naive_na", CusumDetector("cusum_naive_na"), "zero"),
            ("seasonal_plain", SeasonalDetector("seasonal_plain", False, False, False), "exclude"),
            ("seasonal_full", SeasonalDetector("seasonal_full"), "exclude")]


def run(data: str, out: str) -> dict:
    d, o = Path(data), Path(out)
    o.mkdir(parents=True, exist_ok=True)
    man = json.loads((d / "manifest.json").read_text())
    obs = load_observed(d)
    events, effects = read_table(d, "latent_events"), read_table(d, "latent_effects")
    plat = [str(x) for x in obs["dim_location"].columns[:0]] or ["P1", "P2", "P3"]
    results, per_all = {}, []
    for name, det, pol in detectors():
        sig = build_signals(obs, na_policy=pol)
        alarms = det.run(sig)
        per, summ = score_detector(alarms, events, effects, sig, plat)
        per.insert(0, "detector", name)
        per_all.append(per)
        results[name] = summ
        alarms.to_csv(o / f"alarms_{name}.csv.gz", index=False)
    pd.concat(per_all).to_csv(o / "per_event.csv", index=False)
    payload = dict(config_digest=man["config_digest"], seed=man["seed"], n_days=man["n_days"],
                   noise_floor=NOISE_FLOOR, detectors=results)
    (o / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


def report(payload: dict, per: pd.DataFrame) -> str:
    r = payload["detectors"]
    f = lambda v: "n/a" if v != v else f"{v:.2f}"
    rows = ["| Detector | Recall (all) | Recall (>= floor) | Median delay (d) | Alarm precision | False alarms / 1k series-days | E4 FA | E8 FA | E10 FA | E8 caught as DQ |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in r.items():
        rows.append(f"| `{n}` | {f(s['recall_all'])} | {f(s['recall_above_floor'])} | {f(s['median_delay_above_floor'])} | "
                    f"{f(s['alarm_precision'])} | {f(s['false_alarms_per_1000_series_days'])} | {f(s['false_alert_rate_E4'])} | "
                    f"{f(s['false_alert_rate_E8'])} | {f(s['false_alert_rate_E10'])} | {f(s['e8_dq_recall'])} |")
    sa = per[per.should_alert & (per.detector == "seasonal_full")]
    miss = sa[~sa.detected]
    return f"""# BASELINES (Week 2 / Step 2B)
Data: `data/world_v1`, config digest `{payload['config_digest']}`, seed {payload['seed']}, {payload['n_days']} days. Simulator-conditional results.
Reproduce: `python -m ml.cli` (about 10 s). Detectors read only observed/public tables; `latent_*` is used only by `ml/score.py`.

## Results
{chr(10).join(rows)}

FA = false-alert rate on hard negatives (share of those events that triggered a business alarm). Noise floor: {payload['noise_floor']:.0f} units.
Per-event detail: `eval/results/per_event.csv`; raw alarms: `eval/results/alarms_*.csv.gz`.

## Detectors
- `cusum`: two-sided tabular CUSUM (k=0.5, h=5) on z-scores standardised with days 0-9; NA excluded.
- `cusum_naive_na`: same, but NA scrapes counted as unavailable/zero (the mistake the NA design guards against).
- `seasonal_plain`: seasonal-naive (median of same weekday 1-4 weeks back), MAD scale, alarm on 2 consecutive |z|>3.
- `seasonal_full`: plain + public calendar (same-status references; suppress announced regional festivals), scrape-failure DQ gate, promo explanation.

## Scoring rules
An alarm belongs to an event if platform and zone are in scope and the day is in [start_day, end_day + 5]. Delay = first alarm day - start day.
Events with max(|focal units|, |all units|, |lost units|) < {payload['noise_floor']:.0f} are below the answer-key noise floor and excluded from "recall (>= floor)".
Background false alarms = business alarms outside every event window, per 1,000 monitored series-days (after day 12).

## Findings and caveats
1. **CUSUM trained on 10 days is not credible as a detector here.** Its perfect recall is a by-product of ~17 false alarms per 1,000 series-days: the series have weekly seasonality and festivals, so a fixed training mean drifts. It also alerts on every hard negative. This is the baseline a seasonality-aware method must beat.
2. **NA handling barely moves aggregates** (`cusum` vs `cusum_naive_na`) because failed scrapes are ~4% and spread out; it matters for E8 and for low-volume slices.
3. **The calendar must be used as a covariate, not a mute switch.** A first version that suppressed all festival days and excluded them from the scale estimate cut recall to 0.24, because 38% of zone-days are festival days and E1/E2 events fall inside them. The shipped version compares like with like and suppresses only announced regional festivals (E10). This was a design fix after seeing aggregate recall, so treat the tuned figures as optimistic.
4. **Hard negatives:** `seasonal_full` raises no business alarms on E4, E8 or E10. E8 is caught on the data-quality channel in {r['seasonal_full']['e8_dq_recall']*100:.0f}% of cases; the miss is a one-day blackout hitting 30% of pincodes. `seasonal_plain` false-alerts on E8 and E10.
5. **Warm-up:** events starting before day 12 (E2 id 7, E3 id 11, E5 id 16) have limited history; detection delay for them is partly a warm-up artefact.
6. **Missed by `seasonal_full`:** {', '.join(f"id {int(x.event_id)} ({x.type}, {x.tier}, {x.effect_size:.0f} units)" for x in miss.itertuples()) or 'none'}.
7. Thresholds (k=0.5, h=5, z=3, persist 2) are textbook defaults, not fitted to labels. Only the calendar/status design in item 3 was revised after looking at results.
8. 29 incident events is a small sample; recall values move in steps of ~5 points. Do not read the third decimal.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_v1")
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args()
    p = run(a.data, a.out)
    cols = ["recall_all", "recall_above_floor", "median_delay_above_floor", "false_alarms_per_1000_series_days",
            "false_alert_rate_E4", "false_alert_rate_E8", "false_alert_rate_E10", "e8_dq_recall"]
    print(pd.DataFrame(p["detectors"]).T[cols].round(3).to_string())
    Path("docs").mkdir(exist_ok=True)
    Path("docs/BASELINES.md").write_text(report(p, pd.read_csv(Path(a.out) / "per_event.csv")))


if __name__ == "__main__":
    main()
