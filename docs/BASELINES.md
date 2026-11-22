# BASELINES (Week 2 / Step 2B)
Data: `data/world_v1`, config digest `3f1b674dec220724`, seed 20261121, 90 days. Simulator-conditional results.
Reproduce: `python -m ml.cli` (about 10 s). Detectors read only observed/public tables; `latent_*` is used only by `ml/score.py`.

## Results
| Detector | Recall (all) | Recall (>= floor) | Median delay (d) | Alarm precision | False alarms / 1k series-days | E4 FA | E8 FA | E10 FA | E8 caught as DQ |
|---|---|---|---|---|---|---|---|---|---|
| `cusum` | 1.00 | 1.00 | 0.00 | 0.39 | 16.91 | 1.00 | 1.00 | 1.00 | 1.00 |
| `cusum_naive_na` | 1.00 | 1.00 | 0.00 | 0.39 | 16.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| `seasonal_plain` | 0.86 | 0.90 | 1.00 | 0.51 | 1.01 | 0.00 | 0.33 | 0.50 | 0.67 |
| `seasonal_full` | 0.90 | 0.95 | 1.00 | 0.67 | 0.51 | 0.00 | 0.00 | 0.00 | 0.67 |

FA = false-alert rate on hard negatives (share of those events that triggered a business alarm). Noise floor: 15 units.
Per-event detail: `eval/results/per_event.csv`; raw alarms: `eval/results/alarms_*.csv.gz`.

## Detectors
- `cusum`: two-sided tabular CUSUM (k=0.5, h=5) on z-scores standardised with days 0-9; NA excluded.
- `cusum_naive_na`: same, but NA scrapes counted as unavailable/zero (the mistake the NA design guards against).
- `seasonal_plain`: seasonal-naive (median of same weekday 1-4 weeks back), MAD scale, alarm on 2 consecutive |z|>3.
- `seasonal_full`: plain + public calendar (same-status references; suppress announced regional festivals), scrape-failure DQ gate, promo explanation.

## Scoring rules
An alarm belongs to an event if platform and zone are in scope and the day is in [start_day, end_day + 5]. Delay = first alarm day - start day.
Events with max(|focal units|, |all units|, |lost units|) < 15 are below the answer-key noise floor and excluded from "recall (>= floor)".
Background false alarms = business alarms outside every event window, per 1,000 monitored series-days (after day 12).

## Findings and caveats
1. **CUSUM trained on 10 days is not credible as a detector here.** Its perfect recall is a by-product of ~17 false alarms per 1,000 series-days: the series have weekly seasonality and festivals, so a fixed training mean drifts. It also alerts on every hard negative. This is the baseline a seasonality-aware method must beat.
2. **NA handling barely moves aggregates** (`cusum` vs `cusum_naive_na`) because failed scrapes are ~4% and spread out; it matters for E8 and for low-volume slices.
3. **The calendar must be used as a covariate, not a mute switch.** A first version that suppressed all festival days and excluded them from the scale estimate cut recall to 0.24, because 38% of zone-days are festival days and E1/E2 events fall inside them. The shipped version compares like with like and suppresses only announced regional festivals (E10). This was a design fix after seeing aggregate recall, so treat the tuned figures as optimistic.
4. **Hard negatives:** `seasonal_full` raises no business alarms on E4, E8 or E10. E8 is caught on the data-quality channel in 67% of cases; the miss is a one-day blackout hitting 30% of pincodes. `seasonal_plain` false-alerts on E8 and E10.
5. **Warm-up:** events starting before day 12 (E2 id 7, E3 id 11, E5 id 16) have limited history; detection delay for them is partly a warm-up artefact.
6. **Missed by `seasonal_full`:** id 10 (E3, hard, 9 units), id 18 (E6, medium, 56 units).
7. Thresholds (k=0.5, h=5, z=3, persist 2) are textbook defaults, not fitted to labels. Only the calendar/status design in item 3 was revised after looking at results.
8. 29 incident events is a small sample; recall values move in steps of ~5 points. Do not read the third decimal.
