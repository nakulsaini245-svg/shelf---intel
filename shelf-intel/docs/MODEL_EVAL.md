# MODEL_EVAL (Week 3 / Step 3A)
Data `data/world_v1`, digest `3f1b674dec220724`, seed 20261121. Synthetic, simulator-conditional. Reproduce: `python -m ml.model_eval` (about 15 s).
`world_v1` was **held out** from every design choice below; the one tuned knob (`tol`) was chosen on a separate seed-777 world.

## 1. Detection vs Week 2 baseline
| Detector | Recall (all) | Recall (>= floor) | Median delay (d) | Alarm precision | False alarms / 1k series-days | E4 FA | E8 FA | E10 FA | E8 as DQ |
|---|---|---|---|---|---|---|---|---|---|
| seasonal_full (Week 2) | 0.90 | 0.95 | 1.00 | 0.67 | 0.51 | 0.00 | 0.00 | 0.00 | 0.67 |
| conformal | 0.76 | 0.80 | 1.00 | 0.65 | 0.40 | 0.00 | 0.00 | 0.50 | 0.67 |

FA = hard-negative false-alert rate. Noise floor 15 units. Conformal scores need >= 10 days of calibration history, so its first day is about 14 (Week 2: 12).

**Read this plainly:** the conformal scorer is *not* a detection upgrade. On held-out data it matches or trails `seasonal_full` on recall and is not safer on hard negatives
(see E10). What it adds is a calibrated, per-series interval, which the ranker uses to put bounds on revenue at risk.

Selection run (seed-777 world, 35 events, used only to choose `tol=1.25`): `seasonal_full` recall(>=floor) 0.60, 0.08 FA/1k; conformal tol 1.0 / 1.25 / 1.5 =
0.75 / 0.65 / 0.50 recall with 0.96 / 0.25 / 0.00 FA/1k (tol 1.0 false-alerts E8). Rule fixed in advance: highest recall with zero hard-negative alerts and <= 0.5 FA/1k.
Incident-level corroboration (>= 2 signals or strength >= 1.75-3.0) was tried and rejected: recall fell to 0.30-0.45. The code stays in `ml/incidents.py` as an option.

## 2. Interval calibration (conformal, alpha = 0.05)
| Signal | Series-days | Empirical coverage (nominal 0.95) |
|---|---|---|
| scrape | 1063 | 0.980 |
| avail | 1063 | 0.961 |
| focal_units | 1063 | 0.970 |
| price | 5315 | 0.957 |
| share | 8504 | 0.963 |

Coverage is measured on series-days outside every event window (plus 8 days). Values below 0.95 mean the interval is optimistic (autocorrelated residuals break exchangeability).

## 3. Revenue-at-risk ranking (38 incidents, top-10 budget)
Value of an action = assumed recoverable share of the incident's **true** focal GMV loss (answer key, actionable events only) minus its assumed intervention cost.
Oracle top-10 true actionable loss: Rs 152,648.
| Ordering | Spearman vs true loss | Capture@10 | Precision@10 | Wasted actions@10 | Realized net Rs @10 |
|---|---|---|---|---|---|
| ML: net benefit (Rs) | 0.45 | 0.68 | 0.70 | 3 | -13,006 |
| ML: net benefit lower bound | 0.44 | 0.59 | 0.60 | 4 | -13,539 |
| ML: est. lost Rs (no cost) | 0.42 | 0.59 | 0.60 | 4 | -49,054 |
| conformal alarm count | 0.31 | 0.65 | 0.80 | 2 | -24,273 |
| conformal peak strength | 0.48 | 0.61 | 0.80 | 2 | -59,606 |
| Week 2 incidents by z strength | 0.24 | 0.31 | 0.50 | 5 | -113,060 |

Decision policy (no budget cap; act on every incident the rule selects):
| Policy | Actions | Precision | Wasted | Realized net Rs | True loss left unaddressed (Rs) |
|---|---|---|---|---|---|
| ml_net_benefit_gt0 | 7 | 0.71 | 2 | -931 | 82,081 |
| act_on_all_incidents | 38 | 0.42 | 22 | -379,354 | 0 |
| week2_all_incidents | 50 | 0.38 | 31 | -420,414 | 0 |

Impact estimate on matched actionable incidents with >= 15 true units (n = 16): median est/true units = 1.19, Spearman = 0.49,
share whose true loss falls inside the [low, high] interval = 1.00. Estimates are made **online** (detection day + 1, plus a 4-day projection at the current rate), so they are partial-information estimates. The interval covering every true loss with n = 16 is partly because it is wide; do not read it as tight calibration.

## 4. Findings
1. **Detection is at best parity.** Conformal trades recall (0.80 vs 0.95 above the floor) for slightly fewer background alarms, and lets one E10 through. Week 2 `seasonal_full` remains the better detector on this world.
2. **Intervals are well calibrated** (0.96-0.98 empirical coverage against 0.95 nominal on event-free days). That is the real addition, and it feeds the loss bounds.
3. **Ranking is where the layer pays off.** Cost-aware net-benefit ordering has the best capture@10 and realized net of the five orderings and clearly beats ranking Week 2 alarms by z strength (capture 0.68 vs 0.31). It does not win everything: ranking by conformal peak strength has similar rank correlation and higher precision@10, but wastes far more money because it ignores cost (Rs -59.6k vs -13.0k).
4. **Top-10 realized net is negative for every ordering** because a fixed budget of 10 forces actions that the rule itself would not take (only 7 incidents have positive estimated net). The *policy* row (act only if estimated net > 0) is break-even (about Rs -0.9k) versus Rs -379k for acting on everything, but it leaves Rs 82k of true loss unaddressed (under-estimated or below-cost incidents).
5. The ranker and the answer key share no code path: estimates use observed data only; `latent_*` is read only in `ml/model_eval.py`.

## 5. Assumptions and limits
- Action costs and recoverable shares (`ml/ranker.py: ACTIONS`) are illustrative assumptions, not measured. Rankings change with them; the *orderings* are compared under the same assumptions.
- True loss uses focal first-party GMV only (the brand's own feed); competitor-side or halo effects are ignored.
- Hard negatives get zero actionable value, so an incident on E4/E8/E10 counts as a wasted action.
- SKU / pincode drill-down (`top_skus`, `top_pincodes`) is descriptive and not scored here.
- 29 events is a small sample; differences of a few points in the tables are not meaningful.
