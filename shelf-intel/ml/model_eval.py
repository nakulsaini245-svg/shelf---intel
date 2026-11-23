"""python -m ml.model_eval  -> eval/results/model_eval.json, ranked_incidents.csv, docs/MODEL_EVAL.md

Compares (a) detection: Week 2 `seasonal_full` vs conformal scorer, (b) conformal interval coverage on event-free days,
(c) ranking: five orderings of the same incidents scored against the answer key (value of acting = recoverable share of
true focal GMV loss minus the intervention cost). Answer-key tables are used ONLY here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sim.io import read_table
from .conformal import ConformalDetector, coverage_report
from .detectors import SeasonalDetector
from .features import build_signals, load_observed
from .incidents import build_incidents
from .ranker import ACTIONS, RankerCfg, rank_incidents
from .score import event_table, score_detector

PLAT = ["P1", "P2", "P3"]
TOP_K = 10


def truth_table(events: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    ev = event_table(events, effects)
    piv = effects.pivot_table(index="event_id", columns="metric", values="true_delta", aggfunc="first")
    ev["true_units_loss"] = ev["event_id"].map(-piv["delta_sold_units_focal"]).clip(lower=0).fillna(0.0)
    ev["true_rs_loss"] = ev["event_id"].map(-piv["delta_gmv_focal"]).clip(lower=0).fillna(0.0)
    ev["actionable_rs"] = np.where(ev["should_alert"], ev["true_rs_loss"], 0.0)
    return ev


def match_incidents(inc: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Each incident -> the overlapping in-scope event with most overlap (or none). Duplicates get no extra value."""
    ev_id, ev_type, ev_alert, ev_val, ev_units = [], [], [], [], []
    for r in inc.itertuples():
        best, bo = None, 0
        for e in ev.itertuples():
            zones = {int(z) for z in str(e.scope_zones).split(",")}
            if int(r.zone_id) not in zones or (e.scope_platform != "ALL" and PLAT.index(e.scope_platform) != r.platform_idx):
                continue
            o = min(int(r.last_day), int(e.window_end)) - max(int(r.first_day), int(e.start_day)) + 1
            if o > bo:
                best, bo = e, o
        ev_id.append(None if best is None else int(best.event_id))
        ev_type.append(None if best is None else best.type)
        ev_alert.append(bool(best.should_alert) if best is not None else False)
        ev_val.append(float(best.actionable_rs) if best is not None else 0.0)
        ev_units.append(float(best.true_units_loss) if best is not None and best.should_alert else 0.0)
    out = inc.copy()
    out["event_id"], out["event_type"], out["is_actionable"] = ev_id, ev_type, ev_alert
    out["true_rs"], out["true_units"] = ev_val, ev_units
    out = out.sort_values("first_day")                                            # earliest incident keeps the event's value
    dup = out.duplicated("event_id", keep="first") & out["event_id"].notna()
    out.loc[dup, ["true_rs", "true_units"]] = 0.0
    out["duplicate"] = dup
    return out.sort_index()


def ranking_metrics(r: pd.DataFrame, key: str, oracle: np.ndarray, k: int = TOP_K) -> dict:
    d = r.sort_values(key, ascending=False).reset_index(drop=True)
    top = d.head(k)
    rec = np.array([ACTIONS[a].recoverable for a in top["action"]])
    realized = float((rec * top["true_rs"].to_numpy() - top["intervention_cost_rs"].to_numpy()).sum())
    return dict(
        spearman_vs_true=float(d[key].rank().corr(d["true_rs"].rank())),
        capture_at_k=float(top["true_rs"].sum() / oracle[:k].sum()) if oracle[:k].sum() > 0 else float("nan"),
        precision_at_k=float((top["is_actionable"] & ~top["duplicate"]).mean()),
        realized_net_rs_at_k=realized,
        wasted_actions_at_k=int((~(top["is_actionable"] & ~top["duplicate"])).sum()))


def policy_metrics(r: pd.DataFrame) -> dict:
    """The deployed decision rule: act on every incident whose estimated net benefit is positive."""
    a = r[r["recommendation"] == "act"]
    rec = np.array([ACTIONS[x].recoverable for x in a["action"]])
    ok = a["is_actionable"] & ~a["duplicate"]
    return dict(n_actions=int(len(a)), precision=float(ok.mean()) if len(a) else float("nan"),
                wasted_actions=int((~ok).sum()), realized_net_rs=float((rec * a["true_rs"].to_numpy() - a["intervention_cost_rs"].to_numpy()).sum()),
                missed_true_rs=float(r.loc[~r.index.isin(a.index), "true_rs"].sum()))


def run(data: str, out: str, docs: str | None = "docs/MODEL_EVAL.md") -> dict:
    d, o = Path(data), Path(out)
    o.mkdir(parents=True, exist_ok=True)
    man = json.loads((d / "manifest.json").read_text())
    obs = load_observed(d)
    events, effects = read_table(d, "latent_events"), read_table(d, "latent_effects")
    sig = build_signals(obs)
    base_det, conf = SeasonalDetector("seasonal_full"), ConformalDetector()
    a_base, a_conf = base_det.run(sig), conf.run(sig)
    det = {}
    for name, al in (("seasonal_full (Week 2)", a_base), ("conformal", a_conf)):
        per, s = score_detector(al, events, effects, sig, PLAT)
        det[name] = s
    # --- coverage on event-free series-days (answer key used only to mask event windows)
    ev = truth_table(events, effects)
    P, Z, T = sig.n_platforms, sig.n_zones, sig.n_days
    mask = np.zeros((P, Z, T), dtype=bool)
    for e in ev.itertuples():
        zs = [int(z) for z in str(e.scope_zones).split(",")]
        ps = range(P) if e.scope_platform == "ALL" else [PLAT.index(e.scope_platform)]
        for p in ps:
            for z in zs:
                mask[p, z, int(e.start_day):int(e.window_end) + 8] = True
    cov = coverage_report(conf.bands, sig, mask)
    # --- ranking
    _, inc_c = build_incidents(a_conf)
    ranked = rank_incidents(inc_c, conf.bands, sig, obs)
    m = match_incidents(ranked, ev).sort_values("rank")
    nalarm = build_incidents(a_conf)[0].groupby("incident_id").size()
    m["n_alarms"] = m["incident_id"].map(nalarm).astype(float)
    _, inc_b = build_incidents(a_base)
    base = match_incidents(rank_incidents(inc_b, conf.bands, sig, obs), ev)       # Week 2 incidents, ordered by z strength
    oracle = np.sort(ev["actionable_rs"].to_numpy())[::-1]
    rk = {"ML: net benefit (Rs)": ranking_metrics(m, "net_benefit_rs", oracle),
          "ML: net benefit lower bound": ranking_metrics(m, "net_benefit_low_rs", oracle),
          "ML: est. lost Rs (no cost)": ranking_metrics(m, "est_lost_rs", oracle),
          "conformal alarm count": ranking_metrics(m, "n_alarms", oracle),
          "conformal peak strength": ranking_metrics(m, "peak_strength", oracle),
          "Week 2 incidents by z strength": ranking_metrics(base, "peak_strength", oracle)}
    policy = dict(ml_net_benefit_gt0=policy_metrics(m),
                  act_on_all_incidents=policy_metrics(m.assign(recommendation="act")),
                  week2_all_incidents=policy_metrics(base.assign(recommendation="act")))
    # --- impact estimate quality on matched, actionable, above-floor incidents
    q = m[m["is_actionable"] & ~m["duplicate"] & (m["true_units"] >= 15)]
    imp = dict(n=int(len(q)),
               median_est_over_true=float((q["est_lost_units"] / q["true_units"]).median()) if len(q) else float("nan"),
               interval_covers_true=float(((q["true_units"] >= q["est_lost_units_low"]) & (q["true_units"] <= q["est_lost_units_high"])).mean()) if len(q) else float("nan"),
               spearman_units=float(q["est_lost_units"].rank().corr(q["true_units"].rank())) if len(q) > 2 else float("nan"))
    k = min(TOP_K, len(m))
    payload = dict(config_digest=man["config_digest"], seed=man["seed"], n_incidents=int(len(m)), top_k=k,
                   oracle_top_k_rs=float(oracle[:k].sum()), detection=det, coverage=cov.to_dict("records"),
                   ranking=rk, policy=policy, impact=imp, tol=conf.tol, alpha=conf.alpha)
    (o / "model_eval.json").write_text(json.dumps(payload, indent=2, default=float))
    m.drop(columns=["duplicate"]).to_csv(o / "ranked_incidents.csv", index=False)
    if docs:
        Path(docs).parent.mkdir(exist_ok=True)
        Path(docs).write_text(report(payload))
    return payload


def report(p: dict) -> str:
    f = lambda v, n=2: "n/a" if v != v else f"{v:.{n}f}"
    det = ["| Detector | Recall (all) | Recall (>= floor) | Median delay (d) | Alarm precision | False alarms / 1k series-days | E4 FA | E8 FA | E10 FA | E8 as DQ |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in p["detection"].items():
        det.append(f"| {n} | {f(s['recall_all'])} | {f(s['recall_above_floor'])} | {f(s['median_delay_above_floor'])} | {f(s['alarm_precision'])} | "
                   f"{f(s['false_alarms_per_1000_series_days'])} | {f(s['false_alert_rate_E4'])} | {f(s['false_alert_rate_E8'])} | "
                   f"{f(s['false_alert_rate_E10'])} | {f(s['e8_dq_recall'])} |")
    cov = ["| Signal | Series-days | Empirical coverage (nominal 0.95) |", "|---|---|---|"] + [
        f"| {c['signal']} | {c['n']} | {f(c['coverage'], 3)} |" for c in p["coverage"]]
    rk = ["| Ordering | Spearman vs true loss | Capture@10 | Precision@10 | Wasted actions@10 | Realized net Rs @10 |", "|---|---|---|---|---|---|"]
    for n, s in p["ranking"].items():
        rk.append(f"| {n} | {f(s['spearman_vs_true'])} | {f(s['capture_at_k'])} | {f(s['precision_at_k'])} | {s['wasted_actions_at_k']} | {s['realized_net_rs_at_k']:,.0f} |")
    pol = ["| Policy | Actions | Precision | Wasted | Realized net Rs | True loss left unaddressed (Rs) |", "|---|---|---|---|---|---|"]
    for n, s in p["policy"].items():
        pol.append(f"| {n} | {s['n_actions']} | {f(s['precision'])} | {s['wasted_actions']} | {s['realized_net_rs']:,.0f} | {s['missed_true_rs']:,.0f} |")
    i = p["impact"]
    return f"""# MODEL_EVAL (Week 3 / Step 3A)
Data `data/world_v1`, digest `{p['config_digest']}`, seed {p['seed']}. Synthetic, simulator-conditional. Reproduce: `python -m ml.model_eval` (about 15 s).
`world_v1` was **held out** from every design choice below; the one tuned knob (`tol`) was chosen on a separate seed-777 world.

## 1. Detection vs Week 2 baseline
{chr(10).join(det)}

FA = hard-negative false-alert rate. Noise floor 15 units. Conformal scores need >= 10 days of calibration history, so its first day is about 14 (Week 2: 12).

**Read this plainly:** the conformal scorer is *not* a detection upgrade. On held-out data it matches or trails `seasonal_full` on recall and is not safer on hard negatives
(see E10). What it adds is a calibrated, per-series interval, which the ranker uses to put bounds on revenue at risk.

Selection run (seed-777 world, 35 events, used only to choose `tol={p['tol']}`): `seasonal_full` recall(>=floor) 0.60, 0.08 FA/1k; conformal tol 1.0 / 1.25 / 1.5 =
0.75 / 0.65 / 0.50 recall with 0.96 / 0.25 / 0.00 FA/1k (tol 1.0 false-alerts E8). Rule fixed in advance: highest recall with zero hard-negative alerts and <= 0.5 FA/1k.
Incident-level corroboration (>= 2 signals or strength >= 1.75-3.0) was tried and rejected: recall fell to 0.30-0.45. The code stays in `ml/incidents.py` as an option.

## 2. Interval calibration (conformal, alpha = {p['alpha']})
{chr(10).join(cov)}

Coverage is measured on series-days outside every event window (plus 8 days). Values below 0.95 mean the interval is optimistic (autocorrelated residuals break exchangeability).

## 3. Revenue-at-risk ranking ({p['n_incidents']} incidents, top-{p['top_k']} budget)
Value of an action = assumed recoverable share of the incident's **true** focal GMV loss (answer key, actionable events only) minus its assumed intervention cost.
Oracle top-{p['top_k']} true actionable loss: Rs {p['oracle_top_k_rs']:,.0f}.
{chr(10).join(rk)}

Decision policy (no budget cap; act on every incident the rule selects):
{chr(10).join(pol)}

Impact estimate on matched actionable incidents with >= 15 true units (n = {i['n']}): median est/true units = {f(i['median_est_over_true'])}, Spearman = {f(i['spearman_units'])},
share whose true loss falls inside the [low, high] interval = {f(i['interval_covers_true'])}. Estimates are made **online** (detection day + 1, plus a 4-day projection at the current rate), so they are partial-information estimates. The interval covering every true loss with n = {i['n']} is partly because it is wide; do not read it as tight calibration.

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
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_v1")
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args()
    p = run(a.data, a.out)
    print(pd.DataFrame(p["detection"]).T[["recall_above_floor", "median_delay_above_floor", "false_alarms_per_1000_series_days",
                                         "false_alert_rate_E4", "false_alert_rate_E8", "false_alert_rate_E10"]].round(3).to_string())
    print(pd.DataFrame(p["coverage"]).round(3).to_string(index=False))
    print(pd.DataFrame(p["ranking"]).T.round(3).to_string())
    print(pd.DataFrame(p["policy"]).T.round(2).to_string())
    print(p["impact"])


if __name__ == "__main__":
    main()
