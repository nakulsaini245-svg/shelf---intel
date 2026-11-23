"""Conformal anomaly scoring on the signal panels.

Point forecast: status-matched seasonal-naive expectation (as in Week 2; festival days compared with festival days).
Uncertainty: split-conformal interval per series. For day t the calibration set is the series' own absolute
residuals on days [t-gap-calib+1, t-gap]; the half-width q_t is the finite-sample-corrected (1-alpha) empirical
quantile (floored at the signal's noise floor). The `gap` keeps an ongoing incident's own residuals out of its interval.
Anomaly score s_t = resid_t / q_t, so |s|>1 means "outside the (1-alpha) interval". Alarm: `persist` consecutive
days outside the interval in the signal's direction. Same gates as Week 2 (regional festival, scrape-failure DQ gate,
promo explanation). Online: day t uses data <= t.

Exchangeability is approximate (autocorrelated residuals), so nominal coverage is checked empirically on event-free
series-days (`coverage_report`), not assumed.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .detectors import _alarms, _pz_day_mask, _zone_day_mask, persistent_hits, seasonal_z
from .features import Panel, Signals

LAGS = (7, 14, 21, 28)


def expected_series(x: np.ndarray, status: np.ndarray | None = None, lags=LAGS, start: int = 7) -> np.ndarray:
    S, T = x.shape
    E = np.full((S, T), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in range(start, T):
            ls = [l for l in lags if t - l >= 0]
            matched = np.stack([np.where(status[:, t - l] == status[:, t], x[:, t - l], np.nan) if status is not None
                                else x[:, t - l] for l in ls])
            e = np.nanmedian(matched, axis=0)
            if status is not None:
                raw = np.nanmedian(np.stack([x[:, t - l] for l in ls]), axis=0)
                e = np.where(np.isnan(e), raw, e)
            E[:, t] = e
    return E


def conformal_bands(x: np.ndarray, floor: float, status: np.ndarray | None = None, alpha: float = 0.05,
                    calib: int = 28, gap: int = 3, min_cal: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (expected, q, score) with shapes [S, T]; NaN where there is not enough calibration history."""
    E = expected_series(x, status)
    R = x - E
    S, T = x.shape
    q = np.full((S, T), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in range(T):
            hi = t - gap
            if hi < 0:
                continue
            cal = np.abs(R[:, max(0, hi - calib + 1):hi + 1])
            n = (~np.isnan(cal)).sum(1)
            ok = n >= min_cal
            if not ok.any():
                continue
            level = np.minimum(1.0, (1 - alpha) * (n + 1) / np.maximum(n, 1))        # finite-sample correction
            srt = np.sort(np.where(np.isnan(cal), np.inf, cal), axis=1)
            k = np.clip(np.ceil(level * n).astype(int) - 1, 0, cal.shape[1] - 1)
            qv = srt[np.arange(S), k]
            qv = np.where(np.isinf(qv), np.nan, qv)
            q[:, t] = np.where(ok, np.maximum(qv, floor), np.nan)
    return E, q, R / q


@dataclass
class ConformalDetector:
    name: str = "conformal"
    alpha: float = 0.05
    tol: float = 1.25          # alarm when the residual exceeds tol x the conformal half-width (chosen on a seed-777 world)
    persist: int = 2
    calendar_pad: int = 3
    promo_lookback: int = 7
    gap: int = 3
    use_gates: bool = True
    bands: dict = field(default_factory=dict, repr=False)      # populated by run(): panel -> (expected, q, score)

    def run(self, sig: Signals) -> pd.DataFrame:
        P, Z, T = sig.n_platforms, sig.n_zones, sig.n_days
        fest = sig.festival
        reg = sig.festival_regional if sig.festival_regional is not None else np.zeros_like(fest)
        padded = reg.copy()
        for k in range(1, self.calendar_pad + 1):
            padded[:, k:] |= reg[:, :-k]
        out, self.bands = [], {}
        # DQ channel: scrape-failure intervals; same series also gates business alarms
        dq = sig.panels["scrape"]
        E, q, s = conformal_bands(dq.values, dq.floor, None, self.alpha, gap=self.gap)
        self.bands["scrape"] = (E, q, s)
        gate = np.zeros((P, Z, T), dtype=bool)
        with np.errstate(invalid="ignore"):
            gate[dq.meta.platform_idx.to_numpy(), dq.meta.zone_id.to_numpy()] = np.nan_to_num(s > self.tol, nan=0).astype(bool)
        out.append(_alarms(self.name, dq, "scrape", persistent_hits(s, "up", self.tol, self.persist)))
        promo_expl = np.zeros((P, Z, T), dtype=bool)
        fp = sig.panels["focal_promo"]
        zp = seasonal_z(fp.values, fp.floor)
        with np.errstate(invalid="ignore"):
            spike = np.nan_to_num(zp > 2.0, nan=0).astype(bool)
        for i in range(len(fp.values)):
            pi, zi = int(fp.meta.platform_idx.iat[i]), int(fp.meta.zone_id.iat[i])
            for t in np.where(spike[i])[0]:
                promo_expl[pi, zi, t:t + self.promo_lookback + 1] = True
        for key, nm in (("avail", "avail"), ("focal_units", "focal_units"), ("price_nopromo", "price"), ("share", "share")):
            p = sig.panels[key]
            status = _zone_day_mask(p, fest)
            E, q, s = conformal_bands(p.values, p.floor, status, self.alpha, gap=self.gap)
            self.bands[nm] = (E, q, s)
            suppress = _zone_day_mask(p, padded)
            if self.use_gates:
                if nm != "focal_units":
                    suppress = suppress | _pz_day_mask(p, gate)
                else:
                    suppress = suppress | _pz_day_mask(p, promo_expl)
            out.append(_alarms(self.name, p, nm, persistent_hits(s, p.direction, self.tol, self.persist, suppress=suppress)))
        return pd.concat(out, ignore_index=True)


def coverage_report(bands: dict, sig: Signals, event_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Empirical share of series-days INSIDE the interval (nominal 1-alpha). `event_mask` [P,Z,T] excludes event windows."""
    rows = []
    for nm, (E, q, s) in bands.items():
        p = sig.panels["scrape" if nm == "scrape" else {"price": "price_nopromo"}.get(nm, nm)]
        valid = ~np.isnan(s)
        if event_mask is not None:
            valid &= ~event_mask[p.meta.platform_idx.to_numpy(), p.meta.zone_id.to_numpy()]
        rows.append((nm, int(valid.sum()), float((np.abs(s[valid]) <= 1.0).mean())))
    return pd.DataFrame(rows, columns=["signal", "n", "coverage"])
