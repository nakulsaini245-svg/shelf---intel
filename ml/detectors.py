"""Baseline change detectors over signal panels.

1. CUSUM: tabular two-sided CUSUM on z-scores standardised with the first `train_days` (event-free by design).
2. Seasonal residual: seasonal-naive robust expectation (median of same-weekday values 1-4 weeks back),
   robust rolling scale (MAD), alarm on `persist` consecutive exceedances. Optional awareness:
     use_calendar  - public dim_calendar: each day is compared only with same-status reference days
                     (festival vs festival, normal vs normal); alarms are suppressed during announced
                     REGIONAL festivals (+ `calendar_pad` days after). Baseline festivals are not suppressed:
                     a stockout during Diwali is still a stockout.
     use_dq_gate   - observation-dependent signals (availability/price/share) are muted on days the scrape-failure
                     signal is anomalous (the system should say "data problem", not "incident")
     use_promo     - price signal drops promo-flagged rows; focal sell-out drops are explained away for
                     `promo_lookback` days after an unusual rise in the focal promo share

Both consume only `Signals` (observed/public data). Alarm days are the day the rule fires; an alarm on day t
uses data up to and including t only (online).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import Panel, Signals

ALARM_COLS = ["detector", "signal", "channel", "platform_idx", "zone_id", "key", "day", "direction", "score"]


def _alarms(detector: str, panel: Panel, name: str, hits: list[tuple[int, int, str, float]]) -> pd.DataFrame:
    rows = [(detector, name, panel.channel, int(panel.meta.platform_idx.iat[s]), int(panel.meta.zone_id.iat[s]),
             int(panel.meta.key.iat[s]), int(t), d, float(sc)) for s, t, d, sc in hits]
    return pd.DataFrame(rows, columns=ALARM_COLS)


# ------------------------------------------------------------------------------------------------ CUSUM
def cusum_hits(x: np.ndarray, direction: str, floor: float, train_days: int = 10, k: float = 0.5,
               h: float = 5.0, rel_floor: float = 0.05) -> list[tuple[int, int, str, float]]:
    mu = np.nanmean(x[:, :train_days], axis=1)
    sd = np.nanstd(x[:, :train_days], axis=1, ddof=1)
    scale = np.maximum.reduce([np.nan_to_num(sd), np.full_like(mu, floor), rel_floor * np.abs(np.nan_to_num(mu))])
    z = (x - mu[:, None]) / scale[:, None]
    up = np.zeros(len(x))
    dn = np.zeros(len(x))
    hits = []
    for t in range(train_days, x.shape[1]):
        zt = z[:, t]
        ok = ~np.isnan(zt)
        up = np.where(ok, np.maximum(0.0, up + zt - k), up)
        dn = np.where(ok, np.maximum(0.0, dn - zt - k), dn)
        for s in np.where(ok)[0]:
            fire_up = direction in ("up", "both") and up[s] > h
            fire_dn = direction in ("down", "both") and dn[s] > h
            if fire_up or fire_dn:
                hits.append((int(s), t, "up" if fire_up and (not fire_dn or up[s] >= dn[s]) else "down",
                             float(max(up[s] if fire_up else 0, dn[s] if fire_dn else 0))))
                up[s] = dn[s] = 0.0
    return hits


@dataclass
class CusumDetector:
    name: str = "cusum"
    train_days: int = 10
    k: float = 0.5
    h: float = 5.0

    def run(self, sig: Signals) -> pd.DataFrame:
        out = []
        for key in ("avail", "focal_units", "price_all", "share", "scrape"):
            p = sig.panels[key]
            out.append(_alarms(self.name, p, key.replace("_all", ""),
                               cusum_hits(p.values, p.direction, p.floor, self.train_days, self.k, self.h)))
        return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------------- seasonal residual
def seasonal_z(x: np.ndarray, floor: float, status: np.ndarray | None = None, lags=(7, 14, 21, 28),
               start: int = 7, scale_window: int = 28, min_hist: int = 5) -> np.ndarray:
    """Robust z-score of each day vs a seasonal-naive expectation (refs restricted to same-`status` days).
    z[:, t] uses data <= t only."""
    S, T = x.shape
    resid = np.full((S, T), np.nan)
    z = np.full((S, T), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in range(start, T):
            refs = np.stack([np.where(status[:, t - l] == status[:, t], x[:, t - l], np.nan) if status is not None
                             else x[:, t - l] for l in lags if t - l >= 0])
            expected = np.nanmedian(refs, axis=0)
            if status is not None:                      # no same-status reference yet (first week of a festival run)
                raw = np.nanmedian(np.stack([x[:, t - l] for l in lags if t - l >= 0]), axis=0)
                expected = np.where(np.isnan(expected), raw, expected)
            resid[:, t] = x[:, t] - expected
            hist = resid[:, max(0, t - scale_window):t]
            n = (~np.isnan(hist)).sum(1)
            med = np.nanmedian(hist, axis=1, keepdims=True)
            mad = 1.4826 * np.nanmedian(np.abs(hist - med), axis=1)
            scale = np.maximum(np.nan_to_num(mad), floor)
            z[:, t] = np.where(n >= min_hist, (x[:, t] - expected) / scale, np.nan)
    return z


def persistent_hits(z: np.ndarray, direction: str, thr: float = 3.0, persist: int = 2, cooldown: int = 3,
                    suppress: np.ndarray | None = None) -> list[tuple[int, int, str, float]]:
    S, T = z.shape
    hits = []
    for s in range(S):
        last = -10**9
        for t in range(persist - 1, T):
            w = z[s, t - persist + 1:t + 1]
            if np.isnan(w).any() or t - last <= cooldown:
                continue
            for d, ok in (("up", direction in ("up", "both") and (w > thr).all()),
                          ("down", direction in ("down", "both") and (w < -thr).all())):
                if ok and not (suppress is not None and suppress[s, t]):
                    hits.append((s, t, d, float(np.abs(w).min())))
                    last = t
                    break
    return hits


def _zone_day_mask(panel: Panel, zd: np.ndarray) -> np.ndarray:
    """Broadcast a [Z,T] mask onto a panel's series via its zone."""
    return zd[panel.meta.zone_id.to_numpy()]


def _pz_day_mask(panel: Panel, pz: np.ndarray) -> np.ndarray:
    """Broadcast a [P,Z,T] mask onto a panel's series via (platform, zone)."""
    return pz[panel.meta.platform_idx.to_numpy(), panel.meta.zone_id.to_numpy()]


@dataclass
class SeasonalDetector:
    name: str = "seasonal"
    use_calendar: bool = True
    use_dq_gate: bool = True
    use_promo: bool = True
    z_thr: float = 3.0
    persist: int = 2
    calendar_pad: int = 3
    promo_lookback: int = 7

    def run(self, sig: Signals) -> pd.DataFrame:
        P, Z, T = sig.n_platforms, sig.n_zones, sig.n_days
        fest = sig.festival
        regional = sig.festival_regional if sig.festival_regional is not None else np.zeros_like(fest)
        padded = regional.copy()
        for k in range(1, self.calendar_pad + 1):
            padded[:, k:] |= regional[:, :-k]
        out = []
        # data-quality gate: platform x zone x day where the scrape-failure rate is anomalous
        dq = sig.panels["scrape"]
        zdq = seasonal_z(dq.values, dq.floor)
        gate = np.zeros((P, Z, T), dtype=bool)
        with np.errstate(invalid="ignore"):
            hit = zdq > self.z_thr
        gate[dq.meta.platform_idx.to_numpy(), dq.meta.zone_id.to_numpy()] = np.nan_to_num(hit, nan=0).astype(bool)
        out.append(_alarms(self.name, dq, "scrape", persistent_hits(zdq, "up", self.z_thr, self.persist)))
        # promo explanation: unusual rise in focal promo share within the last `promo_lookback` days
        promo_expl = np.zeros((P, Z, T), dtype=bool)
        if self.use_promo:
            fp = sig.panels["focal_promo"]
            zp = seasonal_z(fp.values, fp.floor)
            with np.errstate(invalid="ignore"):
                spike = np.nan_to_num(zp > 2.0, nan=0).astype(bool)
            for i in range(len(fp.values)):
                pi, zi = int(fp.meta.platform_idx.iat[i]), int(fp.meta.zone_id.iat[i])
                for t in np.where(spike[i])[0]:
                    promo_expl[pi, zi, t:t + self.promo_lookback + 1] = True
        for key, nm in (("avail", "avail"), ("focal_units", "focal_units"),
                        ("price_nopromo" if self.use_promo else "price_all", "price"), ("share", "share")):
            p = sig.panels[key]
            status = _zone_day_mask(p, fest) if self.use_calendar else None
            suppress = np.zeros(p.values.shape, dtype=bool)
            if self.use_calendar:
                suppress |= _zone_day_mask(p, padded)
            if self.use_dq_gate and nm != "focal_units":      # first-party sell-out is immune to scraping
                suppress |= _pz_day_mask(p, gate)
            if self.use_promo and nm == "focal_units":
                suppress |= _pz_day_mask(p, promo_expl)
            z = seasonal_z(p.values, p.floor, status)
            out.append(_alarms(self.name, p, nm, persistent_hits(z, p.direction, self.z_thr, self.persist,
                                                                  suppress=suppress)))
        return pd.concat(out, ignore_index=True)
