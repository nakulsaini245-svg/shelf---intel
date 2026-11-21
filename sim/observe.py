"""Observation layer (scraper noise) and the search-visibility auction.

Scrape status codes per (serviceable pair, snapshot):
    0 ok | 1 timeout | 2 serviceability_error | 3 http_5xx (full-day blackout from an event hook)
Failure mechanisms mixed on purpose so downstream code cannot assume MCAR:
    MCAR   : p_timeout, independent of everything
    MNAR   : evening_extra applies to snapshots 3 and 4 only, scaled by platform flakiness
    stale  : a snapshot silently returns the previous snapshot's availability (status stays "ok";
             truth recorded only in latent_obs_flags)
    svc    : transient false "unserviceable" answers
The scraper-visible reason never reveals which mechanism fired.
"""
from __future__ import annotations

import warnings

import numpy as np

from .events import DayModifiers
from .seeds import rng_for
from .world import World

STATUS_NAMES = {0: "ok", 1: "failed", 2: "failed", 3: "failed"}
REASON_NAMES = {0: None, 1: "timeout", 2: "serviceability_error", 3: "http_5xx"}


def scrape_status(w: World, day: int, mods: DayModifiers) -> tuple[np.ndarray, np.ndarray]:
    """Returns (status [NP,4] int8, stale [NP,4] bool)."""
    ob = w.cfg.observation
    rng = rng_for(w.cfg.seed, "pair", day, "scrape")
    NP = w.n_pairs
    flak = np.array([p.flakiness for p in w.cfg.platforms])[w.pair_plat][:, None]
    evening = np.array([0.0, 0.0, 1.0, 1.0])[None, :]
    p_fail = np.clip(ob.p_timeout * flak + ob.evening_extra * flak * evening, 0, 1)
    u1, u2, u3 = rng.random((NP, 4)), rng.random((NP, 4)), rng.random((NP, 4))
    status = np.zeros((NP, 4), dtype=np.int8)
    status[u1 < p_fail] = 1
    status[(status == 0) & (u2 < ob.p_serviceability_error)] = 2
    if mods.scrape_blackout is not None:
        status[mods.scrape_blackout[w.pair_plat, w.pair_pin], :] = 3
    stale = (u3 < ob.p_stale) & (status == 0)
    return status, stale


class Visibility:
    """Top-N search results per (platform, pincode, query) with sponsored slots, plus brand share-of-voice."""

    def __init__(self, w: World):
        self.w = w
        c = w.cfg
        self.ad = c.ads
        NP, Q = w.n_pairs, w.Q
        self.G = NP * Q
        self.gp_pair = np.repeat(np.arange(NP), Q)
        self.gq = np.tile(np.arange(Q), NP)
        self.gp_plat = w.pair_plat[self.gp_pair]
        self.gp_pin = w.pair_pin[self.gp_pair]
        cs = w.cand_sku[self.gp_plat, self.gq]                      # [G,Cmax]
        valid = cs >= 0
        safe = np.where(valid, cs, 0)
        self.cell_g = w.cell_of[self.gp_plat[:, None], self.gp_pin[:, None], safe]
        self.valid0 = valid & (self.cell_g >= 0)
        self.cell_g = np.where(self.valid0, self.cell_g, 0)
        self.cs = safe
        self.brand_g = w.sku_brand[safe]
        self.log_pop = np.log(w.sku_pop[safe])
        self.pack_kg = w.sku_pack_kg[safe]
        self.strength = np.array([b.ad_strength for b in c.brands])
        self.partic = np.array([b.ad_participation for b in c.brands])
        self.plat_ad = np.array([p.ad_intensity for p in c.platforms])

    def run(self, day: int, avail_snap: np.ndarray, listed_today: np.ndarray, price_cell: np.ndarray,
            conv_ema: np.ndarray, mods: DayModifiers) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
        """Returns ({snapshot: slot arrays}, sov [NP,Q,B])."""
        w, ad = self.w, self.ad
        G, Cm = self.cs.shape
        snaps = ad.snapshots
        rng = rng_for(w.cfg.seed, "group", day, "auction")
        pacing = rng.lognormal(0.0, ad.pacing_sigma, (len(snaps), w.P, w.B))
        p_part = np.clip(self.partic[None, :, None] * self.plat_ad[:, None, None], 0, 1)   # [P,B,1]
        part = rng.random((w.P, w.B, w.Q)) < p_part
        g_sp = rng.gumbel(size=(len(snaps), G, Cm)) * ad.gumbel_sponsored
        g_or = rng.gumbel(size=(len(snaps), G, Cm)) * ad.gumbel_organic
        bidmult = mods.ad_bid_mult if mods.ad_bid_mult is not None else np.ones((w.P, w.B))
        n_sp = len(ad.sponsored_positions)
        out: dict[int, dict[str, np.ndarray]] = {}
        sov_acc = np.zeros((G, w.B))
        ar = np.arange(G)
        for si, snap in enumerate(snaps):
            shown = self.valid0 & listed_today[self.cell_g]
            av = avail_snap[snap - 1][self.cell_g] & shown
            ppk = np.where(shown, price_cell[self.cell_g] / self.pack_kg, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(ppk, axis=1, keepdims=True)
            bid = self.strength[self.brand_g] * pacing[si][self.gp_plat[:, None], self.brand_g] \
                * bidmult[self.gp_plat[:, None], self.brand_g]
            active = part[self.gp_plat[:, None], self.brand_g, self.gq[:, None]]
            with np.errstate(divide="ignore", invalid="ignore"):
                sp = np.where(shown & av & active, np.log(bid) + 0.5 * self.log_pop + g_sp[si], -np.inf)
                org = np.where(
                    shown,
                    self.log_pop + ad.w_conv * np.log(0.2 + conv_ema[self.cell_g])
                    - ad.w_price * np.log(ppk / med) - ad.oos_penalty * (~av) + g_or[si], -np.inf)
            sp_order = np.argsort(-sp, axis=1)[:, :n_sp]
            filled = np.isfinite(np.take_along_axis(sp, sp_order, 1))
            for j in range(n_sp):
                rows = np.where(filled[:, j])[0]
                org[rows, sp_order[rows, j]] = -np.inf
            org_order = np.argsort(-org, axis=1)[:, : ad.n_slots]
            org_ok = np.isfinite(np.take_along_axis(org, org_order, 1))
            slot = -np.ones((G, ad.n_slots), dtype=np.int64)
            is_sp = np.zeros((G, ad.n_slots), dtype=bool)
            ptr = np.zeros(G, dtype=np.int64)
            for pos in range(1, ad.n_slots + 1):
                if pos in ad.sponsored_positions:
                    j = ad.sponsored_positions.index(pos)
                    use_sp = filled[:, j]
                else:
                    j, use_sp = 0, np.zeros(G, dtype=bool)
                pi = np.minimum(ptr, ad.n_slots - 1)
                o_ok = (~use_sp) & org_ok[ar, pi]
                slot[:, pos - 1] = np.where(use_sp, sp_order[:, j], np.where(o_ok, org_order[ar, pi], -1))
                is_sp[:, pos - 1] = use_sp
                ptr = ptr + o_ok
            ok = slot >= 0
            sku = self.cs[ar[:, None], np.maximum(slot, 0)]
            brand = w.sku_brand[sku]
            flat = (ar[:, None] * w.B + brand)[ok]
            cnt = np.bincount(flat, minlength=G * w.B).reshape(G, w.B)
            sov_acc += cnt / np.maximum(ok.sum(1, keepdims=True), 1)
            gi, pos_i = np.where(ok)
            out[snap] = dict(pair=self.gp_pair[gi], query=self.gq[gi], position=(pos_i + 1),
                             sku=sku[gi, pos_i], is_sponsored=is_sp[gi, pos_i])
        sov = (sov_acc / len(snaps)).reshape(w.n_pairs, w.Q, w.B)
        return out, sov
