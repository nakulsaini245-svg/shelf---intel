"""Static world: dimensions, assortment, store network, serviceability, index maps.

Built once per (config, seed). All randomness comes from rng_for(..., WORLD_DAY, ...)
so the world never changes when the number of simulated days changes.

Index conventions (all 0-based, contiguous):
  P platforms, L pincodes, K SKUs, D dark stores (global across platforms),
  Q queries (= subcategories), B brands,
  "pair"   = serviceable (platform, pincode) combination,
  "cell"   = (pair, listed SKU)  -> unit of observation and of demand,
  "series" = (dark store, listed SKU) -> unit of inventory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import WorldConfig
from .seeds import WORLD_DAY, rng_for


@dataclass
class World:
    cfg: WorldConfig
    # sizes
    P: int
    L: int
    K: int
    D: int
    Q: int
    B: int
    subcats: list[str]
    # dimension frames
    dim_platform: pd.DataFrame
    dim_brand: pd.DataFrame
    dim_sku: pd.DataFrame
    dim_location: pd.DataFrame
    dim_dark_store: pd.DataFrame
    dim_query: pd.DataFrame
    # sku arrays [K]
    sku_sub: np.ndarray
    sku_brand: np.ndarray
    sku_pop: np.ndarray
    sku_mrp: np.ndarray
    sku_pack_kg: np.ndarray
    sku_case: np.ndarray
    sku_shelf: np.ndarray
    sku_is_focal: np.ndarray
    sku_promo_mult: np.ndarray
    # platform x sku [P,K]
    listed: np.ndarray
    ref_price: np.ndarray
    floor_price: np.ndarray
    # location / store arrays
    pin_zone: np.ndarray
    pin_dem: np.ndarray
    store_plat: np.ndarray
    store_zone: np.ndarray
    store_lead: np.ndarray
    # pairs
    pair_store: np.ndarray   # [P,L] store index or -1 (unserviced)
    pair_of: np.ndarray      # [P,L] pair index or -1
    pair_plat: np.ndarray    # [NP]
    pair_pin: np.ndarray     # [NP]
    # cells
    cell_pair: np.ndarray
    cell_plat: np.ndarray
    cell_pin: np.ndarray
    cell_sku: np.ndarray
    cell_store: np.ndarray
    cell_series: np.ndarray
    cell_of: np.ndarray      # [P,L,K] cell index or -1
    # series
    series_store: np.ndarray
    series_sku: np.ndarray
    series_plat: np.ndarray
    series_of: np.ndarray    # [D,K] series index or -1
    # candidates for search groups [P,Q,Cmax] padded with -1
    cand_sku: np.ndarray

    @property
    def n_cells(self) -> int:
        return len(self.cell_pair)

    @property
    def n_series(self) -> int:
        return len(self.series_store)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_plat)


def _largest_remainder(total: int, weights: np.ndarray, minimum: int = 0) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    raw = total * w / w.sum()
    out = np.maximum(np.floor(raw).astype(int), minimum)
    rem = total - out.sum()
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while rem > 0:
        out[order[i % len(out)]] += 1
        rem -= 1
        i += 1
    while rem < 0:  # only possible when `minimum` forced an overshoot
        j = int(np.argmax(out))
        out[j] -= 1
        rem += 1
    return out


def _by_rank(values: list[float], i: int) -> float:
    return values[min(i, len(values) - 1)]


def build_world(cfg: WorldConfig) -> World:
    seed = cfg.seed
    sc = cfg.scale
    subcats = list(cfg.subcats)
    Q, B, P, L = len(subcats), len(cfg.brands), len(cfg.platforms), sc.n_pincodes

    # ------------------------------------------------------------ brands / platforms
    dim_brand = pd.DataFrame({
        "brand_id": np.arange(B), "name": [b.name for b in cfg.brands],
        "archetype": [b.archetype for b in cfg.brands], "is_focal": [b.is_focal for b in cfg.brands]})
    dim_platform = pd.DataFrame({
        "platform_id": np.arange(1, P + 1), "code": [p.id for p in cfg.platforms],
        "name": [p.name for p in cfg.platforms], "demand_idx": [p.demand_idx for p in cfg.platforms],
        "price_aggressiveness": [p.price_aggressiveness for p in cfg.platforms],
        "ad_intensity": [p.ad_intensity for p in cfg.platforms],
        "restock_reliability": [p.restock_reliability for p in cfg.platforms]})
    # NOTE: platform_id in facts is 1..P; arrays inside the engine use 0..P-1 (id - 1).

    # ------------------------------------------------------------ SKU master
    rng = rng_for(seed, "sku", WORLD_DAY, "catalogue")
    mix = np.array([cfg.subcats[s].mix for s in subcats], dtype=float)
    counts = _largest_remainder(sc.n_skus, mix)
    rows: list[dict] = []
    focal_idx = next(i for i, b in enumerate(cfg.brands) if b.is_focal)
    for qi, sname in enumerate(subcats):
        sc_cfg = cfg.subcats[sname]
        combos = [(bi, vi, pi) for bi in range(B) for vi in range(len(sc_cfg.variants))
                  for pi in range(len(sc_cfg.packs))]
        focal_combos = [c for c in combos if c[0] == focal_idx]
        other = [c for c in combos if c[0] != focal_idx]
        n_focal = min(sc.focal_skus_per_subcat.get(sname, 0), len(focal_combos), counts[qi])
        chosen = [focal_combos[i] for i in rng.choice(len(focal_combos), size=n_focal, replace=False)]
        n_rest = int(counts[qi]) - n_focal
        if n_rest > len(other):
            raise ValueError(f"subcat {sname}: need {n_rest} SKUs but only {len(other)} combos exist")
        w = np.array([cfg.brands[b].assort_weight * _by_rank(cfg.pack_pop_mult, p) for b, _, p in other])
        chosen += [other[i] for i in rng.choice(len(other), size=n_rest, replace=False, p=w / w.sum())]
        for bi, vi, pi in sorted(chosen):
            br = cfg.brands[bi]
            pack = sc_cfg.packs[pi]
            mrp = sc_cfg.base_price_per_kg * (pack / 1000.0) * br.price_index \
                * _by_rank(cfg.pack_price_mult, pi) * float(rng.lognormal(0.0, 0.04))
            pop = float(rng.lognormal(0.0, cfg.demand.pop_sigma)) * _by_rank(cfg.pack_pop_mult, pi) \
                * _by_rank(cfg.variant_pop_mult, vi) * br.pop_mult * sc_cfg.volume_mult
            ucp = sc_cfg.unit_cost_pct * (0.92 if br.archetype == "private_label" else 1.0)
            rows.append(dict(
                name=f"{br.name} {sc_cfg.variants[vi]} {sname} {pack}{sc_cfg.unit}", brand_id=bi,
                subcategory=sname, variant=sc_cfg.variants[vi], pack_size=pack, unit=sc_cfg.unit,
                mrp=max(10, int(round(mrp))), unit_cost_pct=round(ucp, 3),
                shelf_life_days=sc_cfg.shelf_life_days, popularity_score=pop, _qi=qi, _case=sc_cfg.case_size))
    dim_sku = pd.DataFrame(rows)
    dim_sku.insert(0, "sku_id", np.arange(len(dim_sku)))
    dim_sku["popularity_score"] = dim_sku["popularity_score"] / dim_sku["popularity_score"].mean()
    K = len(dim_sku)
    n_hero = max(1, int(np.ceil(sc.hero_share * K)))
    hero_ids = set(dim_sku.sort_values("popularity_score", ascending=False).head(n_hero)["sku_id"])
    dim_sku["is_hero"] = dim_sku["sku_id"].isin(hero_ids)
    sku_sub = dim_sku["_qi"].to_numpy(np.int64)
    sku_brand = dim_sku["brand_id"].to_numpy(np.int64)
    sku_pop = dim_sku["popularity_score"].to_numpy(float)
    sku_mrp = dim_sku["mrp"].to_numpy(np.int64)
    sku_pack_kg = dim_sku["pack_size"].to_numpy(float) / 1000.0
    sku_case = dim_sku["_case"].to_numpy(np.int64)
    sku_shelf = dim_sku["shelf_life_days"].to_numpy(np.int64)
    sku_is_focal = sku_brand == focal_idx
    sku_promo_mult = np.array([cfg.brands[b].promo_mult for b in sku_brand])

    # ------------------------------------------------------------ platform assortment & prices
    rng = rng_for(seed, "assort", WORLD_DAY, "listing")
    hero = dim_sku["is_hero"].to_numpy()
    p_other = np.clip((sc.platform_listing_share - hero.mean()) / max(1e-9, 1 - hero.mean()), 0, 1)
    listed = np.zeros((P, K), dtype=bool)
    for p in range(P):
        listed[p] = hero | (rng.random(K) < p_other)
        for q in range(Q):  # every platform lists at least min(8, available) SKUs per subcat
            idx = np.where(sku_sub == q)[0]
            need = min(8, len(idx)) - listed[p, idx].sum()
            if need > 0:
                spare = idx[~listed[p, idx]]
                spare = spare[np.argsort(-sku_pop[spare])][:need]
                listed[p, spare] = True
    pr = cfg.pricing
    disc = np.zeros((P, K))
    for p in range(P):
        disc[p] = (pr.base_structural_discount * cfg.platforms[p].price_aggressiveness
                   * np.array([cfg.brands[b].discount_mult for b in sku_brand])
                   * rng.lognormal(0.0, pr.structural_discount_sigma, K))
    disc = np.clip(disc, 0.0, 0.20)
    floor_price = np.repeat(np.ceil(sku_mrp[None, :] * (1 - pr.max_discount)), P, axis=0)  # [P,K]
    ref_price = np.clip(np.round(sku_mrp[None, :] * (1 - disc)), floor_price, sku_mrp[None, :])
    ref_price = np.where(listed, ref_price, np.nan)

    # ------------------------------------------------------------ locations
    rng = rng_for(seed, "loc", WORLD_DAY, "layout")
    Z = sc.n_zones
    sizes = sc.zone_sizes or list(_largest_remainder(L, np.ones(Z)))
    pin_zone = np.repeat(np.arange(Z), sizes)
    aff = [sc.zone_affluence[z % len(sc.zone_affluence)] for z in range(Z)]
    dem = np.array([cfg.demand.affluence_mult[aff[z]] for z in pin_zone]) \
        * rng.lognormal(0.0, cfg.demand.demand_index_sigma, L)
    pin_dem = dem / dem.mean()
    labels = [sc.zone_labels[z % len(sc.zone_labels)] if sc.zone_labels else f"Z{z+1}" for z in range(Z)]

    # ------------------------------------------------------------ dark stores
    n_st = sc.stores_per_platform
    stores_per_zone = _largest_remainder(n_st, np.array(sizes, float), minimum=1)
    store_plat, store_zone, store_lead, cap = [], [], [], []
    for p in range(P):
        for z in range(Z):
            for _ in range(stores_per_zone[z]):
                store_plat.append(p)
                store_zone.append(z)
                store_lead.append(2 if rng.random() < cfg.inventory.store_lead_time_p2 else 1)
                cap.append(float(rng.lognormal(0.0, 0.2)))
    store_plat, store_zone = np.array(store_plat), np.array(store_zone)
    store_lead = np.array(store_lead)
    D = len(store_plat)
    dim_dark_store = pd.DataFrame({
        "dark_store_id": np.arange(D),
        "name": [f"{cfg.platforms[p].id}-DS{(i % n_st) + 1:02d}" for i, p in enumerate(store_plat)],
        "platform_id": store_plat + 1, "zone_id": store_zone, "capacity_index": cap,
        "supplier_lead_time_days": store_lead})

    # ------------------------------------------------------------ serviceability and pair->store
    pair_store = -np.ones((P, L), dtype=np.int64)
    for p in range(P):
        perm = rng.permutation(L)
        for z in range(Z):
            pins = [i for i in perm if pin_zone[i] == z]
            zs = np.where((store_plat == p) & (store_zone == z))[0]
            for j, pin in enumerate(pins):
                pair_store[p, pin] = zs[j % len(zs)]
    drop = rng.random((P, L)) < sc.unserviced_share
    for pin in range(L):  # keep at least one platform per pincode
        if drop[:, pin].all():
            drop[rng.integers(P), pin] = False
    pair_store = np.where(drop, -1, pair_store)
    pair_plat, pair_pin = np.where(pair_store >= 0)
    pair_of = -np.ones((P, L), dtype=np.int64)
    pair_of[pair_plat, pair_pin] = np.arange(len(pair_plat))

    dim_location = pd.DataFrame({
        "pincode_id": np.arange(L), "pincode": [f"NCR-{i+1:03d}" for i in range(L)],
        "zone_id": pin_zone, "zone_label": [labels[z] for z in pin_zone],
        "demand_index": pin_dem, "affluence_tier": [aff[z] for z in pin_zone]})
    for p in range(P):
        col = pair_store[p]
        dim_location[f"dark_store_id_p{p+1}"] = pd.array(np.where(col >= 0, col, -1), dtype="Int16")
        dim_location.loc[col < 0, f"dark_store_id_p{p+1}"] = pd.NA
    dim_query = pd.DataFrame({"query_id": np.arange(Q), "query_text": subcats, "subcategory": subcats})

    # ------------------------------------------------------------ series and cells
    s_store, s_sku = [], []
    series_of = -np.ones((D, K), dtype=np.int64)
    for d in range(D):
        ks = np.where(listed[store_plat[d]])[0]
        series_of[d, ks] = len(s_store) + np.arange(len(ks))
        s_store += [d] * len(ks)
        s_sku += list(ks)
    series_store, series_sku = np.array(s_store), np.array(s_sku)

    c_pair, c_sku = [], []
    for i, (p, l) in enumerate(zip(pair_plat, pair_pin)):
        ks = np.where(listed[p])[0]
        c_pair += [i] * len(ks)
        c_sku += list(ks)
    cell_pair, cell_sku = np.array(c_pair), np.array(c_sku)
    cell_plat, cell_pin = pair_plat[cell_pair], pair_pin[cell_pair]
    cell_store = pair_store[cell_plat, cell_pin]
    cell_series = series_of[cell_store, cell_sku]
    assert (cell_series >= 0).all()
    cell_of = -np.ones((P, L, K), dtype=np.int64)
    cell_of[cell_plat, cell_pin, cell_sku] = np.arange(len(cell_pair))

    cmax = max(int(((sku_sub == q) & listed[p]).sum()) for p in range(P) for q in range(Q))
    cand = -np.ones((P, Q, cmax), dtype=np.int64)
    for p in range(P):
        for q in range(Q):
            ks = np.where((sku_sub == q) & listed[p])[0]
            cand[p, q, : len(ks)] = ks

    sku_cols = ["sku_id", "name", "brand_id", "subcategory", "variant", "pack_size", "unit", "mrp",
                "unit_cost_pct", "shelf_life_days", "is_hero", "popularity_score"]
    return World(
        cfg=cfg, P=P, L=L, K=K, D=D, Q=Q, B=B, subcats=subcats,
        dim_platform=dim_platform, dim_brand=dim_brand, dim_sku=dim_sku[sku_cols].copy(),
        dim_location=dim_location, dim_dark_store=dim_dark_store, dim_query=dim_query,
        sku_sub=sku_sub, sku_brand=sku_brand, sku_pop=sku_pop, sku_mrp=sku_mrp, sku_pack_kg=sku_pack_kg,
        sku_case=sku_case, sku_shelf=sku_shelf, sku_is_focal=sku_is_focal, sku_promo_mult=sku_promo_mult,
        listed=listed, ref_price=ref_price, floor_price=floor_price,
        pin_zone=pin_zone, pin_dem=pin_dem, store_plat=store_plat, store_zone=store_zone,
        store_lead=store_lead, pair_store=pair_store, pair_of=pair_of, pair_plat=pair_plat,
        pair_pin=pair_pin, cell_pair=cell_pair, cell_plat=cell_plat, cell_pin=cell_pin, cell_sku=cell_sku,
        cell_store=cell_store, cell_series=cell_series, cell_of=cell_of, series_store=series_store,
        series_sku=series_sku, series_plat=store_plat[series_store], series_of=series_of, cand_sku=cand)
