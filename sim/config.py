"""Typed, validated configuration for the synthetic world."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml


def _make(cls: type, d: Mapping[str, Any], where: str):
    """Instantiate a dataclass from a mapping, rejecting unknown/missing keys."""
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")
    try:
        return cls(**d)
    except TypeError as e:  # missing required keys
        raise ValueError(f"bad {where}: {e}") from e


@dataclass(frozen=True)
class ScaleCfg:
    n_pincodes: int
    n_zones: int
    zone_sizes: list[int] | None
    zone_affluence: list[str]
    zone_labels: list[str]
    stores_per_platform: int
    n_skus: int
    platform_listing_share: float
    hero_share: float
    unserviced_share: float
    focal_skus_per_subcat: dict[str, int]


@dataclass(frozen=True)
class SubcatCfg:
    base_price_per_kg: float
    packs: list[int]
    variants: list[str]
    unit: str
    shelf_life_days: int
    case_size: int
    elasticity: float
    seasonal_amp: float
    seasonal_peak_doy: int
    unit_cost_pct: float
    mix: float
    volume_mult: float


@dataclass(frozen=True)
class BrandCfg:
    name: str
    archetype: str
    is_focal: bool
    ad_strength: float
    ad_participation: float
    price_index: float
    discount_mult: float
    promo_mult: float
    assort_weight: float
    pop_mult: float


@dataclass(frozen=True)
class PlatformCfg:
    id: str
    name: str
    demand_idx: float
    price_aggressiveness: float
    ad_intensity: float
    restock_reliability: float
    flakiness: float


@dataclass(frozen=True)
class FestivalCfg:
    name: str
    date: str
    window_days: int
    mult: dict[str, float]


@dataclass(frozen=True)
class DemandCfg:
    base_units_per_cell_day: float
    nb_dispersion: float
    pop_sigma: float
    demand_index_sigma: float
    affluence_mult: dict[str, float]
    dow_mult: list[float]
    period_weights: list[float]
    promo_display_lift: float
    sov_eta: float
    cross_elasticity_ratio: float
    demand_mult_clip: list[float]
    baseline_unlist_prob: float
    festivals: list[FestivalCfg]


@dataclass(frozen=True)
class InventoryCfg:
    warmup_days: int
    ewma_alpha: float
    cover_days: float
    min_presentation_units: int
    fill_sd: float
    delay_base: float
    topup_prob: float
    writeoff_scale: float
    initial_cover_days: float
    store_lead_time_p2: float


@dataclass(frozen=True)
class PricingCfg:
    base_structural_discount: float
    structural_discount_sigma: float
    promo_start_prob: float
    promo_len_range: list[int]
    promo_depth_range: list[float]
    match_tolerance: float
    match_prob: float
    max_discount: float


@dataclass(frozen=True)
class AdsCfg:
    n_slots: int
    sponsored_positions: list[int]
    snapshots: list[int]
    gumbel_sponsored: float
    gumbel_organic: float
    w_conv: float
    w_price: float
    oos_penalty: float
    pacing_sigma: float
    conv_ema_alpha: float


@dataclass(frozen=True)
class ObservationCfg:
    snapshot_hours: list[int]
    p_timeout: float
    evening_extra: float
    p_serviceability_error: float
    p_stale: float


@dataclass(frozen=True)
class WorldConfig:
    version: int
    seed: int
    start_date: str
    n_days: int
    scale: ScaleCfg
    subcats: dict[str, SubcatCfg]
    pack_pop_mult: list[float]
    pack_price_mult: list[float]
    variant_pop_mult: list[float]
    brands: list[BrandCfg]
    platforms: list[PlatformCfg]
    demand: DemandCfg
    inventory: InventoryCfg
    pricing: PricingCfg
    ads: AdsCfg
    observation: ObservationCfg
    targets: dict[str, list[float]] = field(default_factory=dict)

    # ------------------------------------------------------------ helpers
    @property
    def start(self) -> date:
        return date.fromisoformat(self.start_date)

    def digest(self) -> str:
        """Stable short hash of the full config (goes in the dataset manifest)."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def with_overrides(self, **top: Any) -> "WorldConfig":
        return replace(self, **top)

    def validate(self) -> None:
        s = self.scale
        if s.zone_sizes is not None:
            if len(s.zone_sizes) != s.n_zones or sum(s.zone_sizes) != s.n_pincodes:
                raise ValueError("scale.zone_sizes must have n_zones entries summing to n_pincodes")
        if abs(sum(self.demand.period_weights) - 1.0) > 1e-9 or len(self.demand.period_weights) != 4:
            raise ValueError("demand.period_weights must be 4 values summing to 1")
        if len(self.observation.snapshot_hours) != 4:
            raise ValueError("observation.snapshot_hours must have 4 entries")
        if sum(b.is_focal for b in self.brands) != 1:
            raise ValueError("exactly one brand must be focal")
        if any(sn not in (1, 2, 3, 4) for sn in self.ads.snapshots):
            raise ValueError("ads.snapshots must be in 1..4")
        if max(self.ads.sponsored_positions) > self.ads.n_slots:
            raise ValueError("sponsored_positions exceed n_slots")
        for k in self.scale.focal_skus_per_subcat:
            if k not in self.subcats:
                raise ValueError(f"focal_skus_per_subcat references unknown subcat {k}")
        if self.inventory.warmup_days < 1:
            raise ValueError("inventory.warmup_days must be >= 1")


def load_config(path: str | Path) -> WorldConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return config_from_dict(raw)


def config_from_dict(raw: Mapping[str, Any]) -> WorldConfig:
    raw = dict(raw)
    demand = dict(raw.pop("demand"))
    demand["festivals"] = [_make(FestivalCfg, f, "demand.festivals[]") for f in demand["festivals"]]
    cfg = WorldConfig(
        version=raw.pop("version"),
        seed=raw.pop("seed"),
        start_date=str(raw.pop("start_date")),
        n_days=raw.pop("n_days"),
        scale=_make(ScaleCfg, raw.pop("scale"), "scale"),
        subcats={k: _make(SubcatCfg, v, f"subcats.{k}") for k, v in raw.pop("subcats").items()},
        pack_pop_mult=raw.pop("pack_pop_mult"),
        pack_price_mult=raw.pop("pack_price_mult"),
        variant_pop_mult=raw.pop("variant_pop_mult"),
        brands=[_make(BrandCfg, b, "brands[]") for b in raw.pop("brands")],
        platforms=[_make(PlatformCfg, p, "platforms[]") for p in raw.pop("platforms")],
        demand=_make(DemandCfg, demand, "demand"),
        inventory=_make(InventoryCfg, raw.pop("inventory"), "inventory"),
        pricing=_make(PricingCfg, raw.pop("pricing"), "pricing"),
        ads=_make(AdsCfg, raw.pop("ads"), "ads"),
        observation=_make(ObservationCfg, raw.pop("observation"), "observation"),
        targets=raw.pop("targets", {}),
    )
    if raw:
        raise ValueError(f"unknown top-level key(s): {sorted(raw)}")
    cfg.validate()
    return cfg
