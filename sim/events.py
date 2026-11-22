"""Event-injection hook: the seam between the baseline world and planted incidents (E1..E10).

The engine asks the hook for a `DayModifiers` once per simulated day (days >= 0 only; warm-up days
are never perturbed). Every field is optional; None means "no change". The hook also owns the
*latent* answer key (what it planted and its counterfactual effect), emitted as latent_events /
latent_effects and never exposed to models.

Array shapes (n_cells / n_series from the World):
    demand_mult        [n_cells]        multiplies true demand in a cell; applied AFTER the organic
                                        clip, so 0 really means zero demand
    price_mult         [n_cells]        multiplies the shelf price (floor and MRP enforced afterwards)
    supply_fill_mult   [n_series]       multiplies that day's INCOMING replenishment (shortfall is lost;
                                        values > 1 model pre-stocking)
    stock_destroy_frac [n_series]       fraction of on-hand stock lost at the start of the day (outage)
    ad_bid_mult        [P, L, B]        multiplies a brand's sponsored bid per platform x pincode
    listing_suppress   [n_cells] bool   cell is DELISTED today (status "unlisted", no demand, stock kept)
    hide_available     [n_cells] bool   cell is shown "unavailable" but stays status "listed" (no demand,
                                        stock kept): observationally identical to a stockout
    promo_flag         [n_cells] bool   cell carries an own-promo flag today (display lift + observed flag)
    scrape_blackout    [P, L] bool      scraper gets nothing for that platform x pincode all day
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from .schema import LATENT, coerce

if TYPE_CHECKING:  # pragma: no cover
    from .world import World


@dataclass
class DayModifiers:
    demand_mult: np.ndarray | None = None
    price_mult: np.ndarray | None = None
    supply_fill_mult: np.ndarray | None = None
    stock_destroy_frac: np.ndarray | None = None
    ad_bid_mult: np.ndarray | None = None
    listing_suppress: np.ndarray | None = None
    hide_available: np.ndarray | None = None
    promo_flag: np.ndarray | None = None
    scrape_blackout: np.ndarray | None = None


class EventHook(Protocol):
    def modifiers(self, day: int, day_date: date, world: "World") -> DayModifiers: ...
    def latent_events(self) -> pd.DataFrame: ...
    def latent_effects(self) -> pd.DataFrame: ...
    def calendar_rows(self) -> pd.DataFrame: ...


class NullHook:
    """No events: the baseline world."""

    def modifiers(self, day: int, day_date: date, world: "World") -> DayModifiers:
        return DayModifiers()

    def latent_events(self) -> pd.DataFrame:
        return coerce(_empty(LATENT["latent_events"]), "latent_events")

    def latent_effects(self) -> pd.DataFrame:
        return coerce(_empty(LATENT["latent_effects"]), "latent_effects")

    def calendar_rows(self) -> pd.DataFrame:
        """Public regional-festival rows (date, festival_name, scope_zone_id). Empty in the baseline."""
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "festival_name": pd.Series(dtype="string"),
                             "scope_zone_id": pd.Series(dtype="Int8")})


def _empty(spec: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in spec})
