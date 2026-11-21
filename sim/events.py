"""Event-injection hook: the seam where Week 2 plants E1..E10.

The engine asks the hook for a `DayModifiers` once per simulated day (days >= 0
only; warm-up days are never perturbed). Every field is optional; None means
"no change". The hook also owns the *latent* answer key (what it planted), which
is emitted as latent_events / latent_effects and never exposed to models.

Array shapes (n = world sizes):
    demand_mult        [n_cells]        multiplies true demand in a cell
    price_mult         [n_cells]        multiplies the shelf price in a cell (<=MRP and >=floor enforced after)
    supply_fill_mult   [n_series]       multiplies that day's INCOMING replenishment (shortfall is lost)
    stock_destroy_frac [n_series]       fraction of on-hand stock lost at the start of the day (outage)
    ad_bid_mult        [P, B]           multiplies a brand's sponsored bid on a platform
    listing_suppress   [n_cells] bool   cell is delisted today although stock may exist (phantom stockout)
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
    scrape_blackout: np.ndarray | None = None


class EventHook(Protocol):
    def modifiers(self, day: int, day_date: date, world: "World") -> DayModifiers: ...
    def latent_events(self) -> pd.DataFrame: ...
    def latent_effects(self) -> pd.DataFrame: ...


class NullHook:
    """No events: the baseline world."""

    def modifiers(self, day: int, day_date: date, world: "World") -> DayModifiers:
        return DayModifiers()

    def latent_events(self) -> pd.DataFrame:
        return coerce(_empty(LATENT["latent_events"]), "latent_events")

    def latent_effects(self) -> pd.DataFrame:
        return coerce(_empty(LATENT["latent_effects"]), "latent_effects")


def _empty(spec: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in spec})
