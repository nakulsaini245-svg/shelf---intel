"""Table schemas: column -> dtype, plus which tables are observed vs latent.

Ids are compact surrogate integer keys (platform_id int8, pincode_id int16,
sku_id int16, ...). The dim_* tables carry the human-readable labels. This keeps
~2M-row fact tables small enough for free-tier containers.

Nullable availability columns use pandas "Int8": <NA> means the scrape failed
and is NEVER silently filled.
"""
from __future__ import annotations

# dtype names are pandas dtype strings
DIMS: dict[str, dict[str, str]] = {
    "dim_platform": {"platform_id": "int8", "code": "string", "name": "string", "demand_idx": "float64",
                     "price_aggressiveness": "float64", "ad_intensity": "float64",
                     "restock_reliability": "float64"},
    "dim_brand": {"brand_id": "int8", "name": "string", "archetype": "string", "is_focal": "bool"},
    "dim_sku": {"sku_id": "int16", "name": "string", "brand_id": "int8", "subcategory": "string",
                "variant": "string", "pack_size": "int32", "unit": "string", "mrp": "int32",
                "unit_cost_pct": "float64", "shelf_life_days": "int16", "is_hero": "bool",
                "popularity_score": "float64"},
    "dim_location": {"pincode_id": "int16", "pincode": "string", "zone_id": "int8", "zone_label": "string",
                     "dark_store_id_p1": "Int16", "dark_store_id_p2": "Int16", "dark_store_id_p3": "Int16",
                     "demand_index": "float64", "affluence_tier": "string"},
    "dim_dark_store": {"dark_store_id": "int16", "name": "string", "platform_id": "int8", "zone_id": "int8",
                       "capacity_index": "float64", "supplier_lead_time_days": "int8"},
    "dim_query": {"query_id": "int8", "query_text": "string", "subcategory": "string"},
}

OBSERVED: dict[str, dict[str, str]] = {
    "fact_listing_daily": {
        "day_index": "int16", "date": "datetime64[ns]", "platform_id": "int8", "pincode_id": "int16",
        "sku_id": "int16", "snap1_avail": "Int8", "snap2_avail": "Int8", "snap3_avail": "Int8",
        "snap4_avail": "Int8", "price": "Int32", "mrp": "int32", "discount_pct": "float32",
        "promo_flag": "Int8", "listing_status": "string", "scrape_ok": "bool"},
    "fact_search_visibility": {
        "day_index": "int16", "date": "datetime64[ns]", "snapshot": "int8", "platform_id": "int8",
        "pincode_id": "int16", "query_id": "int8", "position": "int8", "sku_id": "int16",
        "is_sponsored": "bool"},
    # ASSUMED first-party feed, focal brand only (see DATA_CARD.md).
    "fact_focal_sellout": {
        "day_index": "int16", "date": "datetime64[ns]", "platform_id": "int8", "pincode_id": "int16",
        "sku_id": "int16", "units": "int32", "gmv": "int64"},
    "scrape_log": {
        "day_index": "int16", "date": "datetime64[ns]", "platform_id": "int8", "pincode_id": "int16",
        "snapshot": "int8", "status": "string", "reason": "string"},
}

LATENT: dict[str, dict[str, str]] = {
    "latent_demand": {
        "day_index": "int16", "date": "datetime64[ns]", "platform_id": "int8", "pincode_id": "int16",
        "sku_id": "int16", "true_demand_units": "int32", "sold_units": "int32", "lost_units": "int32"},
    "latent_stock": {
        "day_index": "int16", "date": "datetime64[ns]", "snapshot": "int8", "dark_store_id": "int16",
        "sku_id": "int16", "on_hand_units": "int32"},
    "latent_obs_flags": {  # scrape corruptions the scraper cannot see
        "day_index": "int16", "date": "datetime64[ns]", "platform_id": "int8", "pincode_id": "int16",
        "snapshot": "int8", "was_stale": "bool"},
    "latent_events": {
        "event_id": "int32", "type": "string", "start_day": "int16", "end_day": "int16",
        "scope_platform": "string", "scope_pincodes": "string", "scope_skus": "string",
        "magnitude": "float64", "causal_parent_event_id": "Int32", "is_hard_negative": "bool"},
    "latent_effects": {"event_id": "int32", "metric": "string", "true_delta": "float64"},
}

DAY_INDEXED = [t for t, c in {**OBSERVED, **LATENT}.items() if "day_index" in c]
ALL_TABLES = {**DIMS, **OBSERVED, **LATENT}

# Column names that must never appear in an observed table (leak guard used by tests).
LATENT_COLUMNS = {"true_demand_units", "sold_units", "lost_units", "on_hand_units", "was_stale",
                  "is_hard_negative", "true_delta"}


def coerce(df, table: str):
    """Cast a DataFrame to the declared dtypes for `table` (column order preserved)."""
    spec = ALL_TABLES[table]
    missing = set(spec) - set(df.columns)
    if missing:
        raise ValueError(f"{table}: missing columns {sorted(missing)}")
    return df[list(spec)].astype(spec)
