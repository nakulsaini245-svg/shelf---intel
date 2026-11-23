# DATA CARD: Synthetic Quick-Commerce Dairy World (v1, with planted incidents)

**Status:** Week 2 / Step 2A. Config digest `3f1b674dec220724` (`configs/world_v1.yaml`).

## 1. Purpose and scope
A fully synthetic, reproducible world for building and evaluating a shelf-intelligence system: stockout and
availability dynamics, price/promo shifts, and sponsored-visibility changes for dairy (paneer, cheese, curd,
chaach, milk) across 3 platforms, 75 pincodes in 8 zones, 135 SKUs and 90 days, with **planted incidents and a
counterfactual answer key**.

**Everything is synthetic.** No real company, platform, price, sales or user data is used.
- Brands are fictional (Arvana is the "focal" brand; the others are competitors by archetype). Any resemblance to real firms is coincidental.
- Platforms appear as P1/P2/P3 (the labels "Blinkit-like" etc. describe archetypes only).
- Pincodes are synthetic ids (`NCR-001`...). Zone names (South Delhi, Gurugram...) are illustrative labels, not geography.
- Festival dates are approximate; their effect sizes are invented.

**Results measured on this data are simulator-conditional.** They show whether a method recovers *planted* structure, not how it
would perform on real marketplace data. State this in any report that uses it.

## 2. What is observed, public, and latent
| Group | Tables | Who may use it |
|---|---|---|
| Observed | `fact_listing_daily`, `fact_search_visibility`, `fact_focal_sellout`, `scrape_log` | models and the agent |
| Public dims | `dim_platform/brand/sku/location/dark_store/query`, `dim_calendar` | models and the agent |
| Latent truth | `latent_demand`, `latent_stock`, `latent_obs_flags`, `latent_events`, `latent_effects` | **evaluation only** |

Key assumptions about observation:
- **`fact_focal_sellout` is an assumed first-party feed for the focal brand only** (like a brand-partner dashboard). Competitor sales are never observed. Without it, revenue-at-risk has no grounding.
- Availability is captured at 4 snapshots/day (06, 12, 18, 22h); search results at 2 (12h, 18h). A "day" runs 22:00 to 22:00.
- `NA` availability means the scrape failed and is never filled. Failure mechanisms are mixed on purpose: random timeouts, **evening-skewed (MNAR)** failures scaled by platform flakiness, transient false "unserviceable" answers, and **silently stale** snapshots (logged as `ok`; truth only in `latent_obs_flags`).
- ~9.8% of platform x pincode pairs have no coverage; they appear as `unserviceable` in `scrape_log`.
- `dim_calendar` is a **public** festival calendar (baseline plus regional festivals). A seasonality-aware detector is expected to use it.

## 3. Generative process (summary)
- **Demand** per platform x pincode x SKU x day: popularity x pincode index x platform index x day-of-week x seasonality x festival x price effect x cross-price effect x promo display lift x ad share-of-voice effect, drawn from a negative binomial (overdispersion r=8) across 4 intraday periods.
- **Inventory** per dark store x SKU: forecast-driven order-up-to policy (forecast is an EWMA of *sales*, so stockouts depress it), case-pack rounding, 1-2 day supplier lead times, stochastic fill and delay, overnight arrival, emergency mid-day top-ups, shelf-life write-offs.
- **Pricing:** MRP, structural platform discount, promo spells, and a lagged cross-platform price-match reaction with a floor.
- **Ads and search:** each snapshot runs a Gumbel-perturbed auction: 3 sponsored positions (1, 4, 7) by bid x relevance, the remaining slots by organic score (popularity, availability history, price per kg). Stockouts degrade later organic rank.
- **Cross-effects:** demand responds to own price relative to a popularity-weighted competitor price index (cross-elasticity = own elasticity), and to the brand's share-of-voice relative to its warm-up reference.

## 4. Calibration choices (all are assumptions)
Marginals at full scale (90 days, with events) against target bands in the config:

| Metric | Value | Target band |
|---|---|---|
| Stockout rate (snapshot) | 0.063 | 0.04-0.14 |
| Mean discount % | 6.05 | 4-14 |
| Promo share | 0.101 | 0.06-0.30 |
| Scrape failure rate | 0.042 | 0.02-0.10 |
| Top-3 brand share of sponsored slots | 0.74 | 0.45-0.90 |
| Focal fill rate | 0.88 | 0.85-0.99 |
| Mean units / cell / day | 1.41 | 0.6-3.0 |

Bands are plausibility ranges I chose, not empirical benchmarks. Notable tuning:
- Replenishment cover was cut from 1.6 to **0.2 days** because 1.6 gave only 0.5% stockouts. The result is a deliberately lean network.
- Cross-price elasticity ratio **1.0** and SoV elasticity **1.0** (raised from 0.5 / 0.25) so competitor undercuts and ad takeovers produce effects large enough to be learnable.
- Demand and write-off noise use **common random numbers** (see section 6).

## 5. Planted incidents (E1-E10) and the answer key
29 incidents (35 rows including E6 children) over days 10-84: a clean 10-day baseline prefix, then spread across platforms and zones, with no two events sharing a platform *and* zone within `tail_days + min_gap_days` (7 days). Difficulty tiers (easy / medium / hard) interpolate every knob between an `[easy, hard]` pair in the config.

| ID | n | Mechanism | Hard negative |
|---|---|---|---|
| E1 | 3 | Zone-level supply shock: a share of one platform's dark stores in a zone get a fraction of inbound stock, with per-store severity jitter | no |
| E2 | 4 | Single dark-store outage (stock loss plus inbound cut) | no |
| E3 | 4 | Competitor brands undercut one subcategory in 1-2 zones, ramping in over a 2-day lag | no |
| E4 | 3 | Own promo ends: promo window, then a post-promo demand dip | **yes** (explained by the observed `promo_flag`) |
| E5 | 3 | Competitor ad takeover: bid x2.5-6 and higher participation in 1-3 zones | no |
| E6 | 3 | Compound: the *same* competitor undercuts (E6a) and surges ads (E6b) in one zone | no (parent alerts) |
| E7 | 2 | Phantom stockout: stock exists, SKUs shown unavailable; some shown "unlisted", the rest identical to a stockout | no (ambiguity test) |
| E8 | 3 | Scrape blackout: scraper sees nothing; **no change to the real world** | **yes** |
| E9 | 2 | Slow competitor price drift (6-12% over 14-24 days), then snap-back | no |
| E10 | 2 | Regional festival demand surge, announced in `dim_calendar`, with platform pre-stocking | **yes** |

Classification note: you named E8 and E10 as the hard negatives. I also marked **E4** as one, because a good system should explain its demand cliff by the observed promo flag rather than alert. It is one flag (`is_hard_negative` / `should_alert`) to flip if you disagree.

**Answer key (`latent_effects`).** `true_delta = event world - counterfactual baseline world`, summed over the event's platform x pincode scope for `[first day, end + 5]`. Baseline and event worlds share all random numbers, so the difference isolates the event's causal effect plus its knock-on dynamics (stockouts outlast their cause). Metrics: focal sold units and GMV; all-brand sold, true demand and lost units; out-of-stock snapshot rate; shelf-price change; SoV change (all slots and sponsored-only); scrape failure rate (E8); E4 promo/cliff phases. **E6 attribution** uses two extra counterfactual worlds (undercut only, SoV surge only): `attr_a_*`, `attr_b_*`, and `attr_interaction_*` (total minus the parts).

Observed effect sizes (focal units, per event): E1 -237 to -312; E2 -49 to -309; E3 -5 to -34; E5 -53 to -124; E6 -42 to -56 (undercut part about -5 to -10, SoV part about -41 to -55); E7 -84 to -109; E9 -26 to -78. E3 and E6a are small by design at hard tiers; do not expect every "should alert" event to be detectable.

## 6. Determinism and reproducibility
- Every random stream is keyed by `(seed, scope, day, stream)` and hashed with BLAKE2b; same seed and config give byte-identical tables.
- **Append-only:** generating N+1 days reproduces days 0..N-1 exactly. The incident plan depends only on the seed and config (`events.horizon_days`), never on how many days are simulated. Extending beyond the horizon adds baseline-only days.
- **Common random numbers:** negative-binomial demand uses one uniform per draw (inversion), and write-off uses stochastic rounding. numpy's Poisson/Binomial samplers consume a parameter-dependent number of random bits, which silently decorrelated worlds and made counterfactuals noisy. This was found and fixed in Step 2A; **Week 1 datasets generated before it are not reproducible from the new code** (regenerate them).
- Tests cover determinism, prefix invariance (with and without events), schema and invariants, the hook seam, and the answer key.

## 7. Known limits (read before reporting results)
1. **Simulator-conditional.** Detection and attribution quality here is an upper bound on real-world performance.
2. **Interference residual in the answer key.** Effects are measured against a world with *no* events, so carry-over from an earlier event in the same platform x zone (forecast and stock dynamics decay slowly) leaks in. E8, whose true effect is exactly zero when run alone, shows residuals of about -12 to +14 focal units in the full plan. Treat differences smaller than roughly 15 units per event as within the answer key's own noise.
3. **E10 and E4 net effects can be small or negative for the focal brand** even though all-brand demand rises (E10 all-brand demand: +467 to +2,937 units). Use the phase metrics and all-brand metrics when labelling.
4. **Ad takeovers are modest.** Sponsored positions are 3 of 10 slots, and SoV is measured across all queries, so all-slot share shifts by only about 1-3 points. Use the `*_sponsored_sov_pp` metrics for the direct ad effect. In small catalogues, all-slot share saturates when a query has no more than 10 candidates.
5. **Availability is store-level**, so pincodes served by one dark store are perfectly correlated.
6. **Simplifications:** one first-party feed (focal brand), no competitor sales, no cross-platform effects of planted price changes, a simple auction, no user-level behaviour, and festival effects on subcategories only.
7. **Time resolution** is 4 availability snapshots per day, so sub-6-hour dynamics are not represented.

## 8. Reproduce and verify
```
python -m pytest                       # fast suite (34 tests)
python -m pytest -m "slow or not slow" # adds full-scale event run (36 tests)
python -m sim.cli check                # 90-day run, all invariants and calibration targets
python -m sim.cli generate --out data/world_v1
```
Full 90-day generation with the event world plus 3 counterfactual worlds takes about 14 s locally. Without `pyarrow` the writer falls back to gzipped CSV.
