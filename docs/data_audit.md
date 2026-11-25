# Data Source Audit — U.S. Home Sales (pre-modeling)

Date: 2026-11-25
Evaluator: fotroslab (independent portfolio case study)

We compared four candidate families on target type, geography, recency, columns, licence, join quality, and gaps before writing any model code.

## Summary recommendation

**Primary for MVP:** Allegheny County Property Assessments via WPRDC (CC0) + companion Real Estate Sales table.
- Only candidate with **row-level recorded sale price + building features + CC0 licence + single-table access + 10+ years of history**.
- Geography is honestly scoped to **one county** (Allegheny, PA — Pittsburgh metro), which matches the “750–1,500 USD freelance pilot” brief.
- Limitations (snapshot leakage, valid-sale coding) are documented and manageable with filters and temporal holdout.

Alternatives were kept as **benchmark/context only**: Zillow/Redfin for regional price context, Ames for methodological baseline, King County as runner-up.

## Comparison table

| Source | Target type | Geography & time | Usable rows (est.) | Key columns for our feature list | Licence / redistribution | Freshness & quirks | Join reliability | Gaps |
|---|---|---|---|---|---|---|---|---|
| **1. Allegheny — Property Assessments (WPRDC)** — https://data.wprdc.org/dataset/property-assessments | Recorded **sale price** (`SALEPRICE`) + sale code/date | Allegheny County, PA (130 munic., Pittsburgh). 2013–present; snapshot **2026-10-30** | **147,581 valid** (`SALECODE=0`) in full; 5,800 in curated sample after residential + completeness filters | `FINISHEDLIVINGAREA`, `LOTAREA`, `BEDROOMS`, `FULLBATHS`, `HALFBATHS`, `TOTALROOMS`, `YEARBLT`, `CONDITION`, `GRADE`, `EXTERIORFINISH`, `STORIES`, `BSMTGARAGE`, `FIREPLACES`, `PROPERTYZIP`, `MUNIDESC`, `NEIGHCODE`, `SCHOOLCODE`. No explicit garage count, but `BSMTGARAGE` → `has_basement_garage` | **CC0 1.0** — public domain. No attribution required (we attribute anyway). Allows sample redistribution | Updated ~weekly; assessment snapshot is periodic. `SALEDATE` null for never-sold parcels; validation codes subjective and revised. Commercial/parcels without building data have null living area | Single table: sale + features together via `PARID`. No cross-table join needed for MVP | Snapshot leakage: features are 2026 state applied to older sales. Assessed values must not be used as features (leakage). `BEDROOMS` coverage 75.8%, `FULLBATHS` 75.7% → needs imputation. No direct rental data |
| **1b. Allegheny — Real Estate Sales (WPRDC)** — https://data.wprdc.org/dataset/real-estate-sales | Recorded sale price (`PRICE`) + `SALECODE`/`SALEDESC` | Same county, 2013–present; last update **2026-08-03**; 2012 archives available | ~300k+ rows raw, ~140k+ valid after filtering | `PRICE`, `SALEDATE`, `RECORDDATE`, `PROPERTYZIP`, `MUNIDESC`, `SCHOOLDESC`, `PARID`, `INSTRTYP`. **No building features** | **CC0 1.0** | Deed recording delay 1–6 weeks; multi-parcel deed repeats total price on each parcel (“H” code) — must deduplicate | Join to assessments via `PARID` works but requires dedup logic and date alignment; adds complexity vs. using assessments alone | Needs second table for features; still same county scope. We used it only to verify code meanings |
| **2. King County, WA — Assessor eSales + Data Download** — https://info.kingcounty.gov/assessor/esales/Residential.aspx and https://kingcounty.gov/en/dept/assessor/buildings-and-property/property-value-and-information/reports/ | Recorded residential sale price (excise data) | King County, WA (Seattle/Bellevue). Weekly extracts; history back to ~2004 | ~25k sales/year; ~15k residential single-family after filtering | Residential building extract (`resbldg` — 1–3 units), sales extract (`rpsale`), parcel base — together give `yrBuilt`, `sqFtTotLiving`, `bedrooms`, `bathFull/Half`, `grade`, `condition`, `zip5`, `presentUse` | Public records, free; **no explicit open-data licence** stated on download page. Redistribution of derived sample is tolerated in practice but not formally CC0. Must check King County terms per use | Weekly refresh; sale affidavits; residential vs. condo vs. apartment classification is strict; “Major/Minor → ParcelID” join required | **Three-table join** (parcel → building → sale via Excise Tax Number). Well-documented (`RPACCT_EXTR` FAQ) but more ETL | Heavier pipeline for a one-county pilot; licence less explicit than Allegheny CC0; Seattle median price (~$850k) is very different market — would need separate model |
| **3. Zillow Research & Redfin Data Center** — https://www.zillow.com/research/data/ and https://www.redfin.com/news/data-center/ | **Aggregated indices only**: ZHVI (typical home value), ZORI (rent), inventory, days-on-market — **not a per-parcel price** | USA, state/metro/county/ZIP; ZIP coverage 26,274; monthly back to 2000 | Not applicable as target — no row-level sales. Useful for **context**: e.g., Allegheny ZHVI median ~$215k (2024) vs. our holdout median $308k (reflects valid-residential mix + appreciation) | `RegionName`, `RegionType`, `StateName`, date columns (YYYY-MM-DD) per metric | Zillow: “Free for public use, with attribution to Zillow” (general terms, no formal open licence but widely reused with credit). Redfin: similar attribution. **Not a parcel licence** | Monthly, 2–4 week lag; smoothed/seasonally adjusted; composition-controlled (not same as transaction median) | No parcel join — cannot train per-home model | **Cannot be target**. We cite them only in README/notebook for market context and to show our pilot is not “U.S. national” |
| **4. Ames, Iowa Housing (Kaggle)** — https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques | Recorded sale price (1460 train + 1459 test) | **Ames, IA only**, 2006–2010 | 2,919 total | `SalePrice`, `GrLivArea`, `LotArea`, `BedroomAbvGr`, `FullBath`, `HalfBath`, `YearBuilt`, `OverallCond/Qual`, `Exterior1st`, `Garage*`, `MSZoning`, `Neighborhood` | **Kaggle competition rules + unclear upstream licence** — not CC0; for education/benchmark only | Frozen 2010; tiny; no U.S. geographic variance | Single file, clean | **Not a pilot market**; we treat it as **benchmark/baseline only** to sanity-check pipeline (e.g., we verified our Ridge baseline beats median by ~60% RMSE, similar to Ames) |

## Licence verification notes (2026-11-25)

- WPRDC landing pages list **License: Creative Commons CCZero** in the CKAN “Additional Information” panel for both datasets (checked via API `license_title` and web UI). The portal’s Data Use Agreement grants access under owner-specified licence — here CC0 — so redistribution of a derived sample is permitted.
- King County’s download site states “public records” but does not publish a CC0/MIT-equivalent licence; we therefore did not select it for a redistributable sample.
- Zillow’s research CSVs have no machine-readable licence field (per HousingHandbook audit 2026-07-15) but are described as “free for public use with attribution” — suitable for citing numbers, not for parcel training.
- Ames/Kaggle: competition data — we did not re-host it.

## Gaps & risks that shaped the MVP scope

- **Snapshot leakage** (assessments): we use temporal holdout (most recent 20% = 2024–2026) and report that train (2019–2024) underpredicts holdout by ~$35k mean residual due to market appreciation + feature snapshot; production should retrain monthly.
- **Sale-code subjectivity**: only `SALECODE = 0` (“VALID SALE”) is kept; codes `H` (multi-parcel), `3` (love & affection), `GV` (government), `99` (corrective) etc. are excluded and documented.
- **Missingness**: ~24% of parcels lack bedrooms/baths (commercial/special-use) — we filter to residential with `FINISHEDLIVINGAREA` present and impute median/mode in pipeline (fit on train only).
- **Geographic narrowness**: one county is intentional; multi-county would require per-county licence checks, per-market drift monitoring, and separate evaluation — listed in Roadmap instead.
- **No rent/deposit/photo**: no jurisdiction-safe source with parcel-level rent or permitted photos was found that is both CC0 and parcel-joined; these remain Roadmap items.

## Decision

Proceed with **Allegheny County, PA** as the defined market for Phase 1, target = **recorded single-parcel arm’s-length residential sale price**, evaluation on **temporal holdout**, and demo limited to that county/timeframe with honest limits. If stakeholder prefers King County, the same pipeline (3 models, SHAP, 80% interval) is portable — only the ETL join changes.
