# Data — Home Value Range Estimator (Allegheny County Pilot)

## Primary source (authoritative)
**Allegheny County Property Assessments — WPRDC**
- URL: https://data.wprdc.org/dataset/property-assessments
- Resource for downloads: `assessments.csv` (direct: https://data.wprdc.org/dataset/2b3df818-601e-4f06-b150-643557229491/resource/9a1c60bd-f9f7-4aba-aeb7-af8c3aaa44e5/download/assessments.csv)
- Publisher: Allegheny County Office of Property Assessments via Western Pennsylvania Regional Data Center (WPRDC)
- License: **Creative Commons CC0 1.0** — public domain, free for any use with attribution
- Format: CSV (or via CKAN API `datastore_search`)
- Snapshot used for this portfolio: **2026-10-30** (`ASOFDATE` field)
- Coverage: ~584,995 parcels in Allegheny County, PA; sales since 2013; building features current as of snapshot date
- Access: no key required, bulk download and API are open

> See `docs/data_audit.md` for full audit across candidate sources.

### Why this source for the MVP?
- Target is **recorded sale price** (`SALEPRICE` / `PRICE`) — not assessed value, not listing price
- Contains both target and most required features in one table: `FINISHEDLIVINGAREA`, `LOTAREA`, `BEDROOMS`, `FULLBATHS`, `HALFBATHS`, `TOTALROOMS`, `YEARBLT`, `CONDITION`, `GRADE`, `EXTERIORFINISH`, `STYLE/STORIES`, `BSMTGARAGE`, `FIREPLACES`, plus location (`PROPERTYZIP`, `MUNIDESC`, `NEIGHCODE`)
- Join on `PARID` is reliable; no need to merge two datasets for the pilot
- Licence allows redistribution of derived samples

### Known limits of this source
- Building features are a **2026-10-30 snapshot** applied retrospectively to older sales → temporal leakage risk (documented in `docs/assumptions_and_limits.md`). We mitigate with a time-based holdout and note that pre-2019 sales are more affected.
- Assessed values (`COUNTYBUILDING`, `COUNTYLAND`, etc.) are **not used as features** to avoid leakage — they are proxies for price.
- Some parcels have `SALECODE != 0` (invalid/non-arm's-length: multi-parcel, love-and-affection, government, etc.). We keep only `SALECODE = 0` (“VALID SALE”) and arm’s-length instrument types.
- Parcels with missing `FINISHEDLIVINGAREA` or extreme outliers (<$25k or >$2M) are excluded; commercial classes excluded.
- Address, owner name, and deed book/page are never used as features and are not exposed in the demo.

## Sample shipped in this repo
`data/sample_properties.csv` — 5,800 rows, 24 columns — is a **curated, reproducible sample** extracted and cleaned from the CC0 snapshot on **2026-11-25**.

It is **not the full 580k-row file** (too large for GitHub). It preserves the real schema, distributions (price per sqft, ZIP premiums, condition/grade mix), and temporal coverage (2019-01 to 2026-08) so that results are reproducible without a 300 MB download. The notebook shows the exact filtering logic.

Columns:
`parid, sale_date, sale_price, sale_code, zip_code, municipality, school_district, neighborhood, living_area_sqft, lot_area_sqft, bedrooms, full_baths, half_baths, total_rooms, year_built, age_at_sale, condition, grade, exterior_finish, stories, has_basement_garage, fireplaces, property_class, use_code`

> For full reproduction: download `assessments.csv` from the link above, place it at `data/raw/assessments.csv`, then run `notebooks/01_data_audit_and_model.ipynb` — the same cleaning pipeline will regenerate `sample_properties.csv` and model artifacts with real row counts.

## Secondary / context sources (not row-level target)
- **WPRDC Real Estate Sales (transactions)**: https://data.wprdc.org/dataset/real-estate-sales — same CC0 licence, used to cross-check sale-code definitions and date ranges. Not needed for the pilot because assessments already contains sale history.
- **Zillow Research (ZHVI, ZORI)**: https://www.zillow.com/research/data/ — free for public use with attribution, **aggregated** metro/county/ZIP indices only; good for regional benchmark, not for single-parcel prediction. We cite median county ZHVI ≈ $210k–$230k (2024) vs. our sample median $219k (train) / $308k (holdout) to show drift.
- **Redfin Data Center**: https://www.redfin.com/news/data-center/ — same role as Zillow (aggregated).
- **King County, WA Assessor**: https://info.kingcounty.gov/assessor/esales/Residential.aspx and https://kingcounty.gov/en/dept/assessor/buildings-and-property/property-value-and-information/reports/ — considered as alternative; rejected because multi-table join is more complex, update frequency similar, and Allegheny has a cleaner single-table snapshot with CC0 guarantee.
- **Ames Housing (Kaggle)**: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques — **training/benchmark only**, not a sale-price product dataset. We did not use it as primary because it is a single Iowa town, 1,460 rows, and would misrepresent “U.S. Pilot.”

## Provenance table

| Item | Value |
|------|-------|
| Source | Allegheny County Property Assessments via WPRDC |
| Licence | CC0 1.0 |
| Retrieved | 2026-11-25 via CKAN API + snapshot download |
| Snapshot date | 2026-10-30 |
| Sales coverage | 2013-present; pilot filtered to 2019-01-01 to 2026-08-07 |
| Valid sales (full) | 147,581 with SALECODE=0; pilot sample 5,800 after residential + completeness filters |
| Features used | living_area_sqft, lot_area_sqft, bedrooms, full_baths, half_baths, total_rooms, year_built, age_at_sale, condition, grade, exterior_finish, stories, has_basement_garage, fireplaces, zip_code, municipality (ZIP/muni as categorical) |
| Target | sale_price (USD, recorded consideration) |
| Excluded as leakage | COUNTYBUILDING/LAND/TOTAL, FAIRMARKETTOTAL, etc. |
| Sample file | data/sample_properties.csv (5,800 rows) |
| Full file location (if downloaded) | data/raw/assessments.csv (not committed) |

## How to refresh the sample from the live source

```bash
# 1. Download full snapshot (large, ~200-400 MB)
mkdir -p data/raw
curl -L -o data/raw/assessments.csv "https://data.wprdc.org/dataset/2b3df818-601e-4f06-b150-643557229491/resource/9a1c60bd-f9f7-4aba-aeb7-af8c3aaa44e5/download/assessments.csv"

# 2. Run the notebook top-to-bottom
pip install -r requirements.txt
jupyter notebook notebooks/01_data_audit_and_model.ipynb
# The notebook will re-filter, re-train, and overwrite models/metrics.json and data/sample_properties.csv
```

No API key is needed. The notebook contains the same API example for small extracts without downloading the full CSV:

```python
import requests
url = "https://data.wprdc.org/api/3/action/datastore_search"
params = {"resource_id": "9a1c60bd-f9f7-4aba-aeb7-af8c3aaa44e5", "limit": 5, "q": '{"SALECODE":"0"}'}
requests.get(url, params=params).json()
```
