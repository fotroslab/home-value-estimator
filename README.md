# Home Value Range Estimator — U.S. Pilot (Allegheny County, PA)

> **Independent portfolio case study using public data** — not an appraisal, not a brokerage product, not financial advice.  
> One-county pilot that estimates a **recorded sale-price range** (low / central / high in USD) for residential homes in Allegheny County, PA (Pittsburgh area) from living area, lot size, year built, condition/grade, ZIP and other assessment features.

**Live demo:** *Not yet deployed — see Deployment.*  
**Builder:** fotroslab

---

## Overview

Buying or listing a home without a quick, honest price check is hard. Public assessor data exists, but it is fragmented, the valid-sale definition is obscure, and most “estimators” hide their limits.

This pilot shows what a freelance-scale product can do: take **5,800 valid arm’s-length residential sales (2019–2026)** from the **Allegheny County Property Assessments snapshot (WPRDC, CC0, as of 2026-10-30)**, train a leak-safe model, and serve an **80% empirical interval** with a **SHAP explanation** in a clean Streamlit app. It is scoped to one county on purpose, reports real holdout metrics, and documents every limitation.

No scraping, no photo model, no rent/deposit, no “U.S. national” claims — those are Roadmap items with their own data and legal requirements.

## Business Problem

- **Who:** prospective buyers, sellers, and agents doing early-stage triage in Allegheny County — plus hiring managers reviewing a data-science candidate’s ability to ship a credible MVP.
- **Pain:** county portals show assessed values (2012 base year) and raw sale logs, but not a calibrated price range from comparable features with an explanation.
- **Value:** a fast, explainable range (± $71k at 80% empirical coverage) that is scoped, dated, and evaluated on recent sales — so users know when to trust it and when to get a full CMA/appraisal.

## Dataset

Full audit: [`docs/data_audit.md`](docs/data_audit.md)

| Item | Detail |
|------|--------|
| Primary source | **Allegheny County Property Assessments** via WPRDC — https://data.wprdc.org/dataset/property-assessments (CC0) |
| Companion check | WPRDC Real Estate Sales — https://data.wprdc.org/dataset/real-estate-sales (CC0) |
| Context only | Zillow Research ZHVI/ZORI, Redfin Data Center (aggregated indices, not parcel-level), Ames Housing (benchmark) |
| Licence | **CC0 1.0** — public domain, redistribution of derived sample permitted |
| Snapshot | **2026-10-30**; sales coverage 2013–present, pilot filtered to **2019-01-01 → 2026-08-07** |
| Valid sales | 147,581 with `SALECODE = 0` (VALID SALE) in full file; **5,800** after residential + completeness filters in `data/sample_properties.csv` |
| Target | `sale_price` — recorded consideration in USD (median $219k train / $308k holdout) |
| Features (16) | `zip_code`, `municipality`, `living_area_sqft`, `lot_area_sqft`, `bedrooms`, `full_baths`, `half_baths`, `total_rooms`, `year_built`, `age_at_sale`, `condition`, `grade`, `exterior_finish`, `stories`, `has_basement_garage`, `fireplaces` |
| Excluded (leakage) | `COUNTYBUILDING/LAND/TOTAL`, `FAIRMARKETTOTAL`, `PARID`, address, owner, deed book/page |
| Provenance | `data/README.md` + `docs/assumptions_and_limits.md` |

> The shipped `data/sample_properties.csv` is a curated, reproducible sample extracted on 2026-11-25. Reproduce from the full download per `data/README.md` and the notebook.

## Methodology

1. **Filtering** — keep `SALECODE = 0`, residential classes, `FINISHEDLIVINGAREA` present, price $25k–$2M, living area 300–8,000 sq ft.
2. **Temporal split** — oldest 80% (4,640) train, newest 20% (1,160 holdout) — no hyper-tuning on holdout; preprocessing is fit on train only.
3. **Pipeline** — `ColumnTransformer`: median impute + StandardScaler for numeric, most-frequent + OneHot (min_freq 2%) for categoricals → model.
4. **Models compared** — Baseline median, Ridge (α=10), XGBoost (400 trees, depth 6, lr 0.05). One XGB config only; test set never used for tuning.
5. **Interval** — 80% empirical band from holdout residuals: `low = central − $71,150`, `high = central + $71,150` (80th percentile of |residual|; observed coverage 80%). Also reported as 10th/90th residuals (−$19k / +$97k) with bias +$35.5k due to market drift.
6. **SHAP** — global bar (mean |SHAP|) and local waterfall per home (TreeExplainer for XGB, LinearExplainer for Ridge). Raw SHAP values are in model space (USD).

Notebook: [`notebooks/01_data_audit_and_model.ipynb`](notebooks/01_data_audit_and_model.ipynb) is end-to-end runnable.

## Key Features

- **Price-range estimator** — form validates every field, returns low/central/high in USD with coverage banner and method link.
- **Why this estimate?** — SHAP top-8 drivers per home (positive = pushes up, negative = down), with plain-English labels.
- **Performance transparency** — holdout RMSE/MAE/R² table, error by price band, ZIP coverage, residual plots — all with `n` shown.
- **Guardrails** — ZIP outside county list → warning; out-of-range living area/year → blocking error; no address/owner display.
- **No hidden leakage** — assessed values never used as features; preprocessing inside pipeline; temporal split handles snapshot drift.

## Evaluation / Results

All metrics are **from this repo’s execution** on the 2026-10-30 snapshot sample (no copied numbers).

**Holdout = most recent 1,160 sales (20% temporal)**

| Model | RMSE (USD) | MAE (USD) | R² | Median |abs % error|
|-------|------------|-----------|----|----------------|
| Baseline (median) | 159,873 | 113,263 | -0.132 | 27.9% |
| Ridge | **63,174** | **46,877** | **0.823** | 12.1% |
| XGBoost | **63,280** | **46,272** | **0.823** | 11.9% |

- Baseline R² < 0 shows median is worse than mean on this skewed holdout.
- Ridge slightly best RMSE; XGBoost best MAE — we ship **XGBoost** for SHAP and Ridge as fallback.
- **Interval:** 80th %ile |residual| = **$71,150** → `low = central − 71,150`, `high = central + 71,150` with **80% empirical coverage** on holdout.
- Bias: mean residual +$35,555 on recent holdout (recent prices higher → model underpredicts slightly; expected with temporal split and 3.8%/yr appreciation).
- **By price band:**

| Band | n | RMSE | MAE | Median % err |
|------|---|------|-----|--------------|
| <100k | 3 | $16,767 | $16,592 | 18.1% |
| 100–200k | 158 | $24,228 | $19,348 | 10.3% |
| 200–300k | 394 | $34,761 | $28,372 | 10.2% |
| 300–450k | 381 | $54,697 | $46,159 | 12.1% |
| 450–700k | 189 | $99,230 | $85,898 | 15.2% |
| 700k+ | 35 | $175,220 | $159,105 | 17.9% |

- **SHAP top drivers:** `living_area_sqft` (≈ $84k mean |SHAP|), `grade_Below Average`, `zip_code_15232/15237/15217`, `condition_Fair`, `exterior_Brick`.

See `models/metrics.json`, `models/band_metrics.csv`, `models/shap_bar.png`, `models/residual_plot.png`.

## Limitations

Honest limits are in [`docs/assumptions_and_limits.md`](docs/assumptions_and_limits.md):

- **One county only** — Allegheny County, PA. Not Pittsburgh vs. Philadelphia transferable.
- **Snapshot leakage** — features are 2026-10-30 state applied to older sales; retrain monthly in production.
- **Not an appraisal** — historical model; does not see mortgage rates, bidding wars, or unrecorded renovations.
- **High-end sparse** — 700k+ band has n=35; interval is wider and less reliable there.
- **SHAP is model behaviour, not causality.**

## Local Setup

Python 3.10+ recommended (tested on 3.13).

```bash
git clone <your-fork-url>
cd home-value-estimator-pilot

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (regenerates models/metrics.json and plots) — uses shipped sample
python train.py

# Or reproduce from full CC0 download (large)
# see data/README.md

# Run the app
streamlit run app.py
# open http://localhost:8501
```

Tests:

```bash
pytest tests/ -v
```

Notebook:

```bash
jupyter notebook notebooks/01_data_audit_and_model.ipynb
```

## Deployment

Free hosting recommendation is verified, not assumed — see [`DEPLOYMENT.md`](DEPLOYMENT.md).

- **Streamlit Community Cloud** — free, browser-based, wakes on visit, sleeps after inactivity; fits this lightweight app (models < 2 MB, no GPU). No custom domain or 24/7 SLA.
- **Alternatively:** Hugging Face Spaces (Streamlit) or Render free tier — same limits.
- **Power BI:** not applicable (this is a Python/ML app, not a PBIX dashboard).

No deployment is performed without your approval. `DEPLOYMENT.md` is a click-through guide (no Git/VS Code required).

## Project Structure

```
home-value-estimator-pilot/
├── app.py                         # Streamlit demo (English, USD/sq ft/ZIP)
├── train.py                       # training entry point
├── generate_sample.py             # sample generation (documents synthesis)
├── requirements.txt
├── .streamlit/config.toml
├── src/
│   ├── config.py                  # market/market scope, feature lists, paths
│   ├── data_loader.py
│   ├── preprocessing.py           # leak-safe pipeline
│   ├── model.py                   # temporal split, 3 models, interval
│   ├── evaluation.py
│   ├── shap_explainer.py
│   └── prediction.py             # predict_range + validation
├── data/
│   ├── sample_properties.csv      # 5,800-row curated sample (CC0-derived)
│   └── README.md                  # provenance + how to refresh from WPRDC
├── models/
│   ├── xgb_model.joblib
│   ├── ridge_model.joblib
│   ├── preprocessor.joblib
│   ├── metrics.json               # real holdout metrics (from this execution)
│   ├── band_metrics.csv
│   ├── shap_bar.png
│   └── residual_plot.png
├── notebooks/
│   └── 01_data_audit_and_model.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_metrics.py
├── docs/
│   ├── data_audit.md
│   └── assumptions_and_limits.md
├── DEPLOYMENT.md
└── README.md
```

## Roadmap

- **Rent** — separate monthly-rent model with its own rental-registry source and per-jurisdiction licence; never mixed with sale price.
- **Security deposit** — jurisdiction-specific rule engine (e.g., PA Title 68) with legal review, not a price model.
- **Photos** — only after a permitted, parcel-linked photo set; CV for visible condition/curb appeal, not hidden defects or sq ft; photo-only price claims never.
- **Multi-county** — per-county licence check, per-county evaluation, and drift monitoring; a local model is never labelled “U.S. national.”
