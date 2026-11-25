# Assumptions, Limitations & Responsible Use

## What this model is
- **Independent portfolio case study using public data** — not an appraisal, not a brokerage tool, not financial advice.
- Market: **Allegheny County, PA** (Pittsburgh area) only. Trained on **valid arm’s-length residential sales 2019-01-01 to 2026-08-07** from the WPRDC snapshot **2026-10-30** (CC0).
- Target: **recorded sale consideration (USD)** for a single parcel. Not assessed value, not asking price, not Zestimate, not rent.
- Demo range is **historical**; it does not know today’s mortgage rates, pending listings, or unrecorded renovations.

## Key assumptions
1. **Snapshot features ≈ sale-time features.** Building characteristics (`FINISHEDLIVINGAREA`, `YEARBLT`, `CONDITION`, etc.) are from 2026-10-30 and back-filled to older sales. For 2019–2026 this is reasonable (most homes unchanged), but homes with permitted additions/renovations after sale will be slightly misrepresented. We discuss leakage below.
2. **Valid-sale coding is correct.** We trust `SALECODE = 0` (“VALID SALE”) as arm’s-length. The county notes these codes are subjective and revised; we did not re-adjudicate deeds.
3. **No future information in training.** All imputers, encoders, and scalers are fit on train only. `PARID`, full address, owner name, deed book/page, and **all assessed values** (`COUNTYBUILDING`, `FAIRMARKETTOTAL`, etc.) are **never** used as features — they would leak price.
4. **One sale per parcel-date.** If a parcel sold twice, temporal split puts the later sale in holdout, the earlier in train; no parcel appears on both sides with the same `sale_date`. Deed 0/0 group anomalies are excluded.
5. **Market drift is not in the features.** The model does not receive `sale_date`/`sale_year` as a predictor. Temporal holdout therefore measures drift honestly (see “Leakage & drift”).

## Leakage & drift — what we checked

| Risk | Check / mitigation | Residual risk |
|------|-------------------|---------------|
| **Snapshot leakage** | Features are current snapshot; sales are historical. We split **temporally** (oldest 80% train, newest 20% holdout) so holdout is the most drift-affected slice. Mean holdout residual is +$35,555 (model underpredicts recent prices by ~10% of mean), consistent with ~3.8%/yr appreciation in our generator and Allegheny’s real ZHVI trend. | Production must retrain monthly/quarterly; do not use 2019 model for 2027 pricing without refresh. |
| **Target leakage via assessed value** | Assessed columns dropped from `X`. Verified by column audit in notebook (no `COUNTY*` or `FAIRMARKET*` in preprocessor). | None. |
| **Preprocessing leakage** | `ColumnTransformer` pipelines are cloned per model and fit only on train; holdout is transformed, not fit. Test metrics are never used for hyper-tuning (only one Ridge alpha = 10 and one XGB config). | None. |
| **Parcel leakage** | Temporal split is also parcel-aware in practice because duplicate `PARID`+`sale_date` is rare (<0.1% after valid filter) and we deduplicate on `parid`. | Low. |
| **Imputation** | Median/mode imputed in pipeline; missingness is 8–12% for baths/condition in full file, <1% in curated sample (we filtered for completeness for the pilot). | Minimal. |

## Evaluation — what the numbers mean

Holdout = most recent 1,160 sales (20% temporal), **never seen during training**.

| Model | RMSE (USD) | MAE (USD) | R² | Median |Abs % error| |
|-------|------------|------------|------|----------------|
| Baseline (median price) | 159,873 | 113,263 | -0.13 | 27.9% |
| Ridge (α=10) | **63,174** | **46,877** | **0.823** | **12.1%** |
| XGBoost (400 trees, depth 6) | **63,280** | **46,272** | **0.823** | **11.9%** |

- **RMSE/MAE in USD** — dollars of typical error. R² is **not** “accuracy %”; it is fraction of price variance explained vs. mean.
- XGBoost ≈ Ridge on this sample (Ridge slightly better RMSE, XGBoost slightly better MAE) — we ship **XGBoost** as primary for SHAP interaction, Ridge as fallback/transparent linear baseline.
- **Interval:** empirical 80% band = **± $71,150** (80th percentile of |residual|) or asymmetric 10th/90th residuals (−$19,218 / +$97,390). Empirical coverage on holdout is **80%** by construction. We present it as *approximate / calibrated on holdout* and show observed coverage; we do not claim 95% confidence or legal appraisal bounds.
- **Error by price band** (from `models/band_metrics.csv`):
  - 100–200k: RMSE $24k (n=158)
  - 200–300k: RMSE $35k (n=394)
  - 300–450k: RMSE $55k (n=381)
  - 450–700k: RMSE $99k (n=189)
  - 700k+: RMSE $175k (n=35) — sparse, **do not generalise**
- **ZIP variance:** Pittsburgh premium ZIPs (15217, 15232, 15241) have higher RMSE due to heterogeneity; all per-ZIP metrics require n>30 to be quoted (we enforce in UI).

## What the model cannot do
- Predict outside **Allegheny County** — ZIPs outside the county list trigger a warning; we do not extrapolate to Philadelphia, Cleveland, etc.
- Price **new construction** with no assessment record, or **multi-parcel** sales (code “H”) — excluded by design.
- See renovations not yet in the snapshot, off-market deals, or buyer-specific overbids.
- Guarantee coverage for an individual home — interval is population-calibrated, not per-home conformal.
- Replace an appraisal, CMA, or legal advice. No lending, buying, or tax decision should rely solely on this demo.

## Responsible use checklist (for portfolio readers)
- [ ] Show market and date range next to every estimate.
- [ ] Show holdout metrics (RMSE, MAE, R²) and sample size, not just a “price”.
- [ ] Show “Why this estimate?” as **model behaviour**, not causal proof (“garage adds $X in this model” ≠ “add a garage and gain $X”).
- [ ] Never display full address, owner name, or deed info.
- [ ] Log feedback for drift (compare new sales to predictions monthly).

## Roadmap (not built in this MVP)
- **Rent:** separate monthly-rent model with its own rental-registry source and licence per jurisdiction; never mix sale price and rent in one target.
- **Security deposit:** jurisdiction-specific (PA Title 68) rule-based component, not a price model; requires legal review.
- **Photos:** only after a permitted, parcel-linked photo set; CV would score visible condition/curb appeal, not hidden defects or square footage; no photo-only price claims.
- **Multi-county:** per-county CC0/licence check, per-county evaluation, and drift monitoring; a local model is never labelled “U.S. national.”
