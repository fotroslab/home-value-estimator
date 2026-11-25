"""Prediction + interval logic for the app."""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .config import MODEL_PATH_RIDGE, MODEL_PATH_XGB, PREPROCESSOR_PATH, METRICS_PATH, PRICE_MIN, PRICE_MAX

def load_artifacts():
    # Prefer XGBoost if present
    if MODEL_PATH_XGB.exists():
        model = joblib.load(MODEL_PATH_XGB)
    elif MODEL_PATH_RIDGE.exists():
        model = joblib.load(MODEL_PATH_RIDGE)
    else:
        raise FileNotFoundError("No model found. Train first.")
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, metrics

def predict_range(input_dict: dict):
    """Given user input dict, return central, low, high + warnings."""
    model, metrics = load_artifacts()
    df = pd.DataFrame([input_dict])
    # Ensure expected cols
    # Compute age_at_sale if not provided (assume sale year 2024 for display)
    if "age_at_sale" not in df.columns and "year_built" in df.columns:
        df["age_at_sale"] = 2024 - df["year_built"]
    # Predict
    central = float(model.predict(df)[0])
    # Clip to plausible
    central = float(np.clip(central, PRICE_MIN, PRICE_MAX))

    # Interval from metrics — prefer symmetric 80% band for user intuition
    interval = metrics.get("interval", {})
    # Prefer symmetric abs80 if available, else fall back to q10/q90
    if "abs80" in interval:
        delta = float(interval["abs80"])
        q_low, q_high = -delta, delta
    else:
        q_low = float(interval.get("q10_residual", -35000))
        q_high = float(interval.get("q90_residual", 35000))

    low = float(max(PRICE_MIN, central + q_low))
    high = float(min(PRICE_MAX, central + q_high))

    # Ensure low < central < high
    if low >= central:
        low = central * 0.85
    if high <= central:
        high = central * 1.15

    # Warnings for out-of-distribution
    warnings = []
    if df["living_area_sqft"].iloc[0] < 500 or df["living_area_sqft"].iloc[0] > 5000:
        warnings.append("Living area is outside typical training range (500–5,000 sq ft); estimate is less reliable.")
    if df["year_built"].iloc[0] < 1900:
        warnings.append("Year built is very old; training data is sparse for pre-1900 homes.")
    if df["lot_area_sqft"].iloc[0] > 50000:
        warnings.append("Lot size is unusually large for this county; interval may be wider.")
    # Check ZIP
    from .config import ZIP_CODES
    if "zip_code" in df.columns and int(df["zip_code"].iloc[0]) not in ZIP_CODES:
        warnings.append("ZIP code is outside the most common areas in training data; estimate uses municipality average.")

    return {
        "central": central,
        "low": low,
        "high": high,
        "warnings": warnings,
        "model_name": "XGBoost (tree ensemble)" if MODEL_PATH_XGB.exists() else "Ridge Regression",
        "coverage_note": f"Interval calibrated on holdout: ~{interval.get('empirical_coverage_80', 0.8)*100:.0f}% empirical coverage at 80% nominal (q10={q_low:,.0f}, q90={q_high:,.0f})"
    }

def validate_input(input_dict: dict):
    errors = []
    if input_dict.get("living_area_sqft", 0) < 300 or input_dict.get("living_area_sqft", 0) > 8000:
        errors.append("Living area must be between 300 and 8,000 sq ft.")
    if input_dict.get("lot_area_sqft", 0) < 0 or input_dict.get("lot_area_sqft", 0) > 100_000:
        errors.append("Lot area must be between 0 and 100,000 sq ft.")
    if input_dict.get("bedrooms", 0) < 0 or input_dict.get("bedrooms", 0) > 8:
        errors.append("Bedrooms must be 0–8.")
    if input_dict.get("full_baths", 0) < 0 or input_dict.get("full_baths", 0) > 6:
        errors.append("Full baths must be 0–6.")
    if input_dict.get("year_built", 2000) < 1850 or input_dict.get("year_built", 2000) > 2026:
        errors.append("Year built must be 1850–2026.")
    if input_dict.get("sale_price") is not None:
        # not expected from user, but catch leakage attempt
        errors.append("Sale price cannot be provided as input.")
    return errors
