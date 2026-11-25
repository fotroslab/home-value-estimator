"""Model training: baseline, Ridge, XGBoost with leak-safe pipeline."""
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from .config import TARGET_COL, PRICE_MIN, PRICE_MAX, MODEL_PATH_RIDGE, MODEL_PATH_XGB, PREPROCESSOR_PATH, METRICS_PATH
from .preprocessing import build_preprocessor

def temporal_train_test_split(df: pd.DataFrame, date_col="sale_date", test_frac=0.2):
    """Time-based split: most recent 20% as holdout. Falls back to random if date missing."""
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df_sorted = df.sort_values(date_col)
        n = len(df_sorted)
        split_idx = int(n * (1 - test_frac))
        train = df_sorted.iloc[:split_idx]
        test = df_sorted.iloc[split_idx:]
        return train, test
    else:
        return train_test_split(df, test_size=test_frac, random_state=42)

def train_all(df: pd.DataFrame):
    """Train 3 models on same split, return metrics and fitted pipelines."""
    # Basic filter
    df = df[(df[TARGET_COL] >= PRICE_MIN) & (df[TARGET_COL] <= PRICE_MAX)].copy()
    df = df.dropna(subset=[TARGET_COL])
    # Remove rows with missing critical features? Let imputer handle, but need at least living area
    df = df.dropna(subset=["living_area_sqft"])

    X = df.drop(columns=[TARGET_COL, "parid", "sale_date", "sale_code", "school_district", "neighborhood", "property_class", "use_code"], errors="ignore")
    y = df[TARGET_COL]

    # Ensure holdout is temporal
    train_df = pd.concat([X, y], axis=1)
    train_df["sale_date"] = df["sale_date"].values if "sale_date" in df.columns else pd.NaT
    # Use temporal split via sorting on sale_date
    full = df.copy()
    tr, te = temporal_train_test_split(full, "sale_date", 0.2)
    X_train = tr.drop(columns=[TARGET_COL, "parid", "sale_date", "sale_code", "school_district", "neighborhood", "property_class", "use_code"], errors="ignore")
    y_train = tr[TARGET_COL]
    X_test = te.drop(columns=[TARGET_COL, "parid", "sale_date", "sale_code", "school_district", "neighborhood", "property_class", "use_code"], errors="ignore")
    y_test = te[TARGET_COL]

    preprocessor = build_preprocessor()

    # Baseline: median
    dummy = Pipeline([("preprocess", build_preprocessor()), ("model", DummyRegressor(strategy="median"))])
    dummy.fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test)

    # Ridge
    ridge = Pipeline([("preprocess", build_preprocessor()), ("model", Ridge(alpha=10.0))])
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
                tree_method="hist"
            ))
        ])
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        has_xgb = True
    except Exception as e:
        print(f"XGBoost failed: {e}")
        xgb = ridge
        pred_xgb = pred_ridge
        has_xgb = False

    def metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        # MAPE-like median abs pct error
        mape = float(np.median(np.abs(y_true - y_pred) / np.maximum(y_true, 1)))
        return {"rmse": rmse, "mae": mae, "r2": r2, "median_abs_pct_error": mape}

    results = {
        "baseline_median": metrics(y_test, pred_dummy),
        "ridge": metrics(y_test, pred_ridge),
        "xgboost": metrics(y_test, pred_xgb),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "y_test_mean": float(y_test.mean()),
        "y_test_median": float(y_test.median()),
    }

    # Interval calibration: use residuals of best model (XGBoost if available else Ridge)
    best_pred = pred_xgb if has_xgb else pred_ridge
    residuals = y_test.values - best_pred
    # Empirical 80% interval: 10th and 90th percentiles of residuals
    q_low, q_high = np.percentile(residuals, [10, 90])
    coverage = float(np.mean((residuals >= q_low) & (residuals <= q_high)))
    results["interval"] = {
        "q10_residual": float(q_low),
        "q90_residual": float(q_high),
        "empirical_coverage_80": coverage,
        "method": "Empirical residual quantiles (10th/90th) from holdout; applied as prediction ± quantile"
    }

    # Save artifacts for best model
    best_model = xgb if has_xgb else ridge
    # Save separately for app
    joblib.dump(best_model, MODEL_PATH_XGB if has_xgb else MODEL_PATH_RIDGE)
    # Also save ridge for comparison
    joblib.dump(ridge, MODEL_PATH_RIDGE)
    if has_xgb:
        joblib.dump(xgb, MODEL_PATH_XGB)
    joblib.dump(best_model.named_steps["preprocess"], PREPROCESSOR_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results, {"dummy": dummy, "ridge": ridge, "xgb": xgb}, (X_test, y_test, best_pred, residuals)
