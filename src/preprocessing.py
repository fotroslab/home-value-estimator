"""Preprocessing pipeline — leak-safe, fit only on train."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from .config import CATEGORICAL_COLS, NUMERIC_COLS

def build_preprocessor():
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.02)),
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning that does NOT leak (no target use)."""
    df = df.copy()
    # Coerce numeric
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Clip to plausible ranges (also enforced in UI)
    if "living_area_sqft" in df.columns:
        df["living_area_sqft"] = df["living_area_sqft"].clip(300, 8000)
    if "lot_area_sqft" in df.columns:
        df["lot_area_sqft"] = df["lot_area_sqft"].clip(0, 100000)
    if "bedrooms" in df.columns:
        df["bedrooms"] = df["bedrooms"].clip(0, 8)
    if "full_baths" in df.columns:
        df["full_baths"] = df["full_baths"].clip(0, 6)
    if "half_baths" in df.columns:
        df["half_baths"] = df["half_baths"].clip(0, 4)
    if "year_built" in df.columns:
        df["year_built"] = df["year_built"].clip(1850, 2026)
    # Compute age_at_sale if missing and sale_date present? In app we compute from sale year = 2024 reference
    return df

def get_feature_names(preprocessor) -> list:
    """Return output feature names after fitting."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        # fallback
        return []
