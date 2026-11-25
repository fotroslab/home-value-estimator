"""Data loading utilities.

Primary source: Allegheny County Property Assessments (WPRDC, CC0)
Snapshot: 2026-10-30
Secondary source for context: WPRDC Property Sales Transactions

For reproducibility in this portfolio repo we ship a curated sample
`data/sample_properties.csv` extracted from the assessments snapshot
on 2026-11-25 using the logic in notebooks/01_data_audit_and_model.ipynb.

If you have the full assessments.csv, set DATA_FULL_PATH.
"""
from pathlib import Path
import pandas as pd

from .config import PROJECT_ROOT

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_properties.csv"
RAW_ASSESSMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "assessments.csv"
RAW_SALES_PATH = PROJECT_ROOT / "data" / "raw" / "sales.csv"

EXPECTED_COLS = [
    "parid", "sale_date", "sale_price", "sale_code", "zip_code", "municipality",
    "school_district", "neighborhood",
    "living_area_sqft", "lot_area_sqft", "bedrooms", "full_baths", "half_baths",
    "total_rooms", "year_built", "age_at_sale", "condition", "grade",
    "exterior_finish", "stories", "has_basement_garage", "fireplaces",
    "property_class", "use_code"
]

def load_sample() -> pd.DataFrame:
    """Load curated sample (always available)."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Sample file not found at {SAMPLE_PATH}. "
            "Run notebooks/01_data_audit_and_model.ipynb to generate it, "
            "or download full data per data/README.md"
        )
    df = pd.read_csv(SAMPLE_PATH, parse_dates=["sale_date"])
    return df

def load_full_assessments(nrows=None) -> pd.DataFrame:
    """Load full assessments.csv if present (heavy, ~500k rows)."""
    if not RAW_ASSESSMENTS_PATH.exists():
        raise FileNotFoundError(f"Full file not at {RAW_ASSESSMENTS_PATH}")
    return pd.read_csv(RAW_ASSESSMENTS_PATH, nrows=nrows, low_memory=False)

def describe_dataset(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "date_min": str(df["sale_date"].min()),
        "date_max": str(df["sale_date"].max()),
        "price_median": float(df["sale_price"].median()),
        "price_mean": float(df["sale_price"].mean()),
        "missing_pct": df.isna().mean().round(3).to_dict(),
    }
