"""Project-wide constants and configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Geographic and time scope — must match README/app
MARKET_NAME = "Allegheny County, PA (Pittsburgh area)"
MARKET_SHORT = "Allegheny County, PA"
DATA_AS_OF = "2026-10-30"
SALES_COVERAGE = "2013-01-01 to 2026-08-07 (WPRDC snapshot 2026-10-30)"
MODEL_COVERAGE_NOTE = "Model trained on valid arm's-length residential sales 2019-2026 with building features as of 2026-10-30 snapshot"

# Target
TARGET_COL = "sale_price"
TARGET_LOG = False  # we train on raw USD; keep option for future

# Feature columns used in model (human-readable names mapped internally)
FEATURE_COLS = [
    "zip_code",
    "municipality",
    "living_area_sqft",
    "lot_area_sqft",
    "bedrooms",
    "full_baths",
    "half_baths",
    "total_rooms",
    "year_built",
    "age_at_sale",
    "condition",
    "grade",
    "exterior_finish",
    "stories",
    "has_basement_garage",
    "fireplaces",
]

CATEGORICAL_COLS = ["zip_code", "municipality", "condition", "grade", "exterior_finish"]
NUMERIC_COLS = [
    "living_area_sqft",
    "lot_area_sqft",
    "bedrooms",
    "full_baths",
    "half_baths",
    "total_rooms",
    "year_built",
    "age_at_sale",
    "stories",
    "has_basement_garage",
    "fireplaces",
]

# Price filters for training (USD)
PRICE_MIN = 25_000
PRICE_MAX = 2_000_000

# Living area filters (sq ft)
LIVING_AREA_MIN = 300
LIVING_AREA_MAX = 8_000

# Lot area filters
LOT_AREA_MIN = 0
LOT_AREA_MAX = 100_000

# Model artifacts
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH_RIDGE = MODEL_DIR / "ridge_model.joblib"
MODEL_PATH_XGB = MODEL_DIR / "xgb_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SHAP_PATH = MODEL_DIR / "shap_values.npz"

# Interval calibration
INTERVAL_LEVEL = 0.80  # 80% empirical interval

# App limits for validation
ZIP_CODES = [
    15006, 15044, 15090, 15101, 15102, 15108, 15116, 15120, 15129, 15131,
    15132, 15136, 15139, 15143, 15146, 15202, 15206, 15208, 15210, 15213,
    15217, 15219, 15221, 15222, 15224, 15226, 15227, 15228, 15232, 15235,
    15237, 15241
]
MUNICIPALITIES = [
    "Pittsburgh (1st-31st Wards)", "Mt. Lebanon", "Bethel Park", "McCandless",
    "Penn Hills", "Upper St. Clair", "Ross", "Monroeville", "White Oak",
    "South Park", "Franklin Park", "Fox Chapel", "Shaler", "Baldwin",
    "Scott", "Wilkinsburg", "Collier", "Hampton"
]
CONDITIONS = ["Excellent", "Very Good", "Good", "Average", "Fair", "Poor"]
GRADES = ["Excellent", "Very Good", "Good", "Average", "Below Average", "Poor"]
EXTERIORS = ["Brick", "Frame", "Stone", "Vinyl/Aluminum", "Stucco", "Block", "Other"]
