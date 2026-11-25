"""Train models and produce artifacts."""
import pandas as pd
from src.data_loader import load_sample
from src.model import train_all

df = load_sample()
print(f"Loaded {len(df)} rows")
print(df.head())
results, models, extra = train_all(df)
print("\n=== RESULTS ===")
import json, pathlib
print(json.dumps(results, indent=2))

# Also compute per-band and per-zip analysis
from src.evaluation import price_band_analysis, zip_analysis, residual_plot
X_test, y_test, y_pred, residuals = extra
# Need X_test with zip for zip analysis
# We have test split inside train_all, but we recompute? Instead load again and split same way
# For evaluation we can use X_test from extra's model pipeline? We have y_test/y_pred
# Let's do band analysis
band = price_band_analysis(y_test.values, y_pred, y_test.values)
print("\nBand analysis:")
print(band.to_string())

# Save band to models dir
band.to_csv("models/band_metrics.csv", index=False)

# Residual plot
residual_plot(y_test.values, y_pred, "models/residual_plot.png")
print("Saved residual plot")

# SHAP
from src.shap_explainer import compute_shap, plot_shap_bar
import pandas as pd
# Re-split to get X_train sample for SHAP
from src.model import temporal_train_test_split
tr, te = temporal_train_test_split(df, "sale_date", 0.2)
feature_cols = ["zip_code","municipality","living_area_sqft","lot_area_sqft","bedrooms","full_baths","half_baths","total_rooms","year_built","age_at_sale","condition","grade","exterior_finish","stories","has_basement_garage","fireplaces"]
X_train = tr[feature_cols]
X_test_f = te[feature_cols]

# Use best model (XGB)
best = models.get("xgb") if models.get("xgb") is not models.get("ridge") else models["ridge"]
print(f"Best model type: {type(best.named_steps['model'])}")

shap_vals, feat_names, importance, explainer = compute_shap(best, X_train.sample(500, random_state=42) if len(X_train)>500 else X_train, X_test_f.sample(300, random_state=42) if len(X_test_f)>300 else X_test_f)
if importance is not None:
    print(importance.head(10))
    importance.to_csv("models/shap_importance.csv", index=False)
    plot_shap_bar(importance, "models/shap_bar.png")
    print("Saved SHAP bar")

# Save sample prediction for local explanation
sample_input = X_test_f.iloc[0].to_dict()
print("Sample input for local explanation:")
print(sample_input)
# Save to json for app
import json
with open("models/sample_input.json","w") as f:
    json.dump(sample_input, f, indent=2, default=str)

print("Training complete.")
