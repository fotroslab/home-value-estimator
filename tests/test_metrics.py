import json, pathlib
from src.config import METRICS_PATH, PRICE_MIN, PRICE_MAX

def test_metrics_exist():
    assert METRICS_PATH.exists(), "metrics.json missing — run train.py"
    with open(METRICS_PATH) as f:
        m = json.load(f)
    for k in ["baseline_median","ridge","xgboost"]:
        assert k in m
        assert "rmse" in m[k]
        assert m[k]["rmse"] > 0
        assert m[k]["rmse"] < 300000
        assert -1 <= m[k]["r2"] <= 1
    # Ridge should beat baseline
    assert m["ridge"]["rmse"] < m["baseline_median"]["rmse"]
    # Interval sanity
    assert "interval" in m
    assert m["interval"]["empirical_coverage_80"] >= 0.75
    print("metrics OK:", {k: round(m[k]["rmse"]) for k in ["baseline_median","ridge","xgboost"]})

def test_sample_file():
    import pandas as pd
    from src.config import PROJECT_ROOT
    p = PROJECT_ROOT / "data" / "sample_properties.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) >= 5000
    assert "sale_price" in df.columns
    assert df["sale_price"].between(PRICE_MIN, PRICE_MAX).mean() > 0.95
    print(f"sample file OK: {len(df)} rows")

if __name__ == "__main__":
    test_metrics_exist()
    test_sample_file()
