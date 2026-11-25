from src.prediction import predict_range, validate_input

def test_predict_range():
    inp = {
        "zip_code": 15217,
        "municipality": "Pittsburgh (1st-31st Wards)",
        "living_area_sqft": 1800,
        "lot_area_sqft": 5000,
        "bedrooms": 3,
        "full_baths": 2,
        "half_baths": 1,
        "total_rooms": 7,
        "year_built": 1965,
        "age_at_sale": 59,
        "condition": "Good",
        "grade": "Good",
        "exterior_finish": "Brick",
        "stories": 2,
        "has_basement_garage": 1,
        "fireplaces": 1
    }
    errs = validate_input(inp)
    assert errs == [], f"unexpected validation errors {errs}"
    out = predict_range(inp)
    assert "central" in out and "low" in out and "high" in out
    assert out["low"] < out["central"] < out["high"]
    assert out["central"] >= 25000 and out["central"] <= 2000000
    print(f"prediction OK: {out}")

def test_validate_rejects():
    inp = {"living_area_sqft": 100, "lot_area_sqft": 5000, "bedrooms": 3, "full_baths": 2, "year_built": 1965}
    errs = validate_input(inp)
    assert len(errs) > 0
    print("validation reject OK")

if __name__ == "__main__":
    test_predict_range()
    test_validate_rejects()
