import pandas as pd
from src.preprocessing import build_preprocessor, clean_input_df
from src.config import FEATURE_COLS

def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "zip_code": [15217, 15217, 15206],
        "municipality": ["Pittsburgh (1st-31st Wards)", "Mt. Lebanon", "Shaler"],
        "living_area_sqft": [1500, 1800, None],
        "lot_area_sqft": [4000, 5000, 6000],
        "bedrooms": [3, 4, 3],
        "full_baths": [2, 2, 1],
        "half_baths": [0, 1, 0],
        "total_rooms": [7, 8, 6],
        "year_built": [1950, 1975, 1920],
        "age_at_sale": [74, 49, 104],
        "condition": ["Good", "Average", "Fair"],
        "grade": ["Good", "Average", "Below Average"],
        "exterior_finish": ["Brick", "Vinyl/Aluminum", "Frame"],
        "stories": [2, 1, 1.5],
        "has_basement_garage": [1, 0, 1],
        "fireplaces": [1, 0, 0]
    })
    pre = build_preprocessor()
    pre.fit(df)
    out = pre.transform(df)
    assert out.shape[0] == 3
    assert out.shape[1] > 10
    print("preprocessor fit/transform OK")

def test_clean_clip():
    df = pd.DataFrame({"living_area_sqft":[100, 9000], "bedrooms":[10, -1], "year_built":[1800, 2030]})
    cleaned = clean_input_df(df)
    assert cleaned["living_area_sqft"].min() >= 300
    assert cleaned["living_area_sqft"].max() <= 8000
    print("clip OK")

if __name__ == "__main__":
    test_preprocessor_fit_transform()
    test_clean_clip()
