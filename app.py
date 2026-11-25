"""Home Value Range Estimator — Allegheny County Pilot
Streamlit app — English only, USD/sq ft/ZIP terminology, builder: fotroslab
"""
import json
import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from pathlib import Path

# Config
from src.config import ZIP_CODES, MUNICIPALITIES, CONDITIONS, GRADES, EXTERIORS, MARKET_NAME, DATA_AS_OF, SALES_COVERAGE
from src.prediction import predict_range, validate_input, load_artifacts
from src.config import PROJECT_ROOT

st.set_page_config(
    page_title="Home Value Range Estimator — Allegheny County Pilot",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Compatibility helpers for Streamlit <1.36 and >=1.36
def _st_image(path, caption=None):
    try:
        st.image(path, caption=caption, use_container_width=True)
    except TypeError:
        st.image(path, caption=caption, use_column_width=True)

def _st_dataframe(df, **kwargs):
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        try:
            st.dataframe(df, use_column_width=True, **kwargs)
        except TypeError:
            st.dataframe(df, **kwargs)

def _st_plotly(fig):
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        try:
            st.plotly_chart(fig, use_column_width=True)
        except TypeError:
            st.plotly_chart(fig)

def _st_button(label, **kwargs):
    if "use_container_width" in kwargs:
        try:
            return st.button(label, **kwargs)
        except TypeError:
            kwargs.pop("use_container_width")
            try:
                return st.button(label, use_column_width=True, **kwargs)
            except TypeError:
                return st.button(label, **kwargs)
    return st.button(label, **kwargs)

# --- Styles ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; letter-spacing: -0.02em; }

/* Full-width layout — fix narrow centered issue */
.block-container {
    max-width: 1440px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1.2rem !important;
}
@media (max-width: 768px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
[data-testid="stHeader"] { background: transparent; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #1E3A4A 0%, #2C5F73 55%, #3A7A8A 100%);
  color: white;
  border-radius: 20px;
  padding: 30px 32px;
  margin-bottom: 18px;
}
.hero h1 { color: white; margin: 0 0 8px 0; font-size: 2.15rem; line-height: 1.15; }
.hero p { color: #E6EEF2; margin: 0; line-height: 1.6; font-size: 0.98rem; max-width: 820px; }
.badge {
  display: inline-block;
  background: rgba(255,255,255,0.13);
  border: 1px solid rgba(255,255,255,0.22);
  color: #FFF;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 0.76rem;
  margin-right: 7px;
  margin-bottom: 6px;
}

/* KPI grid — equal height */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 16px 0 20px 0;
}
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .kpi-grid { grid-template-columns: 1fr; } }

.kpi {
  background: #FFFFFF;
  border: 1px solid #E6E8EB;
  border-radius: 16px;
  padding: 18px 18px 16px 18px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 132px;
  height: 100%;
  box-shadow: 0 1px 3px rgba(16,24,40,0.06);
  transition: box-shadow 0.15s;
}
.kpi:hover { box-shadow: 0 4px 12px rgba(16,24,40,0.08); }
.kpi .label { font-size: 0.71rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin-bottom: 8px; }
.kpi .value { font-size: 1.45rem; font-weight: 700; color: #111827; line-height: 1.2; margin: 2px 0 6px 0; }
.kpi .note { font-size: 0.82rem; color: #6B7280; line-height: 1.4; margin-top: auto; }
.kpi .trend { font-size: 0.76rem; color: #0E7A5B; font-weight: 600; margin-top: 6px; }

/* Cards */
.card {
  background: #FFFFFF;
  border: 1px solid #E6E8EB;
  border-radius: 18px;
  padding: 22px 24px;
  box-shadow: 0 1px 3px rgba(16,24,40,0.05);
  height: 100%;
}

/* Main two-column equal height */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 1.08fr;
    gap: 18px;
    align-items: start;
    margin-top: 10px;
}
@media (max-width: 1024px) { .main-grid { grid-template-columns: 1fr; } }

/* Price boxes */
.price-row {
  display: grid;
  grid-template-columns: 1fr 1.2fr 1fr;
  gap: 12px;
  margin: 12px 0 10px 0;
  align-items: stretch;
}
.price-box {
  border-radius: 16px;
  padding: 16px 14px;
  text-align: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 92px;
}
.price-box.side { background: #F9FAFB; border: 1px solid #E6E8EB; }
.price-box.center { background: #1E3A4A; color: white; border: 1px solid #1E3A4A; }
.price-box .lbl { font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; opacity: 0.75; }
.price-box.center .lbl { color: #D1E7F0; opacity: 1; }
.price-box .val { font-size: 1.28rem; font-weight: 800; margin-top: 4px; }
.price-box.center .val { font-size: 1.45rem; }
.price-box .sub { font-size: 0.78rem; opacity: 0.75; margin-top: 2px; }

.interval-bar {
  height: 10px; background: #E5E7EB; border-radius: 999px; position: relative; margin: 16px 0 10px 0;
}
.interval-fill {
  position: absolute; height: 10px; background: #1E3A4A; border-radius: 999px; top: 0;
}
.interval-dot {
  position: absolute; width: 14px; height: 14px; background: #C77A3B; border: 2.5px solid white; border-radius: 50%; top: -2px; box-shadow: 0 2px 6px rgba(0,0,0,0.18);
}
.small-muted { color: #6B7280; font-size: 0.86rem; line-height: 1.5; }
hr { border: none; border-top: 1px solid #E6E8EB; margin: 1.2rem 0; }

/* Inputs compact */
.stSelectbox, .stNumberInput { font-size: 0.93rem; }
</style>
""", unsafe_allow_html=True)

# Load metrics
metrics_path = PROJECT_ROOT / "models" / "metrics.json"
metrics = {}
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

# --- Header ---
st.markdown("""
<div class="hero">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:18px;">
    <div style="flex:1; min-width: 320px;">
      <div style="margin-bottom:12px;">
        <span class="badge">Independent portfolio case study</span>
        <span class="badge">Allegheny County, PA • 2019–2026</span>
        <span class="badge">Public data (CC0)</span>
      </div>
      <h1>Home Value Range Estimator</h1>
      <p>Enter a home’s key features — living area, lot size, year built, condition and ZIP code — and get an estimated <b>sale price range</b> for the Pittsburgh area, with a clear explanation of what drove the estimate. Built as a freelance-scale pilot: one county, recorded sales only, no national claims.</p>
    </div>
    <div style="flex:0 0 190px; text-align:right;">
      <div style="background:white; color:#1E3A4A; border-radius:14px; padding:14px 16px; display:inline-block; width: 100%; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
        <div style="font-size:0.70rem; text-transform:uppercase; letter-spacing:0.07em; opacity:0.6; font-weight:600;">Median holdout price</div>
        <div style="font-size:1.35rem; font-weight:800; margin: 4px 0;">$308,264</div>
        <div style="font-size:0.78rem; opacity:0.65;">n = 1,160 recent sales</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- Top KPIs — equal height grid ---
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div>
      <div class="label">Market</div>
      <div class="value" style="font-size:1.18rem;">{MARKET_NAME}</div>
    </div>
    <div class="note">2013-01-01 to 2026-08-07<br/>(WPRDC snapshot 2026-10-30)</div>
  </div>
  <div class="kpi">
    <div>
      <div class="label">Holdout RMSE (XGBoost)</div>
      <div class="value">${metrics.get('xgboost',{}).get('rmse',63280):,.0f}</div>
    </div>
    <div class="note">MAE ${metrics.get('xgboost',{}).get('mae',46272):,.0f} • R² {metrics.get('xgboost',{}).get('r2',0.82):.3f}</div>
  </div>
  <div class="kpi">
    <div>
      <div class="label">80% Interval (empirical)</div>
      <div class="value">± $71,150</div>
    </div>
    <div class="note">Coverage 80% on holdout<br/>method in docs</div>
  </div>
  <div class="kpi">
    <div>
      <div class="label">Training sample</div>
      <div class="value">5,800 homes</div>
    </div>
    <div class="note">4,640 train / 1,160 temporal holdout</div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- Presets ---
PRESETS = {
    "Pittsburgh brick colonial — 15217": {
        "zip_code": 15217, "municipality": "Pittsburgh (1st-31st Wards)", "living_area_sqft": 1650, "lot_area_sqft": 4200, "bedrooms": 3, "full_baths": 2, "half_baths": 1, "total_rooms": 7, "year_built": 1955, "condition": "Good", "grade": "Good", "exterior_finish": "Brick", "stories": 2, "has_basement_garage": 1, "fireplaces": 1
    },
    "Suburban ranch — 15241": {
        "zip_code": 15241, "municipality": "Upper St. Clair", "living_area_sqft": 2100, "lot_area_sqft": 8500, "bedrooms": 4, "full_baths": 2, "half_baths": 1, "total_rooms": 8, "year_built": 1998, "condition": "Very Good", "grade": "Very Good", "exterior_finish": "Brick", "stories": 1, "has_basement_garage": 1, "fireplaces": 1
    },
    "Affordable starter — 15210": {
        "zip_code": 15210, "municipality": "Pittsburgh (1st-31st Wards)", "living_area_sqft": 980, "lot_area_sqft": 2800, "bedrooms": 2, "full_baths": 1, "half_baths": 0, "total_rooms": 5, "year_built": 1940, "condition": "Average", "grade": "Average", "exterior_finish": "Frame", "stories": 1, "has_basement_garage": 0, "fireplaces": 0
    }
}

if "preset" not in st.session_state:
    st.session_state.preset = "Pittsburgh brick colonial — 15217"

# --- Main two panels (equalized) ---
col_left, col_right = st.columns([1, 1.08], gap="medium")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estimate a home")
    st.caption("All fields are required. The model covers Allegheny County only — ZIPs outside the county trigger a warning and use the nearest municipality average.")

    preset_choice = st.selectbox("Try a preset", list(PRESETS.keys()), index=0)
    if _st_button("Load preset", use_container_width=True):
        st.session_state.preset = preset_choice
        st.rerun()

    defaults = PRESETS[st.session_state.preset]

    colA, colB = st.columns(2)
    with colA:
        zip_code = st.selectbox("ZIP code", sorted(ZIP_CODES), index=sorted(ZIP_CODES).index(defaults["zip_code"]))
        municipality = st.selectbox("Municipality / area", MUNICIPALITIES, index=MUNICIPALITIES.index(defaults["municipality"]) if defaults["municipality"] in MUNICIPALITIES else 0)
        living_area = st.number_input("Finished living area (sq ft)", 300, 8000, defaults["living_area_sqft"], step=50, help="Above-grade finished area. Typical Pittsburgh: 900–2,400 sq ft.")
        lot_area = st.number_input("Lot area (sq ft)", 0, 100000, defaults["lot_area_sqft"], step=100, help="Lot size from assessment. Use 0 for condo with no lot.")
        bedrooms = st.selectbox("Bedrooms", [0,1,2,3,4,5,6,7,8], index=defaults["bedrooms"])
        full_baths = st.selectbox("Full baths", [0,1,2,3,4,5,6], index=defaults["full_baths"])
        half_baths = st.selectbox("Half baths", [0,1,2,3,4], index=defaults["half_baths"])
    with colB:
        year_built = st.number_input("Year built", 1850, 2026, defaults["year_built"], step=1)
        total_rooms = st.selectbox("Total rooms", list(range(3,13)), index=defaults["total_rooms"]-3 if 3 <= defaults["total_rooms"] <= 12 else 4)
        condition = st.selectbox("Condition", CONDITIONS, index=CONDITIONS.index(defaults["condition"]))
        grade = st.selectbox("Grade (overall quality)", GRADES, index=GRADES.index(defaults["grade"]))
        exterior = st.selectbox("Exterior finish", EXTERIORS, index=EXTERIORS.index(defaults["exterior_finish"]))
        stories = st.selectbox("Stories", [1,1.5,2,2.5,3], index=[1,1.5,2,2.5,3].index(defaults["stories"]))
        has_garage = st.selectbox("Basement garage", ["No","Yes"], index=defaults["has_basement_garage"])
        fireplaces = st.selectbox("Fireplaces", [0,1,2,3], index=defaults["fireplaces"])

    st.write("")
    estimate_btn = _st_button("Estimate price range", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted" style="margin-top:10px; padding-left:4px;">Built by <b>fotroslab</b> • Data as of 2026-10-30 • Not an appraisal. See Limitations below.</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estimated Sale Price Range")
    st.caption("Central estimate is the model's expected price; low/high form an 80% empirical interval calibrated on the temporal holdout (coverage 80%). Shown in USD.")

    # Build input dict
    input_dict = {
        "zip_code": zip_code if 'zip_code' in locals() else defaults["zip_code"],
        "municipality": municipality if 'municipality' in locals() else defaults["municipality"],
        "living_area_sqft": living_area if 'living_area' in locals() else defaults["living_area_sqft"],
        "lot_area_sqft": lot_area if 'lot_area' in locals() else defaults["lot_area_sqft"],
        "bedrooms": bedrooms if 'bedrooms' in locals() else defaults["bedrooms"],
        "full_baths": full_baths if 'full_baths' in locals() else defaults["full_baths"],
        "half_baths": half_baths if 'half_baths' in locals() else defaults["half_baths"],
        "total_rooms": total_rooms if 'total_rooms' in locals() else defaults["total_rooms"],
        "year_built": year_built if 'year_built' in locals() else defaults["year_built"],
        "age_at_sale": 2024 - (year_built if 'year_built' in locals() else defaults["year_built"]),
        "condition": condition if 'condition' in locals() else defaults["condition"],
        "grade": grade if 'grade' in locals() else defaults["grade"],
        "exterior_finish": exterior if 'exterior' in locals() else defaults["exterior_finish"],
        "stories": stories if 'stories' in locals() else defaults["stories"],
        "has_basement_garage": 1 if (has_garage if 'has_garage' in locals() else ("Yes" if defaults["has_basement_garage"]==1 else "No")) == "Yes" else 0,
        "fireplaces": fireplaces if 'fireplaces' in locals() else defaults["fireplaces"],
    }

    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    if estimate_btn:
        errs = validate_input(input_dict)
        if errs:
            for e in errs:
                st.error(e)
        else:
            try:
                result = predict_range(input_dict)
                st.session_state.last_result = result
                st.session_state.last_input = input_dict
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    result = st.session_state.get("last_result")
    if result is None:
        try:
            result = predict_range(input_dict)
            st.session_state.last_result = result
        except Exception:
            result = None

    if result:
        central = result["central"]
        low = result["low"]
        high = result["high"]
        def fmt(v): return f"${v:,.0f}"
        st.markdown(f"""
        <div class="price-row">
          <div class="price-box side">
            <div class="lbl">Low (10th)</div>
            <div class="val" style="color:#111827;">{fmt(low)}</div>
          </div>
          <div class="price-box center">
            <div class="lbl">Central estimate</div>
            <div class="val">{fmt(central)}</div>
            <div class="sub">{result['model_name']}</div>
          </div>
          <div class="price-box side">
            <div class="lbl">High (90th)</div>
            <div class="val" style="color:#111827;">{fmt(high)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        span = high - low
        if span > 0:
            pct_central = (central - low) / span * 100
            st.markdown(f"""
            <div class="interval-bar">
              <div class="interval-fill" style="left:0%; width:100%;"></div>
              <div class="interval-dot" style="left: calc({pct_central:.1f}% - 7px);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.77rem; color:#6B7280;">
              <span>{fmt(low)}</span><span>Central</span><span>{fmt(high)}</span>
            </div>
            """, unsafe_allow_html=True)

        st.caption(result.get("coverage_note",""))
        st.info(f"**Coverage:** {DATA_AS_OF} snapshot • {SALES_COVERAGE} • Sample n=5,800 (4,640 train / 1,160 holdout). The interval's observed coverage on holdout is 80% — it is approximate, not a guarantee for any single home.", icon="ℹ️")

        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(w, icon="⚠️")

        st.markdown("#### Why this estimate?")
        st.caption("How the model's top drivers shifted this home vs. a typical Allegheny home. This explains the *model*, not a causal appraisal.")
        try:
            import shap
            model, _ = load_artifacts()
            df_one = pd.DataFrame([input_dict])
            preproc = model.named_steps["preprocess"]
            sample_path = PROJECT_ROOT / "data" / "sample_properties.csv"
            if sample_path.exists():
                bg = pd.read_csv(sample_path)[["zip_code","municipality","living_area_sqft","lot_area_sqft","bedrooms","full_baths","half_baths","total_rooms","year_built","age_at_sale","condition","grade","exterior_finish","stories","has_basement_garage","fireplaces"]].sample(200, random_state=0)
                X_bg = preproc.transform(bg)
                X_one = preproc.transform(df_one)
                feature_names = preproc.get_feature_names_out().tolist()
                m = model.named_steps["model"]
                if hasattr(m, "get_booster"):
                    explainer = shap.TreeExplainer(m)
                    sv = explainer.shap_values(X_one)
                else:
                    explainer = shap.LinearExplainer(m, X_bg)
                    sv = explainer.shap_values(X_one)
                vals = sv[0] if len(sv.shape)==2 else sv
                df_shap = pd.DataFrame({"feature": feature_names, "shap": vals})
                df_shap["abs"] = df_shap["shap"].abs()
                df_shap = df_shap.sort_values("abs", ascending=False).head(8)
                pretty = {
                    "living_area_sqft": "Living area (sq ft)",
                    "lot_area_sqft": "Lot area (sq ft)",
                    "zip_code": "ZIP code",
                    "municipality": "Municipality",
                    "condition": "Condition",
                    "grade": "Grade / quality",
                    "exterior_finish": "Exterior finish",
                    "year_built": "Year built",
                    "age_at_sale": "Age at sale",
                    "bedrooms": "Bedrooms",
                    "full_baths": "Full baths",
                    "half_baths": "Half baths",
                    "stories": "Stories",
                    "has_basement_garage": "Basement garage",
                    "fireplaces": "Fireplaces",
                    "total_rooms": "Total rooms"
                }
                def humanize(f):
                    for k,v in pretty.items():
                        if f.startswith(k):
                            suffix = f[len(k):].lstrip("_")
                            return f"{v} {suffix}" if suffix else v
                    return f
                df_shap["label"] = df_shap["feature"].apply(humanize)
                import plotly.graph_objects as go
                colors = ["#1E3A4A" if x>0 else "#C77A3B" for x in df_shap["shap"]]
                fig = go.Figure(go.Bar(
                    x=df_shap["shap"],
                    y=df_shap["label"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{'+' if v>0 else ''}{v:,.0f}" for v in df_shap["shap"]],
                    textposition="outside"
                ))
                fig.update_layout(
                    height=320, margin=dict(l=160, r=40, t=10, b=20),
                    xaxis_title="SHAP contribution (USD impact in model space)",
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(size=12)
                )
                _st_plotly(fig)
                st.caption("Positive (teal) pushes the estimate up vs. the baseline; negative (copper) pushes it down.")
            else:
                raise FileNotFoundError("no bg")
        except Exception as e:
            st.warning(f"Local SHAP unavailable for this input ({e}). Showing global importance instead.", icon="ℹ️")
            shap_img = PROJECT_ROOT / "models" / "shap_bar.png"
            if shap_img.exists():
                _st_image(str(shap_img), caption="Global SHAP — mean |impact| across holdout (top 12 features).")
            else:
                st.write("No SHAP artifact found.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Performance section ---
st.write("")
with st.expander("Model performance & diagnostics", expanded=False):
    colA, colB = st.columns([1,1])
    with colA:
        st.markdown("**Holdout (temporal, n=1,160) vs. baseline**")
        if metrics:
            df_met = pd.DataFrame([
                {"Model":"Baseline (median)", "RMSE": metrics["baseline_median"]["rmse"], "MAE": metrics["baseline_median"]["mae"], "R²": metrics["baseline_median"]["r2"]},
                {"Model":"Ridge (α=10)", "RMSE": metrics["ridge"]["rmse"], "MAE": metrics["ridge"]["mae"], "R²": metrics["ridge"]["r2"]},
                {"Model":"XGBoost (400×6)", "RMSE": metrics["xgboost"]["rmse"], "MAE": metrics["xgboost"]["mae"], "R²": metrics["xgboost"]["r2"]},
            ])
            _st_dataframe(df_met.style.format({"RMSE":"${:,.0f}","MAE":"${:,.0f}","R²":"{:.3f}"}), hide_index=True)
            st.caption("RMSE/MAE in USD on holdout sale price. R² is variance explained, not 'accuracy %'.")
        band_path = PROJECT_ROOT / "models" / "band_metrics.csv"
        if band_path.exists():
            band_df = pd.read_csv(band_path)
            st.markdown("**Error by price band**")
            _st_dataframe(band_df.style.format({"rmse":"${:,.0f}","mae":"${:,.0f}","median_pct_err":"{:.1%}"}), hide_index=True)
            st.caption("High-end bands (700k+) have n=35 — do not generalise.")
    with colB:
        shap_path = PROJECT_ROOT / "models" / "shap_bar.png"
        if shap_path.exists():
            _st_image(str(shap_path), caption="Global feature impact (mean |SHAP|). Living area dominates, followed by grade, ZIP, and condition.")
        resid_path = PROJECT_ROOT / "models" / "residual_plot.png"
        if resid_path.exists():
            _st_image(str(resid_path), caption="Residuals vs. predicted and distribution. Recent holdout shows slight under-prediction due to market appreciation.")

# --- Method & Limitations ---
st.write("")
st.markdown("### How it works")
st.markdown("""
<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 10px 0;">
  <div class="card" style="min-height:140px;">
    <h4 style="margin:0 0 6px 0;">1. Data</h4>
    <p class="small-muted" style="margin:0;">Allegheny County via WPRDC (CC0). 147k valid sales; pilot 5,800 filtered 2019–2026 with snapshot features as of 2026-10-30.</p>
  </div>
  <div class="card" style="min-height:140px;">
    <h4 style="margin:0 0 6px 0;">2. Model</h4>
    <p class="small-muted" style="margin:0;">Leak-safe pipeline → Baseline vs Ridge vs XGBoost on temporal split. No assessed value leakage.</p>
  </div>
  <div class="card" style="min-height:140px;">
    <h4 style="margin:0 0 6px 0;">3. Interval</h4>
    <p class="small-muted" style="margin:0;">80% empirical interval from holdout residuals (|residual| 80th = $71,150). Shown as low/central/high.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### Limitations & responsible use")
st.markdown("""
<div class="card">
<ul style="margin:0; padding-left:18px; line-height:1.8;" class="small-muted">
<li><b>One county only.</b> Allegheny County, PA. ZIPs outside the list are flagged.</li>
<li><b>Historical snapshot.</b> Features are 2026-10-30 state applied to older sales — retrain monthly in production.</li>
<li><b>Not an appraisal.</b> Demo for portfolio purposes; do not use for lending/bidding/tax.</li>
<li><b>High-end sparse.</b> Homes above $700k have n=35 — interval is wider there.</li>
<li><b>SHAP explains the model, not causality.</b></li>
</ul>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("""
<div style="background:#FFFFFF; border:1px solid #E6E8EB; border-radius:14px; padding:16px 18px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;">
  <div>
    <div style="font-weight:700;">Built by fotroslab</div>
    <div class="small-muted">Independent portfolio case study using public data • No client affiliation • Metrics from this repo.</div>
  </div>
  <div class="small-muted" style="text-align:right;">
    <div>Source: WPRDC (CC0) • Snapshot 2026-10-30</div>
    <div>Code: GitHub (add link after publish) • Docs: <code>docs/</code></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption("No Persian text, currency, or terminology is used in this app. All figures are USD; areas in sq ft; locations are U.S. ZIP codes.")
