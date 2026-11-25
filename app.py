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
    initial_sidebar_state="collapsed"
)

# Compatibility helpers for Streamlit <1.36 (use_column_width) and >=1.36 (use_container_width)
def _st_image(path, caption=None):
    try:
        st.image(path, caption=caption, use_container_width=True)
    except TypeError:
        st.image(path, caption=caption, use_column_width=True)

def _st_dataframe(df, **kwargs):
    # kwargs may include hide_index etc.
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
    # handle use_container_width vs older API
    if "use_container_width" in kwargs:
        try:
            return st.button(label, **kwargs)
        except TypeError:
            kwargs.pop("use_container_width")
            # fallback: try use_column_width or just default
            try:
                return st.button(label, use_column_width=True, **kwargs)
            except TypeError:
                return st.button(label, **kwargs)
    return st.button(label, **kwargs)

# --- Styles ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; letter-spacing: -0.02em; }
.block-container { padding-top: 1.5rem; max-width: 1180px; }
div[data-testid="stMetricValue"] { font-weight: 700; }

/* Cards */
.card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 16px;
  padding: 20px 22px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.hero {
  background: linear-gradient(135deg, #2C4A5A 0%, #3A6B7D 100%);
  color: white;
  border-radius: 18px;
  padding: 28px 30px;
}
.hero h1 { color: white; margin: 0 0 6px 0; font-size: 2.0rem; }
.hero p { color: #E6EEF2; margin: 0; line-height: 1.5; }
.badge {
  display: inline-block;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.22);
  color: #FFF;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  margin-right: 6px;
}
.kpi {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  padding: 16px;
  text-align: left;
}
.kpi .label { font-size: 0.78rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi .value { font-size: 1.35rem; font-weight: 700; color: #1A2026; margin-top: 4px; }
.kpi .note { font-size: 0.82rem; color: #6B7280; margin-top: 2px; }
.interval-bar {
  height: 10px; background: #E5E7EB; border-radius: 999px; position: relative; margin: 14px 0;
}
.interval-fill {
  position: absolute; height: 10px; background: #2C4A5A; border-radius: 999px; top: 0;
}
.interval-dot {
  position: absolute; width: 14px; height: 14px; background: #B87333; border: 2px solid white; border-radius: 50%; top: -2px; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.small-muted { color: #6B7280; font-size: 0.86rem; }
hr { border: none; border-top: 1px solid #E5E7EB; margin: 1.2rem 0; }
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
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
    <div style="max-width: 760px;">
      <div style="margin-bottom:10px;">
        <span class="badge">Independent portfolio case study</span>
        <span class="badge">Allegheny County, PA • 2019–2026</span>
        <span class="badge">Public data (CC0)</span>
      </div>
      <h1>Home Value Range Estimator</h1>
      <p>Enter a home’s key features — living area, lot size, year built, condition and ZIP code — and get an estimated <b>sale price range</b> for the Pittsburgh area, with a clear explanation of what drove the estimate. Built as a freelance-scale pilot: one county, recorded sales only, no national claims.</p>
    </div>
    <div style="text-align:right; min-width: 180px;">
      <div style="background:white; color:#2C4A5A; border-radius:12px; padding:12px 16px; display:inline-block;">
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.07em; opacity:0.7;">Median holdout price</div>
        <div style="font-size:1.25rem; font-weight:700;">$308,264</div>
        <div style="font-size:0.78rem; opacity:0.7;">n = 1,160 recent sales</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# --- Top KPIs ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi"><div class="label">Market</div><div class="value">{MARKET_NAME}</div><div class="note">{SALES_COVERAGE}</div></div>""", unsafe_allow_html=True)
with c2:
    ridge_rmse = metrics.get("ridge", {}).get("rmse", 63174)
    st.markdown(f"""<div class="kpi"><div class="label">Holdout RMSE (XGBoost)</div><div class="value">${metrics.get('xgboost',{}).get('rmse',63280):,.0f}</div><div class="note">MAE ${metrics.get('xgboost',{}).get('mae',46272):,.0f} • R² {metrics.get('xgboost',{}).get('r2',0.82):.3f}</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi"><div class="label">80% interval (empirical)</div><div class="value">± $71,150</div><div class="note">Coverage 80% on holdout • method in docs</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi"><div class="label">Training sample</div><div class="value">5,800 homes</div><div class="note">4,640 train / 1,160 temporal holdout</div></div>""", unsafe_allow_html=True)

st.write("")

# --- Main: Form + Result ---
left, right = st.columns([1.05, 1.15], gap="large")

# Presets
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

with left:
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

    st.markdown('<div class="small-muted" style="margin-top:10px;">Built by <b>fotroslab</b> • Data as of 2026-10-30 • Not an appraisal. See Limitations below.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estimated Sale Price Range")
    st.caption("Central estimate is the model’s expected price; low/high form an 80% empirical interval calibrated on the temporal holdout (coverage 80%). Shown in USD.")

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

    # Auto-estimate on first load if no button yet
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
        # Show placeholder and trigger initial estimate
        try:
            result = predict_range(input_dict)
            st.session_state.last_result = result
        except Exception:
            result = None

    if result:
        central = result["central"]
        low = result["low"]
        high = result["high"]
        # Format USD
        def fmt(v): return f"${v:,.0f}"
        st.markdown(f"""
        <div style="display:flex; gap:12px; margin: 8px 0 6px 0;">
          <div style="flex:1; background:#F8F6F1; border:1px solid #E5E7EB; border-radius:12px; padding:14px; text-align:center;">
            <div style="font-size:0.75rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.06em;">Low (10th)</div>
            <div style="font-size:1.15rem; font-weight:700; color:#1A2026;">{fmt(low)}</div>
          </div>
          <div style="flex:1.2; background:#2C4A5A; border-radius:12px; padding:14px; text-align:center; color:white;">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; opacity:0.85;">Central estimate</div>
            <div style="font-size:1.35rem; font-weight:800;">{fmt(central)}</div>
            <div style="font-size:0.78rem; opacity:0.85;">{result['model_name']}</div>
          </div>
          <div style="flex:1; background:#F8F6F1; border:1px solid #E5E7EB; border-radius:12px; padding:14px; text-align:center;">
            <div style="font-size:0.75rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.06em;">High (90th)</div>
            <div style="font-size:1.15rem; font-weight:700; color:#1A2026;">{fmt(high)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Interval bar visual
        # Map to relative scale: low to high
        span = high - low
        if span > 0:
            # central position as pct
            pct_central = (central - low) / span * 100
            st.markdown(f"""
            <div class="interval-bar">
              <div class="interval-fill" style="left:0%; width:100%;"></div>
              <div class="interval-dot" style="left: calc({pct_central:.1f}% - 7px);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#6B7280;">
              <span>{fmt(low)}</span><span>Central</span><span>{fmt(high)}</span>
            </div>
            """, unsafe_allow_html=True)

        st.caption(result.get("coverage_note",""))
        # Coverage details
        st.info(f"**Coverage:** {DATA_AS_OF} snapshot • {SALES_COVERAGE} • Sample n=5,800 (4,640 train / 1,160 holdout). The interval’s observed coverage on holdout is 80% — it is approximate, not a guarantee for any single home.", icon="ℹ️")

        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(w, icon="⚠️")

        # Why this estimate? — SHAP
        st.markdown("#### Why this estimate?")
        st.caption("How the model’s top drivers shifted this home vs. a typical Allegheny home. This explains the *model*, not a causal appraisal.")
        try:
            import shap
            model, _ = load_artifacts()
            # Prepare single-row SHAP
            df_one = pd.DataFrame([input_dict])
            # Need to use the same feature order as training
            preproc = model.named_steps["preprocess"]
            # For SHAP we need background; use small sample from data
            sample_path = PROJECT_ROOT / "data" / "sample_properties.csv"
            if sample_path.exists():
                bg = pd.read_csv(sample_path)[["zip_code","municipality","living_area_sqft","lot_area_sqft","bedrooms","full_baths","half_baths","total_rooms","year_built","age_at_sale","condition","grade","exterior_finish","stories","has_basement_garage","fireplaces"]].sample(200, random_state=0)
                # Transform
                X_bg = preproc.transform(bg)
                X_one = preproc.transform(df_one)
                feature_names = preproc.get_feature_names_out().tolist()
                # Choose explainer type
                m = model.named_steps["model"]
                if hasattr(m, "get_booster"):
                    explainer = shap.TreeExplainer(m)
                    sv = explainer.shap_values(X_one)
                else:
                    explainer = shap.LinearExplainer(m, X_bg)
                    sv = explainer.shap_values(X_one)
                # sv shape (1, n_features)
                vals = sv[0] if len(sv.shape)==2 else sv
                # Map to feature names
                df_shap = pd.DataFrame({"feature": feature_names, "shap": vals})
                df_shap["abs"] = df_shap["shap"].abs()
                df_shap = df_shap.sort_values("abs", ascending=False).head(8)
                # Pretty names
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
                # Collapse one-hot features: group by pretty prefix
                # For display, just show top 6 raw features with human labels
                # Simplify label: take first part before underscore or full
                def humanize(f):
                    # try to map base
                    for k,v in pretty.items():
                        if f.startswith(k):
                            # append category value if one-hot
                            suffix = f[len(k):].lstrip("_")
                            return f"{v} {suffix}" if suffix else v
                    return f
                df_shap["label"] = df_shap["feature"].apply(humanize)
                # Color by sign
                import plotly.graph_objects as go
                colors = ["#2C4A5A" if x>0 else "#B87333" for x in df_shap["shap"]]
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
                st.caption("Positive (teal) pushes the estimate up vs. the baseline; negative (copper) pushes it down. Units reflect the model’s internal scale (USD).")
            else:
                raise FileNotFoundError("no bg")
        except Exception as e:
            # Fallback: show global importance
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
            st.caption("RMSE/MAE in USD on holdout sale price. R² is variance explained, not ‘accuracy %’. Lower RMSE/MAE is better.")
        band_path = PROJECT_ROOT / "models" / "band_metrics.csv"
        if band_path.exists():
            band_df = pd.read_csv(band_path)
            st.markdown("**Error by price band**")
            _st_dataframe(band_df.style.format({"rmse":"${:,.0f}","mae":"${:,.0f}","median_pct_err":"{:.1%}"}), hide_index=True)
            st.caption("High-end bands (700k+) have n=35 — do not generalise. Reported with n for transparency.")
    with colB:
        shap_path = PROJECT_ROOT / "models" / "shap_bar.png"
        if shap_path.exists():
            _st_image(str(shap_path), caption="Global feature impact (mean |SHAP|). Living area dominates, followed by grade, ZIP, and condition.")
        resid_path = PROJECT_ROOT / "models" / "residual_plot.png"
        if resid_path.exists():
            _st_image(str(resid_path), caption="Residuals vs. predicted and distribution. Recent holdout shows slight under-prediction due to market appreciation.")

# --- Method & Limitations ---
st.write("")
st.markdown("## How it works")
a,b,c = st.columns(3)
with a:
    st.markdown("""
<div class="card" style="height:100%;">
<h4>1. Data</h4>
<p class="small-muted">Allegheny County Property Assessments via WPRDC (CC0). 147k valid sales (2013–present); pilot uses 5,800 filtered residential sales 2019–2026 with snapshot features as of 2026-10-30. Full provenance in <code>data/README.md</code>.</p>
</div>
""", unsafe_allow_html=True)
with b:
    st.markdown("""
<div class="card" style="height:100%;">
<h4>2. Model</h4>
<p class="small-muted">Leak-safe pipeline (impute → one-hot/standardize → model). Baseline median vs. Ridge vs. XGBoost on same temporal split. No assessed value as feature; no address/owner leakage.</p>
</div>
""", unsafe_allow_html=True)
with c:
    st.markdown("""
<div class="card" style="height:100%;">
<h4>3. Interval</h4>
<p class="small-muted">80% empirical interval from holdout residuals (80th %ile |residual| = $71,150). Observed coverage 80% on holdout; shown as low/central/high — approximate, not a legal guarantee.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Limitations & responsible use")
st.markdown("""
<div class="card">
<ul style="margin:0; padding-left:18px; line-height:1.7;" class="small-muted">
<li><b>One county only.</b> Allegheny County, PA. ZIPs outside the county list are flagged.</li>
<li><b>Historical snapshot.</b> Features are 2026-10-30 state applied to older sales — recent sales are most reliable; retrain monthly in production.</li>
<li><b>Not an appraisal.</b> Demo for portfolio purposes; do not use for lending, bidding, or tax decisions. No photo/deposit/rent features.</li>
<li><b>High-end sparse.</b> Homes above $700k have n=35 in holdout — interval is wider and less reliable there.</li>
<li><b>SHAP explains the model, not causality.</b> “Brick adds $X” means the model associates it, not that you’ll gain $X by re-siding.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("""
<div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:16px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;">
  <div>
    <div style="font-weight:600;">Built by fotroslab</div>
    <div class="small-muted">Independent portfolio case study using public data • No client affiliation • Metrics from this repo’s execution</div>
  </div>
  <div class="small-muted" style="text-align:right;">
    <div>Source: WPRDC (CC0) • Snapshot 2026-10-30</div>
    <div>Code: GitHub (add link after publish) • Docs: <code>docs/</code></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Footer warning
st.caption("No Persian text, currency, or terminology is used in this app. All figures are USD; areas in sq ft; locations are U.S. ZIP codes.")
