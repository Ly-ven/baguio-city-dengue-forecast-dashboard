import os
import json
import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Baguio City Dengue Forecast Dashboard",
    layout="wide"
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive web-based dashboard for dengue prediction and visualization")

# ---------------------------
# Paths
# ---------------------------
ARTIFACTS_DIR = Path("artifacts")
DEFAULT_FEATURE_COLS = [
    "rainfall", "relative_humidity", "temp_mid",
    "cases_lag_1", "cases_lag_2", "cases_lag_3",
    "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
    "relative_humidity_lag_1", "relative_humidity_lag_2", "relative_humidity_lag_3",
    "temp_mid_lag_1", "temp_mid_lag_2", "temp_mid_lag_3",
    "cases_roll3_mean", "cases_roll3_max",
    "month_sin", "month_cos"
]

# ---------------------------
# Helper functions
# ---------------------------
def safe_read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None

def safe_read_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def safe_load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_data
def load_artifacts():
    monthly = safe_read_csv(ARTIFACTS_DIR / "monthly_modeling_dataset.csv")
    model_comparison = safe_read_csv(ARTIFACTS_DIR / "model_comparison.csv")
    feature_importance = safe_read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    feature_sensitivity = safe_read_csv(ARTIFACTS_DIR / "feature_sensitivity.csv")
    forecast = safe_read_csv(ARTIFACTS_DIR / "forecast_5yr.csv")
    barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "barangay_monthly.csv")
    top_barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "top_barangay_monthly.csv")
    top3_barangays_yearly = safe_read_csv(ARTIFACTS_DIR / "top3_barangays_yearly.csv")
    top3_barangays_overall = safe_read_csv(ARTIFACTS_DIR / "top3_barangays_overall.csv")
    test_predictions = safe_read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    confusion_matrix_detail = safe_read_csv(ARTIFACTS_DIR / "confusion_matrix_detail.csv")
    climate_case_correlation = safe_read_csv(ARTIFACTS_DIR / "climate_case_correlation.csv")
    month_profile = safe_read_csv(ARTIFACTS_DIR / "month_profile.csv")
    forecast_barangay_ranking = safe_read_csv(ARTIFACTS_DIR / "forecast_barangay_ranking.csv")
    forecast_top3_barangays = safe_read_csv(ARTIFACTS_DIR / "forecast_top3_barangays.csv")
    barangay_risk_profile = safe_read_csv(ARTIFACTS_DIR / "barangay_risk_profile.csv")
    meta = safe_read_json(ARTIFACTS_DIR / "meta.json")
    return (
        monthly, model_comparison, feature_importance, feature_sensitivity, forecast,
        barangay_monthly, top_barangay_monthly, top3_barangays_yearly, top3_barangays_overall,
        test_predictions, confusion_matrix_detail, climate_case_correlation, month_profile,
        forecast_barangay_ranking, forecast_top3_barangays, barangay_risk_profile, meta
    )

(
    monthly, model_comparison, feature_importance, feature_sensitivity, forecast,
    barangay_monthly, top_barangay_monthly, top3_barangays_yearly, top3_barangays_overall,
    test_predictions, confusion_matrix_detail, climate_case_correlation, month_profile,
    forecast_barangay_ranking, forecast_top3_barangays, barangay_risk_profile, meta
) = load_artifacts()

model = safe_load_model(ARTIFACTS_DIR / "best_model.joblib")

# ---------------------------
# Helper functions for Live Prediction
# ---------------------------
def month_name_from_number(m):
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return month_names.get(int(m), str(m))

def get_forecast_row(forecast_df, year_num, month_num):
    if forecast_df is None or forecast_df.empty:
        return None
    if not {"Year","Month"}.issubset(forecast_df.columns):
        return None
    subset = forecast_df[(pd.to_numeric(forecast_df["Year"], errors="coerce")==int(year_num)) &
                         (pd.to_numeric(forecast_df["Month"], errors="coerce")==int(month_num))]
    if subset.empty:
        return None
    return subset.iloc[0]

def build_live_prediction_features(year_num, month_num, rainfall_now, humidity_now, temp_now,
                                   cases_lag_1, cases_lag_2, cases_lag_3,
                                   month_profile_df, forecast_row=None):
    # Compute lagged climate values
    if forecast_row is not None:
        rainfall_lag_1 = forecast_row.get("rainfall_lag_1", 0.0)
        rainfall_lag_2 = forecast_row.get("rainfall_lag_2", 0.0)
        rainfall_lag_3 = forecast_row.get("rainfall_lag_3", 0.0)
        rh_lag_1 = forecast_row.get("relative_humidity_lag_1",0.0)
        rh_lag_2 = forecast_row.get("relative_humidity_lag_2",0.0)
        rh_lag_3 = forecast_row.get("relative_humidity_lag_3",0.0)
        temp_lag_1 = forecast_row.get("temp_mid_lag_1",0.0)
        temp_lag_2 = forecast_row.get("temp_mid_lag_2",0.0)
        temp_lag_3 = forecast_row.get("temp_mid_lag_3",0.0)
    else:
        rainfall_lag_1 = rainfall_lag_2 = rainfall_lag_3 = 0.0
        rh_lag_1 = rh_lag_2 = rh_lag_3 = 0.0
        temp_lag_1 = temp_lag_2 = temp_lag_3 = 0.0

    cases_roll3_mean = np.mean([cases_lag_1, cases_lag_2, cases_lag_3])
    cases_roll3_max = np.max([cases_lag_1, cases_lag_2, cases_lag_3])
    month_sin = np.sin(2*np.pi*month_num/12)
    month_cos = np.cos(2*np.pi*month_num/12)

    return {
        "rainfall": float(rainfall_now),
        "relative_humidity": float(humidity_now),
        "temp_mid": float(temp_now),
        "cases_lag_1": float(cases_lag_1),
        "cases_lag_2": float(cases_lag_2),
        "cases_lag_3": float(cases_lag_3),
        "rainfall_lag_1": float(rainfall_lag_1),
        "rainfall_lag_2": float(rainfall_lag_2),
        "rainfall_lag_3": float(rainfall_lag_3),
        "relative_humidity_lag_1": float(rh_lag_1),
        "relative_humidity_lag_2": float(rh_lag_2),
        "relative_humidity_lag_3": float(rh_lag_3),
        "temp_mid_lag_1": float(temp_lag_1),
        "temp_mid_lag_2": float(temp_lag_2),
        "temp_mid_lag_3": float(temp_lag_3),
        "cases_roll3_mean": float(cases_roll3_mean),
        "cases_roll3_max": float(cases_roll3_max),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos)
    }

def outbreak_label_from_binary(x):
    return "Outbreak" if int(x)==1 else "Non-outbreak"

# ---------------------------
# Dashboard Tabs
# ---------------------------
tabs = st.tabs(["Overview","Barangay Analytics","Model Results","Feature Transparency","Forecast & Prediction","Live Prediction"])

# ---------------------------
# Overview Tab
# ---------------------------
with tabs[0]:
    st.header("Historical Dengue Overview")
    if monthly is not None:
        st.dataframe(monthly.head(20), use_container_width=True)

# ---------------------------
# Model Results Tab
# ---------------------------
with tabs[2]:
    st.header("Model Comparison")
    if model_comparison is not None and not model_comparison.empty:
        st.dataframe(model_comparison, use_container_width=True)

# ---------------------------
# Forecast Tab
# ---------------------------
with tabs[4]:
    st.header("Forecasts")
    if forecast is not None and not forecast.empty:
        st.dataframe(forecast.head(20), use_container_width=True)

# ---------------------------
# Live Prediction Tab
# ---------------------------
with tabs[5]:
    st.header("Live Prediction")

    st.info("Estimate outbreak likelihood for any year-month using climate inputs and last 3 months’ dengue cases.")

    if model is None:
        st.warning("Model file not found. Live prediction is unavailable.")
    else:
        feature_columns = meta.get("feature_columns", DEFAULT_FEATURE_COLS) if meta else DEFAULT_FEATURE_COLS
        year_options = [2026,2027,2028,2029,2030]
        month_options = list(range(1,13))

        col1,col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Year", year_options)
        with col2:
            selected_month = st.selectbox("Month", month_options, format_func=lambda x: f"{x}-{month_name_from_number(x)}")

        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0)
        humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
        temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=35.0, value=25.0)
        cases_lag_1 = st.number_input("Cases Last Month", min_value=0, value=5)
        cases_lag_2 = st.number_input("Cases 2 Months Ago", min_value=0, value=4)
        cases_lag_3 = st.number_input("Cases 3 Months Ago", min_value=0, value=3)

        if st.button("Predict"):
            features_dict = build_live_prediction_features(
                year_num=selected_year, month_num=selected_month,
                rainfall_now=rainfall, humidity_now=humidity, temp_now=temp,
                cases_lag_1=cases_lag_1, cases_lag_2=cases_lag_2, cases_lag_3=cases_lag_3,
                month_profile_df=month_profile,
                forecast_row=get_forecast_row(forecast, selected_year, selected_month)
            )

            input_df = pd.DataFrame([features_dict])
            pred = int(model.predict(input_df)[0])
            prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model,"predict_proba") else np.nan

            st.success(f"Predicted Class: {outbreak_label_from_binary(pred)}")
            st.info(f"Predicted Outbreak Probability: {prob:.4f}" if not pd.isna(prob) else "Probability not available")

            st.subheader("Barangays with Highest Predicted Risk")
            if barangay_risk_profile is not None:
                barangay_live = barangay_risk_profile.copy()
                barangay_live["risk_score_raw"] = 0.4*barangay_live.get("overall_share",0)+0.35*barangay_live.get("recent_share",0)+0.25*barangay_live.get("seasonal_share",0)
                total_score = barangay_live["risk_score_raw"].sum()
                barangay_live["risk_score"] = barangay_live["risk_score_raw"]/total_score if total_score>0 else 0
                city_cases_proxy = features_dict.get("cases_roll3_mean",0)*(1+(0 if pd.isna(prob) else prob))
                barangay_live["predicted_city_cases_proxy"] = city_cases_proxy
                barangay_live["predicted_barangay_cases_proxy"] = barangay_live["risk_score"]*city_cases_proxy
                top3 = barangay_live.sort_values("predicted_barangay_cases_proxy",ascending=False).head(3)
                st.dataframe(top3, use_container_width=True)