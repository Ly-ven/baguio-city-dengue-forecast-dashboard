import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Baguio City Dengue Forecast Dashboard",
    layout="wide"
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Machine learning-based dengue outbreak prediction using climate and epidemiological indicators.")


# ============================================================
# Paths and default model features
# ============================================================
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


# ============================================================
# Safe loading helpers
# This allows either:
#   artifacts/file.csv
# or:
#   file.csv in the root folder
# ============================================================
def first_existing_path(*filenames):
    """
    Finds the first existing file from artifacts/ or root folder.
    Example:
        first_existing_path("model_results.csv", "model_comparison.csv")
    """
    for filename in filenames:
        artifact_path = ARTIFACTS_DIR / filename
        root_path = Path(filename)

        if artifact_path.exists():
            return artifact_path
        if root_path.exists():
            return root_path

    return None


def safe_read_csv(*filenames):
    path = first_existing_path(*filenames)
    if path is None:
        return None
    return pd.read_csv(path)


def safe_read_json(*filenames):
    path = first_existing_path(*filenames)
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def safe_load_model():
    path = first_existing_path("best_model.joblib")
    if path is None:
        return None
    return joblib.load(path)


@st.cache_data
def load_all_artifacts():
    prepared = safe_read_csv("prepared_dataset.csv", "monthly_modeling_dataset.csv")
    model_results = safe_read_csv("model_results.csv", "model_comparison.csv")
    test_predictions = safe_read_csv("test_predictions.csv")
    feature_importance = safe_read_csv("feature_importance.csv")
    sensitivity = safe_read_csv("sensitivity_analysis.csv", "feature_sensitivity.csv")
    forecast = safe_read_csv("forecast_df.csv", "forecast_5yr.csv")
    forecast_top3 = safe_read_csv("forecast_top3_barangays.csv")
    climate_corr = safe_read_csv("climate_case_correlation.csv")
    month_profile = safe_read_csv("month_profile.csv")

    barangay_monthly = safe_read_csv("barangay_monthly.csv")
    top_barangay_monthly = safe_read_csv("top_barangay_monthly.csv")
    top3_barangays_yearly = safe_read_csv("top3_barangays_yearly.csv")
    top3_barangays_overall = safe_read_csv("top3_barangays_overall.csv")
    forecast_barangay_ranking = safe_read_csv("forecast_barangay_ranking.csv")
    barangay_risk_profile = safe_read_csv("barangay_risk_profile.csv")

    meta = safe_read_json("meta.json")

    return {
        "prepared": prepared,
        "model_results": model_results,
        "test_predictions": test_predictions,
        "feature_importance": feature_importance,
        "sensitivity": sensitivity,
        "forecast": forecast,
        "forecast_top3": forecast_top3,
        "climate_corr": climate_corr,
        "month_profile": month_profile,
        "barangay_monthly": barangay_monthly,
        "top_barangay_monthly": top_barangay_monthly,
        "top3_barangays_yearly": top3_barangays_yearly,
        "top3_barangays_overall": top3_barangays_overall,
        "forecast_barangay_ranking": forecast_barangay_ranking,
        "barangay_risk_profile": barangay_risk_profile,
        "meta": meta
    }


data = load_all_artifacts()
model = safe_load_model()

prepared = data["prepared"]
model_results = data["model_results"]
test_predictions = data["test_predictions"]
feature_importance = data["feature_importance"]
sensitivity = data["sensitivity"]
forecast = data["forecast"]
forecast_top3 = data["forecast_top3"]
climate_corr = data["climate_corr"]
month_profile = data["month_profile"]

barangay_monthly = data["barangay_monthly"]
top_barangay_monthly = data["top_barangay_monthly"]
top3_barangays_yearly = data["top3_barangays_yearly"]
top3_barangays_overall = data["top3_barangays_overall"]
forecast_barangay_ranking = data["forecast_barangay_ranking"]
barangay_risk_profile = data["barangay_risk_profile"]
meta = data["meta"] or {}


# ============================================================
# Data preparation for display
# ============================================================
for df in [
    prepared,
    test_predictions,
    forecast,
    forecast_top3,
    barangay_monthly,
    top_barangay_monthly,
    top3_barangays_yearly,
    top3_barangays_overall,
    forecast_barangay_ranking,
    barangay_risk_profile
]:
    if df is not None and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


def show_missing_file_warning(file_description):
    st.warning(f"{file_description} file is missing or empty. Please check your uploaded artifacts.")


def get_feature_columns():
    """
    Supports both possible meta keys:
    - feature_cols
    - feature_columns
    """
    feature_cols = meta.get("feature_cols") or meta.get("feature_columns") or DEFAULT_FEATURE_COLS
    return list(feature_cols)


# ============================================================
# Live prediction helper functions
# ============================================================
def month_name_from_number(month_number):
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    return month_names.get(int(month_number), str(month_number))


def get_forecast_row(forecast_df, year_num, month_num):
    if forecast_df is None or forecast_df.empty:
        return None

    if not {"Year", "Month"}.issubset(forecast_df.columns):
        return None

    year_series = pd.to_numeric(forecast_df["Year"], errors="coerce")
    month_series = pd.to_numeric(forecast_df["Month"], errors="coerce")

    subset = forecast_df[
        (year_series == int(year_num)) &
        (month_series == int(month_num))
    ]

    if subset.empty:
        return None

    return subset.iloc[0]


def build_live_prediction_features(
    month_num,
    rainfall_now,
    humidity_now,
    temp_now,
    cases_lag_1,
    cases_lag_2,
    cases_lag_3,
    forecast_row=None
):
    """
    Builds the same type of lagged feature row used during model training.
    Current values come from user input.
    Lagged climate values are taken from the forecast row when available.
    """
    if forecast_row is not None:
        rainfall_lag_1 = forecast_row.get("rainfall_lag_1", 0.0)
        rainfall_lag_2 = forecast_row.get("rainfall_lag_2", 0.0)
        rainfall_lag_3 = forecast_row.get("rainfall_lag_3", 0.0)

        rh_lag_1 = forecast_row.get("relative_humidity_lag_1", 0.0)
        rh_lag_2 = forecast_row.get("relative_humidity_lag_2", 0.0)
        rh_lag_3 = forecast_row.get("relative_humidity_lag_3", 0.0)

        temp_lag_1 = forecast_row.get("temp_mid_lag_1", 0.0)
        temp_lag_2 = forecast_row.get("temp_mid_lag_2", 0.0)
        temp_lag_3 = forecast_row.get("temp_mid_lag_3", 0.0)
    else:
        rainfall_lag_1 = rainfall_lag_2 = rainfall_lag_3 = 0.0
        rh_lag_1 = rh_lag_2 = rh_lag_3 = 0.0
        temp_lag_1 = temp_lag_2 = temp_lag_3 = 0.0

    cases_values = [cases_lag_1, cases_lag_2, cases_lag_3]

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

        "cases_roll3_mean": float(np.mean(cases_values)),
        "cases_roll3_max": float(np.max(cases_values)),

        "month_sin": float(np.sin(2 * np.pi * int(month_num) / 12)),
        "month_cos": float(np.cos(2 * np.pi * int(month_num) / 12))
    }


def outbreak_label_from_binary(value):
    return "Outbreak" if int(value) == 1 else "Non-outbreak"


def get_probability_column(df):
    """
    Handles either possible forecast/test prediction probability column name.
    """
    if df is None:
        return None

    for col in [
        "predicted_outbreak_probability",
        "predicted_probability",
        "outbreak_probability",
        "probability"
    ]:
        if col in df.columns:
            return col

    return None


# ============================================================
# Dashboard tabs
# ============================================================
tabs = st.tabs([
    "Overview",
    "Model Performance",
    "Forecast",
    "Barangay Risk Ranking",
    "Feature Importance",
    "Climate Profile",
    "Prediction Records",
    "Live Prediction"
])


# ============================================================
# Overview
# ============================================================
with tabs[0]:
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", meta.get("best_model", "N/A"))
    col2.metric("Outbreak Threshold", round(float(meta.get("outbreak_threshold", 0)), 2))
    col3.metric("Number of Features", len(get_feature_columns()))

    st.write(
        """
        This dashboard presents the prepared dengue dataset, model evaluation results,
        outbreak forecasts, barangay-level risk rankings, climate profiles, and live
        prediction outputs. The target variable is binary: outbreak or non-outbreak.
        """
    )

    st.subheader("Prepared Dataset Preview")
    if prepared is not None and not prepared.empty:
        st.dataframe(prepared.head(20), use_container_width=True)
    else:
        show_missing_file_warning("Prepared dataset")


# ============================================================
# Model Performance
# ============================================================
with tabs[1]:
    st.header("Model Performance Comparison")

    if model_results is not None and not model_results.empty:
        st.dataframe(model_results, use_container_width=True)

        available_metrics = [
            metric for metric in ["accuracy", "precision", "recall", "f1_score"]
            if metric in model_results.columns
        ]

        if available_metrics and "model" in model_results.columns:
            metric = st.selectbox("Select metric to visualize", available_metrics)

            fig = px.bar(
                model_results,
                x="model",
                y=metric,
                text=metric,
                title=f"Model Comparison by {metric}"
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_yaxes(range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)

            st.info("Reliability is represented using the F1-score because it balances precision and recall.")
        else:
            st.info("The model results table is loaded, but expected metric columns were not found.")
    else:
        show_missing_file_warning("Model results")


# ============================================================
# Forecast
# ============================================================
with tabs[2]:
    st.header("Forecasted Dengue Outbreak Probability")

    if forecast is not None and not forecast.empty:
        probability_col = get_probability_column(forecast)

        if "Date" in forecast.columns and probability_col is not None:
            fig = px.line(
                forecast,
                x="Date",
                y=probability_col,
                markers=True,
                title="Forecasted Outbreak Probability"
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Table")
        st.dataframe(forecast, use_container_width=True)

        if {"Year", "Month"}.issubset(forecast.columns) and probability_col is not None:
            heatmap_df = forecast.pivot_table(
                index="Year",
                columns="Month",
                values=probability_col,
                aggfunc="mean"
            )

            fig_heat = px.imshow(
                heatmap_df,
                text_auto=".2f",
                title="Forecast Heatmap of Outbreak Probability",
                aspect="auto"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        show_missing_file_warning("Forecast")


# ============================================================
# Barangay Risk Ranking
# ============================================================
with tabs[3]:
    st.header("Forecasted Barangay Risk Ranking")

    if forecast_top3 is not None and not forecast_top3.empty:
        if "Date" in forecast_top3.columns:
            available_dates = sorted(forecast_top3["Date"].dt.strftime("%Y-%m").dropna().unique())
            selected_date = st.selectbox("Select forecast month", available_dates)

            filtered = forecast_top3[
                forecast_top3["Date"].dt.strftime("%Y-%m") == selected_date
            ]

            st.dataframe(filtered, use_container_width=True)

            if {"Barangay", "predicted_barangay_cases_proxy"}.issubset(filtered.columns):
                fig = px.bar(
                    filtered,
                    x="Barangay",
                    y="predicted_barangay_cases_proxy",
                    text="predicted_barangay_cases_proxy",
                    title=f"Leading Barangays by Forecasted Risk: {selected_date}"
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(forecast_top3, use_container_width=True)
    elif forecast_barangay_ranking is not None and not forecast_barangay_ranking.empty:
        st.dataframe(forecast_barangay_ranking, use_container_width=True)
    else:
        show_missing_file_warning("Barangay risk ranking")

    st.divider()

    st.subheader("Historical Barangay Tables")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Three Leading Barangays per Year")
        if top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
            st.dataframe(top3_barangays_yearly, use_container_width=True)
        else:
            st.info("Yearly barangay ranking file is not available.")

    with col2:
        st.write("Three Leading Barangays Overall")
        if top3_barangays_overall is not None and not top3_barangays_overall.empty:
            st.dataframe(top3_barangays_overall, use_container_width=True)
        else:
            st.info("Overall barangay ranking file is not available.")


# ============================================================
# Feature Importance
# ============================================================
with tabs[4]:
    st.header("Primary Contributing Features")

    if feature_importance is not None and not feature_importance.empty:
        st.dataframe(feature_importance, use_container_width=True)

        if {"feature", "importance_mean"}.issubset(feature_importance.columns):
            fig = px.bar(
                feature_importance.sort_values("importance_mean", ascending=True).tail(15),
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Primary Contributing Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif {"feature", "importance"}.issubset(feature_importance.columns):
            fig = px.bar(
                feature_importance.sort_values("importance", ascending=True).tail(15),
                x="importance",
                y="feature",
                orientation="h",
                title="Primary Contributing Features"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file_warning("Feature importance")

    st.subheader("Sensitivity Analysis")

    if sensitivity is not None and not sensitivity.empty:
        st.dataframe(sensitivity, use_container_width=True)

        if {"feature", "delta_probability"}.issubset(sensitivity.columns):
            fig2 = px.bar(
                sensitivity,
                x="feature",
                y="delta_probability",
                text="delta_probability",
                title="Effect of +10% Climate Change on Outbreak Probability"
            )
            fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        show_missing_file_warning("Sensitivity analysis")


# ============================================================
# Climate Profile
# ============================================================
with tabs[5]:
    st.header("Climate Profile")

    st.subheader("Climate-Case Correlation")

    if climate_corr is not None and not climate_corr.empty:
        st.dataframe(climate_corr, use_container_width=True)

        corr_col = None
        for possible_col in ["pearson_corr_with_CHSO_cases", "correlation", "corr"]:
            if possible_col in climate_corr.columns:
                corr_col = possible_col
                break

        if {"feature"}.issubset(climate_corr.columns) and corr_col is not None:
            fig = px.bar(
                climate_corr,
                x="feature",
                y=corr_col,
                text=corr_col,
                title="Climate-Case Correlation"
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file_warning("Climate-case correlation")

    st.subheader("Average Dengue Cases by Month")

    if month_profile is not None and not month_profile.empty:
        st.dataframe(month_profile, use_container_width=True)

        if {"MonthName", "CHSO_cases"}.issubset(month_profile.columns):
            fig2 = px.bar(
                month_profile,
                x="MonthName",
                y="CHSO_cases",
                text="CHSO_cases",
                title="Average Dengue Cases by Month"
            )
            fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)
        elif {"Month", "CHSO_cases"}.issubset(month_profile.columns):
            fig2 = px.bar(
                month_profile,
                x="Month",
                y="CHSO_cases",
                text="CHSO_cases",
                title="Average Dengue Cases by Month"
            )
            fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        show_missing_file_warning("Monthly climate/case profile")


# ============================================================
# Prediction Records
# ============================================================
with tabs[6]:
    st.header("Month-by-Month Test Predictions")

    if test_predictions is not None and not test_predictions.empty:
        st.dataframe(test_predictions, use_container_width=True)

        probability_col = get_probability_column(test_predictions)

        if "Date" in test_predictions.columns and probability_col is not None:
            fig = px.line(
                test_predictions,
                x="Date",
                y=probability_col,
                markers=True,
                title="Predicted Outbreak Probability on Test Months"
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file_warning("Test predictions")


# ============================================================
# Live Prediction
# ============================================================
with tabs[7]:
    st.header("Live Prediction")

    st.info("Estimate outbreak likelihood for a selected year-month using climate inputs and the last three months of dengue cases.")

    if model is None:
        st.warning("Model file not found. Please upload best_model.joblib to the artifacts folder or root folder.")
    else:
        feature_columns = get_feature_columns()

        if forecast is not None and {"Year", "Month"}.issubset(forecast.columns):
            available_years = sorted(pd.to_numeric(forecast["Year"], errors="coerce").dropna().astype(int).unique())
            if not available_years:
                available_years = [2026, 2027, 2028, 2029, 2030]
        else:
            available_years = [2026, 2027, 2028, 2029, 2030]

        month_options = list(range(1, 13))

        col1, col2 = st.columns(2)

        with col1:
            selected_year = st.selectbox("Year", available_years)

        with col2:
            selected_month = st.selectbox(
                "Month",
                month_options,
                format_func=lambda x: f"{x} - {month_name_from_number(x)}"
            )

        st.subheader("Climate Inputs")
        col3, col4, col5 = st.columns(3)

        with col3:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0)

        with col4:
            humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)

        with col5:
            temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=40.0, value=25.0)

        st.subheader("Recent Dengue Cases")
        col6, col7, col8 = st.columns(3)

        with col6:
            cases_lag_1 = st.number_input("Cases Last Month", min_value=0, value=5)

        with col7:
            cases_lag_2 = st.number_input("Cases 2 Months Ago", min_value=0, value=4)

        with col8:
            cases_lag_3 = st.number_input("Cases 3 Months Ago", min_value=0, value=3)

        if st.button("Predict Outbreak Risk"):
            forecast_row = get_forecast_row(forecast, selected_year, selected_month)

            features_dict = build_live_prediction_features(
                month_num=selected_month,
                rainfall_now=rainfall,
                humidity_now=humidity,
                temp_now=temp,
                cases_lag_1=cases_lag_1,
                cases_lag_2=cases_lag_2,
                cases_lag_3=cases_lag_3,
                forecast_row=forecast_row
            )

            input_df = pd.DataFrame([features_dict])
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            try:
                pred = int(model.predict(input_df)[0])
                prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else np.nan

                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Predicted Class", outbreak_label_from_binary(pred))
                result_col2.metric(
                    "Predicted Outbreak Probability",
                    f"{prob:.4f}" if not pd.isna(prob) else "N/A"
                )

                st.subheader("Input Features Used by the Model")
                st.dataframe(input_df, use_container_width=True)

                st.subheader("Barangays with Highest Predicted Risk")

                if barangay_risk_profile is not None and not barangay_risk_profile.empty:
                    barangay_live = barangay_risk_profile.copy()

                    for col in ["overall_share", "recent_share", "seasonal_share"]:
                        if col not in barangay_live.columns:
                            barangay_live[col] = 0.0

                    barangay_live["risk_score_raw"] = (
                        0.40 * barangay_live["overall_share"] +
                        0.35 * barangay_live["recent_share"] +
                        0.25 * barangay_live["seasonal_share"]
                    )

                    total_score = barangay_live["risk_score_raw"].sum()

                    if total_score > 0:
                        barangay_live["risk_score"] = barangay_live["risk_score_raw"] / total_score
                    else:
                        barangay_live["risk_score"] = 0.0

                    city_cases_proxy = features_dict["cases_roll3_mean"] * (1 + (0 if pd.isna(prob) else prob))

                    barangay_live["predicted_city_cases_proxy"] = city_cases_proxy
                    barangay_live["predicted_barangay_cases_proxy"] = (
                        barangay_live["risk_score"] * city_cases_proxy
                    )

                    top3 = barangay_live.sort_values(
                        "predicted_barangay_cases_proxy",
                        ascending=False
                    ).head(3)

                    st.dataframe(top3, use_container_width=True)

                    if {"Barangay", "predicted_barangay_cases_proxy"}.issubset(top3.columns):
                        fig = px.bar(
                            top3,
                            x="Barangay",
                            y="predicted_barangay_cases_proxy",
                            text="predicted_barangay_cases_proxy",
                            title="Barangays with Highest Predicted Risk"
                        )
                        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Barangay risk profile file is unavailable, so barangay-level live ranking cannot be generated.")

            except Exception as e:
                st.error("Prediction failed. Please check that your model features match the input features.")
                st.exception(e)
