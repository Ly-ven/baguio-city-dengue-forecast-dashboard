import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Baguio City Dengue Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive web-based dashboard for dengue prediction, model evaluation, and forecast visualization")

ARTIFACTS_DIR = Path("artifacts")

DEFAULT_FEATURE_COLS = [
    "rainfall", "relative_humidity", "temp_mid",
    "cases_lag_1", "cases_lag_2", "cases_lag_3",
    "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
    "relative_humidity_lag_1", "relative_humidity_lag_2", "relative_humidity_lag_3",
    "temp_mid_lag_1", "temp_mid_lag_2", "temp_mid_lag_3",
    "cases_roll3_mean", "cases_roll3_max",
    "month_sin", "month_cos",
]

CSV_ALIASES = {
    "monthly": ["prepared_dataset.csv", "monthly_modeling_dataset.csv"],
    "model_results": ["model_results.csv", "model_comparison.csv"],
    "feature_importance": ["feature_importance.csv"],
    "sensitivity": ["sensitivity_analysis.csv", "feature_sensitivity.csv"],
    "forecast": ["forecast_df.csv", "forecast_5yr.csv"],
    "test_predictions": ["test_predictions.csv"],
    "climate_case_correlation": ["climate_case_correlation.csv"],
    "month_profile": ["month_profile.csv"],
    "forecast_barangay_ranking": ["forecast_barangay_ranking.csv"],
    "forecast_top3_barangays": ["forecast_top3_barangays.csv"],
    # Optional legacy or expanded files. The revised Colab export does not require these.
    "barangay_monthly": ["barangay_monthly.csv"],
    "top_barangay_monthly": ["top_barangay_monthly.csv", "monthly_top_barangay.csv"],
    "top3_barangays_yearly": ["top3_barangays_yearly.csv"],
    "top3_barangays_overall": ["top3_barangays_overall.csv"],
    "barangay_risk_profile": ["barangay_risk_profile.csv"],
}

MODEL_ALIASES = ["best_model.joblib", "best_model.pkl"]
META_ALIASES = ["meta.json"]


# -------------------------
# Loading helpers
# -------------------------
def read_csv_from_dir(directory: Path, names):
    for name in names:
        path = directory / name
        if path.exists():
            return pd.read_csv(path)
    return None


def read_json_from_dir(directory: Path, names):
    for name in names:
        path = directory / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_model_from_dir(directory: Path, names):
    for name in names:
        path = directory / name
        if path.exists():
            return joblib.load(path)
    return None


@st.cache_data(show_spinner=False)
def load_csv_artifacts_from_dir():
    return {key: read_csv_from_dir(ARTIFACTS_DIR, names) for key, names in CSV_ALIASES.items()}


@st.cache_data(show_spinner=False)
def load_meta_from_dir():
    return read_json_from_dir(ARTIFACTS_DIR, META_ALIASES)


@st.cache_resource(show_spinner=False)
def load_model_resource_from_dir():
    return load_model_from_dir(ARTIFACTS_DIR, MODEL_ALIASES)


def _zip_contains(zf, filename):
    target = filename.replace("\\", "/")
    for member in zf.namelist():
        if member.replace("\\", "/").split("/")[-1] == target:
            return member
    return None


@st.cache_data(show_spinner=False)
def load_csv_artifacts_from_zip(zip_bytes):
    out = {key: None for key in CSV_ALIASES}
    if not zip_bytes:
        return out
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for key, names in CSV_ALIASES.items():
            for name in names:
                member = _zip_contains(zf, name)
                if member is not None:
                    with zf.open(member) as f:
                        out[key] = pd.read_csv(f)
                    break
    return out


@st.cache_data(show_spinner=False)
def load_meta_from_zip(zip_bytes):
    if not zip_bytes:
        return None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in META_ALIASES:
            member = _zip_contains(zf, name)
            if member is not None:
                with zf.open(member) as f:
                    return json.loads(f.read().decode("utf-8"))
    return None


@st.cache_resource(show_spinner=False)
def load_model_from_zip(zip_bytes):
    if not zip_bytes:
        return None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in MODEL_ALIASES:
            member = _zip_contains(zf, name)
            if member is not None:
                with zf.open(member) as f:
                    return joblib.load(io.BytesIO(f.read()))
    return None


def coalesce_artifacts(base, override):
    merged = dict(base)
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


def parse_dates(df):
    if df is not None and "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def safe_metric_value(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def round_display_columns(df, columns, decimals=2):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df


def month_name_from_number(month_num):
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    try:
        return month_names.get(int(month_num), str(month_num))
    except Exception:
        return str(month_num)


def outbreak_label_from_binary(value):
    try:
        return "Outbreak" if int(value) == 1 else "Non-outbreak"
    except Exception:
        return "Unknown"


def get_feature_columns(meta, model=None):
    if meta:
        if isinstance(meta.get("feature_cols"), list):
            return meta["feature_cols"]
        if isinstance(meta.get("feature_columns"), list):
            return meta["feature_columns"]
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return DEFAULT_FEATURE_COLS


def get_threshold_text(meta):
    if not meta:
        return "N/A"
    threshold = meta.get("outbreak_threshold", meta.get("outbreak_threshold_cases", None))
    if threshold is None:
        return "N/A"
    if isinstance(threshold, (int, float)):
        return f"{threshold:.2f} cases"
    return str(threshold)


def get_month_profile_row(month_num, month_profile_df):
    if month_profile_df is None or month_profile_df.empty or "Month" not in month_profile_df.columns:
        return None
    subset = month_profile_df[pd.to_numeric(month_profile_df["Month"], errors="coerce") == int(month_num)]
    if subset.empty:
        return None
    return subset.iloc[0]


def get_profile_value(month_num, col_name, month_profile_df, fallback_df=None, default=0.0):
    row = get_month_profile_row(month_num, month_profile_df)
    if row is not None and col_name in row.index and pd.notna(row[col_name]):
        return float(row[col_name])
    if fallback_df is not None and col_name in fallback_df.columns:
        series = pd.to_numeric(fallback_df[col_name], errors="coerce").dropna()
        if len(series) > 0:
            return float(series.mean())
    return float(default)


def get_previous_year_months(year_num, month_num):
    periods = []
    current_year = int(year_num)
    current_month = int(month_num)
    for _ in range(3):
        current_month -= 1
        if current_month == 0:
            current_month = 12
            current_year -= 1
        periods.append((current_year, current_month))
    return periods[0], periods[1], periods[2]


def get_forecast_row(forecast_df, year_num, month_num):
    if forecast_df is None or forecast_df.empty or not {"Year", "Month"}.issubset(forecast_df.columns):
        return None
    subset = forecast_df[
        (pd.to_numeric(forecast_df["Year"], errors="coerce") == int(year_num)) &
        (pd.to_numeric(forecast_df["Month"], errors="coerce") == int(month_num))
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def numeric_from_row(row, col_name, default=0.0):
    if row is None:
        return float(default)
    value = row.get(col_name, default)
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return float(default)
    return float(value)


def get_reasonable_range(df, col_name, fallback_min=0.0, fallback_max=100.0, include_values=None):
    values = []
    if df is not None and col_name in df.columns:
        values.extend(pd.to_numeric(df[col_name], errors="coerce").dropna().tolist())
    if include_values:
        values.extend([float(v) for v in include_values if v is not None and not pd.isna(v)])
    if values:
        vmin = float(min(values))
        vmax = float(max(values))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax
    return fallback_min, fallback_max


def clamp(value, min_value, max_value):
    return float(max(min_value, min(max_value, value)))


def build_live_prediction_features(
    year_num,
    month_num,
    rainfall_now,
    humidity_now,
    temp_now,
    cases_lag_1,
    cases_lag_2,
    cases_lag_3,
    month_profile_df,
    fallback_monthly_df,
    forecast_row=None,
):
    if forecast_row is not None:
        rainfall_lag_1 = numeric_from_row(forecast_row, "rainfall_lag_1", 0.0)
        rainfall_lag_2 = numeric_from_row(forecast_row, "rainfall_lag_2", 0.0)
        rainfall_lag_3 = numeric_from_row(forecast_row, "rainfall_lag_3", 0.0)
        rh_lag_1 = numeric_from_row(forecast_row, "relative_humidity_lag_1", 0.0)
        rh_lag_2 = numeric_from_row(forecast_row, "relative_humidity_lag_2", 0.0)
        rh_lag_3 = numeric_from_row(forecast_row, "relative_humidity_lag_3", 0.0)
        temp_lag_1 = numeric_from_row(forecast_row, "temp_mid_lag_1", 0.0)
        temp_lag_2 = numeric_from_row(forecast_row, "temp_mid_lag_2", 0.0)
        temp_lag_3 = numeric_from_row(forecast_row, "temp_mid_lag_3", 0.0)
    else:
        (_, prev1), (_, prev2), (_, prev3) = get_previous_year_months(year_num, month_num)
        rainfall_lag_1 = get_profile_value(prev1, "rainfall", month_profile_df, fallback_monthly_df, 0.0)
        rainfall_lag_2 = get_profile_value(prev2, "rainfall", month_profile_df, fallback_monthly_df, 0.0)
        rainfall_lag_3 = get_profile_value(prev3, "rainfall", month_profile_df, fallback_monthly_df, 0.0)
        rh_lag_1 = get_profile_value(prev1, "relative_humidity", month_profile_df, fallback_monthly_df, 0.0)
        rh_lag_2 = get_profile_value(prev2, "relative_humidity", month_profile_df, fallback_monthly_df, 0.0)
        rh_lag_3 = get_profile_value(prev3, "relative_humidity", month_profile_df, fallback_monthly_df, 0.0)
        temp_lag_1 = get_profile_value(prev1, "temp_mid", month_profile_df, fallback_monthly_df, 0.0)
        temp_lag_2 = get_profile_value(prev2, "temp_mid", month_profile_df, fallback_monthly_df, 0.0)
        temp_lag_3 = get_profile_value(prev3, "temp_mid", month_profile_df, fallback_monthly_df, 0.0)

    cases_roll3_mean = float(np.mean([cases_lag_1, cases_lag_2, cases_lag_3]))
    cases_roll3_max = float(np.max([cases_lag_1, cases_lag_2, cases_lag_3]))

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
        "cases_roll3_mean": cases_roll3_mean,
        "cases_roll3_max": cases_roll3_max,
        "month_sin": float(np.sin(2 * np.pi * int(month_num) / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * int(month_num) / 12.0)),
    }


def get_selected_barangay_ranking(ranking_df, top3_df, year_num, month_num, probability, input_values):
    source = None
    if ranking_df is not None and not ranking_df.empty and {"Year", "Month"}.issubset(ranking_df.columns):
        source = ranking_df[
            (pd.to_numeric(ranking_df["Year"], errors="coerce") == int(year_num)) &
            (pd.to_numeric(ranking_df["Month"], errors="coerce") == int(month_num))
        ].copy()
    if (source is None or source.empty) and top3_df is not None and not top3_df.empty and {"Year", "Month"}.issubset(top3_df.columns):
        source = top3_df[
            (pd.to_numeric(top3_df["Year"], errors="coerce") == int(year_num)) &
            (pd.to_numeric(top3_df["Month"], errors="coerce") == int(month_num))
        ].copy()
    if source is None or source.empty or "Barangay" not in source.columns:
        return None

    if "risk_score" in source.columns:
        scores = pd.to_numeric(source["risk_score"], errors="coerce").fillna(0.0)
    elif "risk_score_raw" in source.columns:
        scores = pd.to_numeric(source["risk_score_raw"], errors="coerce").fillna(0.0)
    elif "predicted_barangay_cases_proxy" in source.columns:
        scores = pd.to_numeric(source["predicted_barangay_cases_proxy"], errors="coerce").fillna(0.0)
    else:
        scores = pd.Series(np.ones(len(source)), index=source.index)

    score_sum = float(scores.sum())
    source["risk_score"] = scores / score_sum if score_sum > 0 else 0.0

    prob = 0.0 if probability is None or pd.isna(probability) else float(probability)
    city_cases_proxy = float(input_values.get("cases_roll3_mean", 0.0)) * (1.0 + prob)
    source["predicted_city_cases_proxy"] = city_cases_proxy
    source["predicted_barangay_cases_proxy"] = source["risk_score"] * city_cases_proxy
    source["predicted_barangay_label"] = "Higher Risk"

    return source.sort_values("predicted_barangay_cases_proxy", ascending=False).head(3)


# -------------------------
# Load artifacts
# -------------------------
st.sidebar.header("About")
st.sidebar.write(
    "This dashboard displays historical dengue cases, model results, feature contributions, "
    "forecast outputs, and live prediction from the revised Google Colab workflow."
)

uploaded_zip = st.sidebar.file_uploader(
    "Upload dashboard_artifacts.zip (optional)",
    type=["zip"],
    help="Use this only if the artifacts folder is not already included in your GitHub repository.",
)

base_artifacts = load_csv_artifacts_from_dir()
base_meta = load_meta_from_dir()
base_model = load_model_resource_from_dir()

if uploaded_zip is not None:
    zip_bytes = uploaded_zip.getvalue()
    zip_artifacts = load_csv_artifacts_from_zip(zip_bytes)
    zip_meta = load_meta_from_zip(zip_bytes)
    zip_model = load_model_from_zip(zip_bytes)
    artifacts = coalesce_artifacts(base_artifacts, zip_artifacts)
    meta = zip_meta if zip_meta is not None else base_meta
    model = zip_model if zip_model is not None else base_model
    st.sidebar.success("ZIP artifacts loaded.")
else:
    artifacts = base_artifacts
    meta = base_meta
    model = base_model

monthly = parse_dates(artifacts["monthly"])
model_results = artifacts["model_results"]
feature_importance = artifacts["feature_importance"]
sensitivity = artifacts["sensitivity"]
forecast = parse_dates(artifacts["forecast"])
test_predictions = parse_dates(artifacts["test_predictions"])
climate_case_correlation = artifacts["climate_case_correlation"]
month_profile = artifacts["month_profile"]
forecast_barangay_ranking = parse_dates(artifacts["forecast_barangay_ranking"])
forecast_top3_barangays = parse_dates(artifacts["forecast_top3_barangays"])
barangay_monthly = parse_dates(artifacts["barangay_monthly"])
top_barangay_monthly = parse_dates(artifacts["top_barangay_monthly"])
top3_barangays_yearly = artifacts["top3_barangays_yearly"]
top3_barangays_overall = artifacts["top3_barangays_overall"]
barangay_risk_profile = artifacts["barangay_risk_profile"]

if meta:
    st.sidebar.success(f"Best Model: {meta.get('best_model', 'Unknown')}")
    st.sidebar.info(f"Outbreak Threshold: {get_threshold_text(meta)}")
    if meta.get("reliability_metric"):
        st.sidebar.caption(f"Reliability metric: {meta.get('reliability_metric')}")
else:
    st.sidebar.warning("meta.json not found.")

if model is None:
    st.sidebar.warning("best_model.joblib not found. Forecast charts will still show, but live prediction will be disabled.")
else:
    st.sidebar.success("Model file loaded.")

if monthly is None:
    st.error("prepared_dataset.csv is required. Put dashboard_artifacts.zip contents inside an artifacts/ folder, or upload the ZIP in the sidebar.")
    st.stop()

# Make sure month profile has month names even if revised export only contains Month and CHSO_cases.
if month_profile is not None and not month_profile.empty and "Month" in month_profile.columns and "MonthName" not in month_profile.columns:
    month_profile = month_profile.copy()
    month_profile["MonthName"] = month_profile["Month"].apply(month_name_from_number)

feature_columns = get_feature_columns(meta, model)

# -------------------------
# Dashboard tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Barangay Analytics",
    "Model Results",
    "Feature Transparency",
    "Forecast & Prediction",
])

with tab1:
    st.header("Historical Dengue Overview")

    total_cases = int(pd.to_numeric(monthly.get("CHSO_cases", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "CHSO_cases" in monthly.columns else 0
    avg_cases = pd.to_numeric(monthly["CHSO_cases"], errors="coerce").mean() if "CHSO_cases" in monthly.columns else np.nan
    outbreak_months = int(pd.to_numeric(monthly.get("outbreak", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "outbreak" in monthly.columns else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Months", len(monthly))
    col2.metric("Total Cases", f"{total_cases:,}")
    col3.metric("Average Monthly Cases", safe_metric_value(avg_cases))
    col4.metric("Outbreak Months", outbreak_months if outbreak_months is not None else "N/A")

    st.subheader("What is the model predicting?")
    threshold_text = get_threshold_text(meta)
    st.info(
        "The model predicts whether a selected month is classified as an **outbreak** or **non-outbreak** month. "
        f"The revised Colab metadata reports an outbreak threshold of **{threshold_text}**."
    )

    st.subheader("Monthly Dengue Cases")
    if {"Date", "CHSO_cases"}.issubset(monthly.columns):
        fig_line = px.line(
            monthly,
            x="Date",
            y="CHSO_cases",
            markers=True,
            title="Monthly Dengue Cases in Baguio City (CHSO)",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Date and CHSO_cases columns are required for the monthly case chart.")

    st.subheader("Year-Month Heatmap of Dengue Cases")
    if {"Year", "Month", "CHSO_cases"}.issubset(monthly.columns):
        heat = monthly.pivot_table(index="Year", columns="Month", values="CHSO_cases", aggfunc="sum")
        fig_heat = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Year-Month Heatmap of Dengue Cases",
        )
        fig_heat.update_xaxes(title="Month")
        fig_heat.update_yaxes(title="Year")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Rainfall vs Relative Humidity Sized by Dengue Cases")
    if {"rainfall", "relative_humidity", "CHSO_cases"}.issubset(monthly.columns):
        hover_cols = [c for c in ["Date", "temp_mid", "outbreak"] if c in monthly.columns]
        fig_bubble = px.scatter(
            monthly,
            x="rainfall",
            y="relative_humidity",
            size="CHSO_cases",
            color="CHSO_cases",
            hover_data=hover_cols,
            title="Rainfall vs Relative Humidity Sized by Dengue Cases",
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.subheader("Climate-Case Correlation")
    if climate_case_correlation is not None and not climate_case_correlation.empty:
        st.dataframe(round_display_columns(climate_case_correlation, ["pearson_corr_with_CHSO_cases"], 4), use_container_width=True)
        if {"feature", "pearson_corr_with_CHSO_cases"}.issubset(climate_case_correlation.columns):
            fig_corr = px.bar(
                round_display_columns(climate_case_correlation, ["pearson_corr_with_CHSO_cases"], 4),
                x="feature",
                y="pearson_corr_with_CHSO_cases",
                text="pearson_corr_with_CHSO_cases",
                title="Climate-Case Correlation",
            )
            fig_corr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("climate_case_correlation.csv not found or empty.")

    st.subheader("Average Monthly Profile")
    if month_profile is not None and not month_profile.empty:
        st.dataframe(round_display_columns(month_profile, ["CHSO_cases", "rainfall", "relative_humidity", "temp_mid"], 2), use_container_width=True)
        if {"MonthName", "CHSO_cases"}.issubset(month_profile.columns):
            fig_month_profile = px.bar(
                round_display_columns(month_profile, ["CHSO_cases"], 2),
                x="MonthName",
                y="CHSO_cases",
                text="CHSO_cases",
                title="Average CHSO Cases by Month",
            )
            fig_month_profile.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig_month_profile, use_container_width=True)
    else:
        st.warning("month_profile.csv not found or empty.")

    with st.expander("Prepared Dataset Preview"):
        st.dataframe(monthly.head(60), use_container_width=True)

with tab2:
    st.header("Barangay Analytics")

    st.subheader("Forecast Barangay Risk Ranking")
    if forecast_top3_barangays is not None and not forecast_top3_barangays.empty:
        display_cols = [c for c in [
            "Date", "Year", "Month", "Barangay", "overall_share", "recent_share", "seasonal_share",
            "risk_score_raw", "risk_score", "predicted_outbreak_probability",
            "predicted_city_cases_proxy", "predicted_barangay_cases_proxy", "predicted_barangay_label",
        ] if c in forecast_top3_barangays.columns]
        st.dataframe(
            round_display_columns(
                forecast_top3_barangays[display_cols],
                ["overall_share", "recent_share", "seasonal_share", "risk_score_raw", "risk_score", "predicted_outbreak_probability", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy"],
                4,
            ),
            use_container_width=True,
        )

        if "Date" in forecast_top3_barangays.columns:
            month_options = forecast_top3_barangays["Date"].dropna().dt.strftime("%Y-%m").unique().tolist()
            selected_month_text = st.selectbox("Select forecast month", month_options, key="barangay_forecast_month")
            selected_rows = forecast_top3_barangays[forecast_top3_barangays["Date"].dt.strftime("%Y-%m") == selected_month_text].copy()
        elif {"Year", "Month"}.issubset(forecast_top3_barangays.columns):
            forecast_top3_barangays["YearMonth"] = forecast_top3_barangays["Year"].astype(str) + "-" + forecast_top3_barangays["Month"].astype(str).str.zfill(2)
            month_options = forecast_top3_barangays["YearMonth"].dropna().unique().tolist()
            selected_month_text = st.selectbox("Select forecast month", month_options, key="barangay_forecast_month")
            selected_rows = forecast_top3_barangays[forecast_top3_barangays["YearMonth"] == selected_month_text].copy()
        else:
            selected_rows = forecast_top3_barangays.head(3).copy()
            selected_month_text = "Selected Forecast Month"

        if {"Barangay", "predicted_barangay_cases_proxy"}.issubset(selected_rows.columns):
            fig_barangay_forecast = px.bar(
                round_display_columns(selected_rows, ["predicted_barangay_cases_proxy"], 2),
                x="Barangay",
                y="predicted_barangay_cases_proxy",
                color="Barangay",
                text="predicted_barangay_cases_proxy",
                title=f"Three Barangays with the Highest Predicted Risk - {selected_month_text}",
            )
            fig_barangay_forecast.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig_barangay_forecast, use_container_width=True)
    else:
        st.warning("forecast_top3_barangays.csv not found or empty.")

    if forecast_barangay_ranking is not None and not forecast_barangay_ranking.empty:
        with st.expander("View complete forecast barangay ranking table"):
            st.dataframe(
                round_display_columns(
                    forecast_barangay_ranking,
                    ["overall_share", "recent_share", "seasonal_share", "risk_score_raw", "risk_score", "predicted_outbreak_probability", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy"],
                    4,
                ),
                use_container_width=True,
            )

    st.subheader("Historical Barangay Tables")
    if top_barangay_monthly is not None and not top_barangay_monthly.empty:
        st.markdown("**Barangay with the highest recorded dengue cases per month**")
        st.dataframe(top_barangay_monthly, use_container_width=True)

    hist_col1, hist_col2 = st.columns(2)
    with hist_col1:
        if top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
            st.markdown("**Three barangays with the highest dengue cases per year**")
            if {"Year", "Barangay", "Barangay_cases"}.issubset(top3_barangays_yearly.columns):
                fig_tree = px.treemap(
                    top3_barangays_yearly,
                    path=["Year", "Barangay"],
                    values="Barangay_cases",
                    color="Barangay_cases",
                    title="Three Barangays with the Highest Dengue Cases per Year",
                )
                st.plotly_chart(fig_tree, use_container_width=True)
            st.dataframe(top3_barangays_yearly, use_container_width=True)
    with hist_col2:
        if top3_barangays_overall is not None and not top3_barangays_overall.empty:
            st.markdown("**Three barangays with the highest dengue cases overall**")
            if {"Barangay", "Barangay_cases"}.issubset(top3_barangays_overall.columns):
                fig_top3 = px.bar(
                    top3_barangays_overall,
                    x="Barangay",
                    y="Barangay_cases",
                    text="Barangay_cases",
                    title="Three Barangays with the Highest Overall Dengue Cases",
                )
                fig_top3.update_traces(textposition="outside")
                st.plotly_chart(fig_top3, use_container_width=True)
            st.dataframe(top3_barangays_overall, use_container_width=True)

    if barangay_monthly is not None and not barangay_monthly.empty:
        with st.expander("View barangay monthly records"):
            st.dataframe(barangay_monthly, use_container_width=True)

with tab3:
    st.header("Model Results")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_results is not None and not model_results.empty:
        metric_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1_score", "auc", "n_test_months"] if c in model_results.columns]
        st.dataframe(round_display_columns(model_results[metric_cols], ["accuracy", "precision", "recall", "f1_score", "auc"], 4), use_container_width=True)

        st.subheader("Model Comparison by SOP Metrics")
        core_metrics = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in model_results.columns]
        if "model" in model_results.columns and core_metrics:
            results_long = model_results.melt(
                id_vars="model",
                value_vars=core_metrics,
                var_name="Metric",
                value_name="Score",
            )
            fig_model = px.bar(
                results_long,
                x="model",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score",
                title="Model Comparison by Accuracy, Precision, Recall, and F1 Score",
            )
            fig_model.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig_model, use_container_width=True)

        if "auc" in model_results.columns:
            with st.expander("View additional AUC metric"):
                auc_df = round_display_columns(model_results[["model", "auc"]].copy(), ["auc"], 4)
                st.dataframe(auc_df, use_container_width=True)
    else:
        st.warning("model_results.csv not found or empty.")

    st.subheader("How to read the metrics")
    st.markdown(
        """
- **Accuracy** shows the overall proportion of correct monthly classifications.  
- **Precision** shows how often an outbreak prediction is correct.  
- **Recall / Sensitivity** shows how many actual outbreak months the model catches.  
- **F1 score** is the reliability metric because it balances precision and recall.  

A model can have high accuracy but still miss outbreak months, so precision, recall, and F1 score should be interpreted together.
"""
    )

    st.subheader("Month-by-Month Test Predictions")
    if test_predictions is not None and not test_predictions.empty:
        total_test = len(test_predictions)
        correct_test = int(pd.to_numeric(test_predictions.get("is_correct", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "is_correct" in test_predictions.columns else None
        test_col1, test_col2 = st.columns(2)
        test_col1.metric("Test Set Months", total_test)
        test_col2.metric("Correct Predictions", correct_test if correct_test is not None else "N/A")
        st.dataframe(round_display_columns(test_predictions, ["predicted_probability"], 4), use_container_width=True)
    else:
        st.warning("test_predictions.csv not found or empty.")

with tab4:
    st.header("Feature Transparency")

    st.subheader("Feature Importance")
    if feature_importance is not None and not feature_importance.empty:
        st.dataframe(round_display_columns(feature_importance, ["importance_mean", "importance_std"], 4), use_container_width=True)
        if {"feature", "importance_mean"}.issubset(feature_importance.columns):
            plot_df = round_display_columns(feature_importance.head(15), ["importance_mean"], 4)
            fig_importance = px.bar(
                plot_df,
                x="importance_mean",
                y="feature",
                orientation="h",
                text="importance_mean",
                title="Most Influential Contributing Features",
            )
            fig_importance.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("feature_importance.csv not found or empty.")

    st.subheader("Sensitivity Analysis")
    if sensitivity is not None and not sensitivity.empty:
        st.dataframe(round_display_columns(sensitivity, ["base_avg_outbreak_probability", "new_avg_outbreak_probability", "delta_probability"], 4), use_container_width=True)
        if {"feature", "delta_probability"}.issubset(sensitivity.columns):
            sens_df = round_display_columns(sensitivity, ["delta_probability"], 4)
            fig_sens = px.bar(
                sens_df,
                x="feature",
                y="delta_probability",
                text="delta_probability",
                title="Effect of +10% Change in Climate Variable on Outbreak Probability",
            )
            fig_sens.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.warning("sensitivity_analysis.csv not found or empty.")

    st.subheader("How to interpret this")
    st.markdown(
        """
- **Feature importance** shows which variables the selected model relied on most during classification.  
- **Lagged case variables** indicate that recent dengue history contributes to prediction.  
- **Sensitivity analysis** estimates how outbreak probability changes when a climate variable increases by 10%.  
- These outputs explain model behavior; they do not prove biological causation by themselves.
"""
    )

with tab5:
    st.header("Forecast & Live Prediction")

    st.subheader("5-Year Forecast")
    if forecast is not None and not forecast.empty:
        st.dataframe(round_display_columns(forecast.head(30), ["predicted_outbreak_probability", "predicted_city_cases_proxy"], 4), use_container_width=True)

        if {"Date", "predicted_outbreak_probability"}.issubset(forecast.columns):
            fig_forecast = px.line(
                round_display_columns(forecast, ["predicted_outbreak_probability"], 4),
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability",
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        if {"Year", "Month", "predicted_outbreak_probability"}.issubset(forecast.columns):
            forecast_heat = forecast.pivot_table(
                index="Year",
                columns="Month",
                values="predicted_outbreak_probability",
            )
            fig_forecast_heat = px.imshow(
                forecast_heat,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Forecast Heatmap of Outbreak Probability",
            )
            fig_forecast_heat.update_xaxes(title="Month")
            fig_forecast_heat.update_yaxes(title="Year")
            st.plotly_chart(fig_forecast_heat, use_container_width=True)
    else:
        st.warning("forecast_df.csv not found or empty.")

    st.subheader("Live Prediction")
    st.info(
        "Select a target year-month, enter climate values for that month, and enter dengue case counts from the previous three months. "
        "The dashboard automatically prepares lagged climate variables, rolling case features, and seasonality inputs."
    )

    with st.expander("How to understand the live prediction inputs"):
        st.markdown(
            """
**Rainfall, relative humidity, and temperature** should describe the selected target month.  
**Cases last month, 2 months ago, and 3 months ago** should describe the three months immediately before the selected target month.  
The result is a monthly outbreak classification, not a population percentage or exact case count.
"""
        )

    if model is None:
        st.warning("Model file not found. Live prediction is unavailable until best_model.joblib is added.")
    else:
        if forecast is not None and not forecast.empty and {"Year", "Month"}.issubset(forecast.columns):
            year_options = sorted(pd.to_numeric(forecast["Year"], errors="coerce").dropna().astype(int).unique().tolist())
        else:
            max_year = int(pd.to_numeric(monthly.get("Year", pd.Series([2025])), errors="coerce").dropna().max()) if "Year" in monthly.columns else 2025
            year_options = list(range(max_year + 1, max_year + 6))
        month_options = list(range(1, 13))

        select_col1, select_col2 = st.columns(2)
        with select_col1:
            selected_year_num = st.selectbox("Select Year", year_options, index=0)
        with select_col2:
            selected_month_num = st.selectbox(
                "Select Month",
                month_options,
                format_func=lambda x: f"{x} - {month_name_from_number(x)}",
                index=0,
            )

        target_forecast_row = get_forecast_row(forecast, selected_year_num, selected_month_num)

        rainfall_default = numeric_from_row(target_forecast_row, "rainfall", get_profile_value(selected_month_num, "rainfall", month_profile, monthly, 0.0))
        humidity_default = numeric_from_row(target_forecast_row, "relative_humidity", get_profile_value(selected_month_num, "relative_humidity", month_profile, monthly, 0.0))
        temp_default = numeric_from_row(target_forecast_row, "temp_mid", get_profile_value(selected_month_num, "temp_mid", month_profile, monthly, 0.0))

        default_cases_lag_1 = numeric_from_row(target_forecast_row, "cases_lag_1", float(monthly["CHSO_cases"].iloc[-1]) if "CHSO_cases" in monthly.columns and len(monthly) >= 1 else 0.0)
        default_cases_lag_2 = numeric_from_row(target_forecast_row, "cases_lag_2", float(monthly["CHSO_cases"].iloc[-2]) if "CHSO_cases" in monthly.columns and len(monthly) >= 2 else default_cases_lag_1)
        default_cases_lag_3 = numeric_from_row(target_forecast_row, "cases_lag_3", float(monthly["CHSO_cases"].iloc[-3]) if "CHSO_cases" in monthly.columns and len(monthly) >= 3 else default_cases_lag_2)

        rain_min, rain_max = 0.0, max(1000.0, rainfall_default + 1.0)
        rh_min, rh_max = get_reasonable_range(monthly, "relative_humidity", 60.0, 100.0, [humidity_default])
        rh_min = min(0.0, rh_min)
        rh_max = max(100.0, rh_max)
        temp_min, temp_max = get_reasonable_range(monthly, "temp_mid", 10.0, 35.0, [temp_default])
        temp_min = min(10.0, temp_min)
        temp_max = max(35.0, temp_max)
        cases_min, cases_max = get_reasonable_range(monthly, "CHSO_cases", 0.0, 3000.0, [default_cases_lag_1, default_cases_lag_2, default_cases_lag_3])
        cases_min = 0.0
        cases_max = max(10.0, cases_max)

        st.markdown(f"### Inputs for {month_name_from_number(selected_month_num)} {selected_year_num}")
        climate_col1, climate_col2, climate_col3 = st.columns(3)
        with climate_col1:
            rainfall_now = st.slider(
                "Current Rainfall (mm)",
                min_value=float(round(rain_min, 2)),
                max_value=float(round(rain_max, 2)),
                value=float(round(clamp(rainfall_default, rain_min, rain_max), 2)),
                step=1.0,
            )
        with climate_col2:
            humidity_now = st.slider(
                "Current Relative Humidity (%)",
                min_value=float(round(rh_min, 2)),
                max_value=float(round(rh_max, 2)),
                value=float(round(clamp(humidity_default, rh_min, rh_max), 2)),
                step=0.1,
            )
        with climate_col3:
            temp_now = st.slider(
                "Current Temperature (°C)",
                min_value=float(round(temp_min, 2)),
                max_value=float(round(temp_max, 2)),
                value=float(round(clamp(temp_default, temp_min, temp_max), 2)),
                step=0.1,
            )

        st.markdown("### Recent Dengue Case History")
        case_col1, case_col2, case_col3 = st.columns(3)
        with case_col1:
            cases_lag_1 = st.slider(
                "Cases Last Month",
                min_value=int(cases_min),
                max_value=int(np.ceil(cases_max)),
                value=int(round(clamp(default_cases_lag_1, cases_min, cases_max))),
                step=1,
            )
        with case_col2:
            cases_lag_2 = st.slider(
                "Cases 2 Months Ago",
                min_value=int(cases_min),
                max_value=int(np.ceil(cases_max)),
                value=int(round(clamp(default_cases_lag_2, cases_min, cases_max))),
                step=1,
            )
        with case_col3:
            cases_lag_3 = st.slider(
                "Cases 3 Months Ago",
                min_value=int(cases_min),
                max_value=int(np.ceil(cases_max)),
                value=int(round(clamp(default_cases_lag_3, cases_min, cases_max))),
                step=1,
            )

        auto_feature_values = build_live_prediction_features(
            year_num=selected_year_num,
            month_num=selected_month_num,
            rainfall_now=rainfall_now,
            humidity_now=humidity_now,
            temp_now=temp_now,
            cases_lag_1=cases_lag_1,
            cases_lag_2=cases_lag_2,
            cases_lag_3=cases_lag_3,
            month_profile_df=month_profile,
            fallback_monthly_df=monthly,
            forecast_row=target_forecast_row,
        )

        input_values = {feature: auto_feature_values.get(feature, 0.0) for feature in feature_columns}

        with st.expander("Show automatically prepared model inputs"):
            st.dataframe(pd.DataFrame([input_values]), use_container_width=True)

        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame([input_values])
            try:
                pred = int(model.predict(input_df)[0])
                prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else np.nan

                result_col1, result_col2 = st.columns(2)
                result_col1.success(f"Predicted Class: {outbreak_label_from_binary(pred)}")
                if pd.isna(prob):
                    result_col2.info("Predicted Outbreak Probability: not available")
                else:
                    result_col2.info(f"Predicted Outbreak Probability: {prob:.4f}")

                st.markdown(
                    """
**How to read this result**  
**0** means non-outbreak month. **1** means outbreak month. The probability is the model's estimated likelihood that the selected month belongs to the outbreak class.
"""
                )

                st.subheader("Likely Highest-Risk Barangays")
                live_barangays = get_selected_barangay_ranking(
                    forecast_barangay_ranking,
                    forecast_top3_barangays,
                    selected_year_num,
                    selected_month_num,
                    prob,
                    input_values,
                )

                if live_barangays is not None and not live_barangays.empty:
                    keep_cols = [c for c in [
                        "Barangay", "overall_share", "recent_share", "seasonal_share", "risk_score_raw",
                        "risk_score", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy", "predicted_barangay_label",
                    ] if c in live_barangays.columns]
                    display_live = round_display_columns(
                        live_barangays[keep_cols],
                        ["overall_share", "recent_share", "seasonal_share", "risk_score_raw", "risk_score", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy"],
                        4,
                    )
                    st.dataframe(display_live, use_container_width=True)

                    fig_live_barangay = px.bar(
                        round_display_columns(live_barangays, ["predicted_barangay_cases_proxy"], 2),
                        x="Barangay",
                        y="predicted_barangay_cases_proxy",
                        color="Barangay",
                        text="predicted_barangay_cases_proxy",
                        title=f"Three Barangays with the Highest Predicted Risk for {month_name_from_number(selected_month_num)} {selected_year_num}",
                    )
                    fig_live_barangay.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    st.plotly_chart(fig_live_barangay, use_container_width=True)
                    st.caption("The barangay values are weighted proxy estimates, not confirmed future case counts.")
                else:
                    st.info("Barangay ranking is unavailable for this selected month. Add forecast_barangay_ranking.csv or forecast_top3_barangays.csv to the artifacts.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Check that meta.json feature_cols exactly match the features used to train best_model.joblib.")

st.markdown("---")
st.caption("Baguio City Dengue Forecast Dashboard")
