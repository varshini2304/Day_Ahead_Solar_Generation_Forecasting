from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_utils import (
    PROJECT_ROOT,
    build_validation_frame,
    data_quality_summary,
    load_predictions,
    load_training_data,
    read_validation_metrics,
    run_forecast_pipeline,
)

st.set_page_config(page_title="Ellume Solar Forecast Dashboard", page_icon="☀️", layout="wide")

with open(Path(__file__).resolve().parent / "README.md", "r", encoding="utf-8") as fh:
    _ = fh.read()

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #071421 0%, #0d1f2a 100%); color: #edf6ff; }
        .block-container { padding-top: 2rem; }
        h1, h2, h3 { color: #dff9ff; }
        [data-testid="stMetricValue"] { color: #7fe7da; }
        div[data-testid="stSidebar"] { background: #091d2b; }
        .stTabs [role="tablist"] button { color: #dff9ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ellume Solar Forecast Intelligence")
st.caption("Day-ahead solar generation forecast dashboard using the existing forecasting pipeline.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a view",
        ["Overview", "Forecast Explorer", "Model Performance", "Data Quality", "Generate Forecast"],
        index=0,
    )
    st.markdown("---")
    st.caption("This dashboard reads the repository’s trained model output and does not replace the underlying forecasting logic.")


def show_metric_card(title: str, value: str, delta: str | None = None) -> None:
    st.metric(title, value, delta=delta or "")


if page == "Overview":
    validation_metrics = read_validation_metrics()
    forecast_df = load_predictions()
    cleaned = load_training_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical rows", f"{validation_metrics.get('cleaned_training_rows', len(cleaned)):,}")
    col2.metric("Validation MAE", f"{validation_metrics.get('model_mae_kw', 0):.2f} kW")
    col3.metric("Validation RMSE", f"{validation_metrics.get('model_rmse_kw', 0):.2f} kW")
    col4.metric("Forecast horizon", f"{len(forecast_df):,} hrs")

    st.markdown("### Forecast overview")
    vdf = build_validation_frame()
    fig = px.line(
        vdf,
        x="timestamp_parsed",
        y=["actual_kw", "predicted_kw"],
        title="Validation period: actual vs forecasted AC output",
        labels={"value": "AC power (kW)", "timestamp_parsed": "Timestamp"},
    )
    fig.update_layout(template="plotly_dark", legend_title_text="Series")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Forecast range")
    start = pd.Timestamp(forecast_df["timestamp"].min())
    end = pd.Timestamp(forecast_df["timestamp"].max())
    st.write(f"Forecast covers {start:%Y-%m-%d %H:%M} through {end:%Y-%m-%d %H:%M} in the published output file.")

if page == "Forecast Explorer":
    forecast_df = load_predictions()
    min_dt = forecast_df["timestamp"].min()
    max_dt = forecast_df["timestamp"].max()
    start_dt, end_dt = st.slider(
        "Select forecast window",
        min_value=min_dt.to_pydatetime(),
        max_value=max_dt.to_pydatetime(),
        value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
        step=pd.Timedelta(hours=1).to_pytimedelta(),
    )
    selected = forecast_df[(forecast_df["timestamp"] >= pd.Timestamp(start_dt)) & (forecast_df["timestamp"] <= pd.Timestamp(end_dt))]

    fig = px.line(
        selected,
        x="timestamp",
        y="predicted_ac_power_kw",
        title="Published day-ahead forecast output",
        labels={"timestamp": "Timestamp", "predicted_ac_power_kw": "Predicted AC power (kW)"},
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(selected.head(12), use_container_width=True)

if page == "Model Performance":
    validation_metrics = read_validation_metrics()
    vdf = build_validation_frame()

    metric_cols = st.columns(4)
    metric_cols[0].metric("MAE", f"{validation_metrics.get('model_mae_kw', 0):.2f} kW")
    metric_cols[1].metric("RMSE", f"{validation_metrics.get('model_rmse_kw', 0):.2f} kW")
    metric_cols[2].metric("R²", f"{validation_metrics.get('model_r2', 0):.4f}")
    metric_cols[3].metric("Baseline MAE", f"{validation_metrics.get('baseline_mae_kw', 0):.2f} kW")

    scatter = px.scatter(
        vdf,
        x="actual_kw",
        y="predicted_kw",
        opacity=0.65,
        title="Actual vs predicted output in the validation window",
        labels={"actual_kw": "Actual AC power (kW)", "predicted_kw": "Predicted AC power (kW)"},
    )
    scatter.add_shape(type="line", x0=vdf["actual_kw"].min(), x1=vdf["actual_kw"].max(), y0=vdf["actual_kw"].min(), y1=vdf["actual_kw"].max(), line={"dash": "dash", "color": "#7fe7da"})
    scatter.update_layout(template="plotly_dark")
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown("### Validation summary")
    st.write(
        "The model is evaluated on a chronological holdout from 2025-05-01 to 2025-06-30, mirroring a true day-ahead forecasting use case without random leakage."
    )

if page == "Data Quality":
    quality = data_quality_summary()
    st.subheader("Input quality checks")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{quality['rows']:,}")
    cols[1].metric("Missing values", f"{quality['missing_values']:,}")
    cols[2].metric("Duplicate timestamps", f"{quality['duplicate_timestamps']:,}")
    cols[3].metric("Parseable timestamps", f"{quality['parseable_timestamps']:,}")

    train_df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
    quality_df = pd.DataFrame({
        "Column": train_df.columns,
        "Missing": train_df.isna().sum().values,
    })
    st.dataframe(quality_df.sort_values("Missing", ascending=False).head(12), use_container_width=True)

    st.info("The pipeline intentionally handles mixed timestamp formats, duplicate timestamps, -999 sentinels, and invalid target values before model training.")

if page == "Generate Forecast":
    st.subheader("Refresh the published predictions file")
    st.info("This action reruns the exact repository forecasting script and rewrites predictions.csv with the validated output schema.")

    if st.button("Run forecasting script", type="primary"):
        with st.spinner("Generating predictions..."):
            result = run_forecast_pipeline()
        st.success(f"Forecast file updated successfully: {result['rows']} rows written.")
        st.code(result["stdout"], language="text")

        output_df = load_predictions()
        st.dataframe(output_df.head(10), use_container_width=True)

st.markdown("---")
st.caption("Note: July–August 2025 values are historical forecast output for the validation/test period and are not live plant telemetry.")
