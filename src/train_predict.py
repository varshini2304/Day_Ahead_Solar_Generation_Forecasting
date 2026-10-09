from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline


WEATHER_COLUMNS = [
    "forecast_cloud_cover",
    "forecast_temp_c",
    "forecast_wind_ms",
    "forecast_humidity_pct",
]
TARGET = "ac_power_kw"
AC_CAPACITY_KW = 10_000


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse ISO timestamps and the alternate DD-MM-YYYY HH:MM format."""
    raw = values.astype(str)
    alternate = raw.str.match(r"^\d{2}-\d{2}-\d{4}")
    parsed = pd.to_datetime(raw, format="mixed", errors="coerce")
    parsed.loc[alternate] = pd.to_datetime(
        raw.loc[alternate], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    if parsed.isna().any():
        bad = raw.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Could not parse timestamps; examples: {bad}")
    return parsed


def clean_and_deduplicate_train(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    train["dt"] = parse_timestamps(train["timestamp"])

    # The data contains -999 sentinels in numeric columns; treat them as missing.
    numeric_columns = train.select_dtypes(include="number").columns
    train.loc[:, numeric_columns] = train.loc[:, numeric_columns].replace(-999, np.nan)

    # Multiple rows can share a timestamp. Aggregate numeric fields using the median.
    # Do not use SCADA status or measured sensors as model features because they are
    # unavailable for the test period.
    aggregation = {col: "median" for col in numeric_columns}
    aggregation["timestamp"] = "first"
    train = train.groupby("dt", as_index=False).agg(aggregation)

    # Drop rows without a usable target and physically implausible AC values.
    # Small negative values are retained because the target is net AC export and
    # can reflect auxiliary consumption; values below -100 kW are treated as bad data.
    train = train[
        train[TARGET].notna()
        & train[TARGET].between(-100, AC_CAPACITY_KW, inclusive="both")
    ].copy()
    return train.sort_values("dt").reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Only use forecast weather and timestamp features available at forecast time."""
    dt = pd.to_datetime(frame["dt"])
    features = frame[WEATHER_COLUMNS].copy()
    features["hour"] = dt.dt.hour
    features["month"] = dt.dt.month
    features["dayofyear"] = dt.dt.dayofyear
    features["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    features["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    features["is_daytime"] = ((dt.dt.hour >= 6) & (dt.dt.hour <= 18)).astype(int)
    return features


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=3,
            max_features=0.9,
            n_jobs=-1,
            random_state=42,
        ),
    )


def evaluate(train: pd.DataFrame) -> dict:
    """Chronological holdout: train before May 2025, validate May–June 2025."""
    features = make_features(train)
    train_mask = train["dt"] < pd.Timestamp("2025-05-01")
    valid_mask = (
        (train["dt"] >= pd.Timestamp("2025-05-01"))
        & (train["dt"] < pd.Timestamp("2025-07-01"))
    )
    X_train, y_train = features.loc[train_mask], train.loc[train_mask, TARGET]
    X_valid, y_valid = features.loc[valid_mask], train.loc[valid_mask, TARGET]

    # Simple month-hour climatology baseline, fitted only on the training window.
    baseline_frame = features.loc[train_mask, ["month", "hour"]].copy()
    baseline_frame["target"] = y_train.to_numpy()
    baseline = baseline_frame.groupby(["month", "hour"])["target"].mean()
    baseline_pred = np.array([
        baseline.get((m, h), y_train.mean())
        for m, h in zip(features.loc[valid_mask, "month"], features.loc[valid_mask, "hour"])
    ])

    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    return {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "baseline_mae_kw": float(mean_absolute_error(y_valid, baseline_pred)),
        "baseline_rmse_kw": float(mean_squared_error(y_valid, baseline_pred) ** 0.5),
        "model_mae_kw": float(mean_absolute_error(y_valid, pred)),
        "model_rmse_kw": float(mean_squared_error(y_valid, pred) ** 0.5),
        "model_r2": float(r2_score(y_valid, pred)),
        "validation_predictions": pd.DataFrame({
            "timestamp": train.loc[valid_mask, "timestamp"].to_numpy(),
            "timestamp_parsed": train.loc[valid_mask, "dt"].to_numpy(),
            "actual_kw": y_valid.to_numpy(),
            "predicted_kw": pred,
            "absolute_error_kw": np.abs(y_valid.to_numpy() - pred),
        }),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--validation-report", type=Path, default=Path("validation_report.txt"))
    args = parser.parse_args()

    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    original_test_timestamps = test["timestamp"].copy()
    test["dt"] = parse_timestamps(test["timestamp"])

    train = clean_and_deduplicate_train(train_raw)
    validation = evaluate(train)
    print(f"Cleaned training rows: {len(train)}")
    print(f"Validation rows: {validation['validation_rows']}")
    print(f"Month-hour baseline MAE: {validation['baseline_mae_kw']:.2f} kW")
    print(f"ExtraTrees MAE: {validation['model_mae_kw']:.2f} kW")
    print(f"ExtraTrees RMSE: {validation['model_rmse_kw']:.2f} kW")
    print(f"ExtraTrees R2: {validation['model_r2']:.4f}")

    # Train final model on all cleaned historical target rows.
    final_model = make_model()
    final_model.fit(make_features(train), train[TARGET])
    predictions = final_model.predict(make_features(test))
    predictions = np.clip(predictions, -100, AC_CAPACITY_KW)

    output = pd.DataFrame({
        "timestamp": original_test_timestamps,
        "predicted_ac_power_kw": predictions,
    })
    output.to_csv(args.output, index=False)

    report = (
        "Ellume day-ahead generation forecasting validation\n"
        f"Cleaned training rows: {len(train)}\n"
        f"Validation rows: {validation['validation_rows']}\n"
        "Validation window: 2025-05-01 through 2025-06-30\n"
        "Split: chronological; no random shuffling\n"
        f"Month-hour baseline MAE (kW): {validation['baseline_mae_kw']:.4f}\n"
        f"Month-hour baseline RMSE (kW): {validation['baseline_rmse_kw']:.4f}\n"
        f"ExtraTrees MAE (kW): {validation['model_mae_kw']:.4f}\n"
        f"ExtraTrees RMSE (kW): {validation['model_rmse_kw']:.4f}\n"
        f"ExtraTrees R2: {validation['model_r2']:.6f}\n"
    )
    worst = validation["validation_predictions"].nlargest(5, "absolute_error_kw")
    report += "\nFive largest absolute validation errors:\n" + worst.to_string(index=False) + "\n"
    args.validation_report.write_text(report, encoding="utf-8")

    # Strict submission checks.
    assert list(output.columns) == ["timestamp", "predicted_ac_power_kw"]
    assert len(output) == len(test) == 1488
    assert output["timestamp"].equals(original_test_timestamps)
    assert output["predicted_ac_power_kw"].notna().all()
    print(f"Wrote {args.output} with {len(output)} rows.")


if __name__ == "__main__":
    main()
