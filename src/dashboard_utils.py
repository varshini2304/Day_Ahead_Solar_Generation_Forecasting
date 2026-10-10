from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import train_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_PATH = PROJECT_ROOT / "predictions.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "validation_report.txt"


def read_validation_metrics(report_path: Path = VALIDATION_REPORT_PATH) -> dict[str, float | int | str]:
    """Read metrics from the validation report text file."""
    metrics: dict[str, float | int | str] = {}
    if not report_path.exists():
        raise FileNotFoundError(f"Validation report not found: {report_path}")

    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Cleaned training rows:"):
            metrics["cleaned_training_rows"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Validation rows:"):
            metrics["validation_rows"] = int(line.split(":", 1)[1].strip())
        elif "baseline MAE" in line.lower():
            metrics["baseline_mae_kw"] = float(line.split(":", 1)[1].strip().replace("kW", "").replace("(", "").replace(")", ""))
        elif "baseline RMSE" in line.lower():
            metrics["baseline_rmse_kw"] = float(line.split(":", 1)[1].strip().replace("kW", "").replace("(", "").replace(")", ""))
        elif "ExtraTrees MAE" in line:
            metrics["model_mae_kw"] = float(line.split(":", 1)[1].strip().replace("kW", "").replace("(", "").replace(")", ""))
        elif "ExtraTrees RMSE" in line:
            metrics["model_rmse_kw"] = float(line.split(":", 1)[1].strip().replace("kW", "").replace("(", "").replace(")", ""))
        elif "ExtraTrees R2" in line:
            metrics["model_r2"] = float(line.split(":", 1)[1].strip())
    return metrics


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the output CSV contract for prediction files."""
    required = ["timestamp", "predicted_ac_power_kw"]
    if list(frame.columns) != required:
        raise ValueError(f"Predictions must have columns {required}, found {list(frame.columns)}")
    if frame.empty:
        raise ValueError("Predictions are empty.")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Prediction timestamps contain invalid values.")
    if frame["predicted_ac_power_kw"].isna().any():
        raise ValueError("Predicted values contain missing values.")
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load the forecast output and validate it before display."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return validate_prediction_frame(pd.read_csv(path))


def load_training_data(path: Path = DATA_DIR / "train.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    raw = pd.read_csv(path)
    if "timestamp" not in raw.columns or "ac_power_kw" not in raw.columns:
        raise ValueError("Training data is missing expected columns.")
    return train_predict.clean_and_deduplicate_train(raw)


def build_validation_frame() -> pd.DataFrame:
    cleaned = load_training_data()
    return train_predict.evaluate(cleaned)["validation_predictions"]


def data_quality_summary(raw_train_path: Path = DATA_DIR / "train.csv") -> dict[str, int | float]:
    raw = pd.read_csv(raw_train_path)
    missing_values = int(raw.isna().sum().sum())
    duplicate_timestamps = int(raw["timestamp"].duplicated().sum())
    parseable_timestamps = int(pd.to_datetime(raw["timestamp"], errors="coerce").notna().sum())
    return {
        "rows": int(len(raw)),
        "missing_values": missing_values,
        "duplicate_timestamps": duplicate_timestamps,
        "parseable_timestamps": parseable_timestamps,
    }


def run_forecast_pipeline() -> dict[str, str | int]:
    """Run the existing forecasting pipeline and return file stats."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train_predict.py"),
        "--data-dir",
        str(DATA_DIR),
        "--output",
        str(PREDICTIONS_PATH),
        "--validation-report",
        str(VALIDATION_REPORT_PATH),
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
    output_df = load_predictions(PREDICTIONS_PATH)
    return {
        "rows": int(len(output_df)),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "path": str(PREDICTIONS_PATH),
    }
