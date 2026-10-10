from pathlib import Path

import pandas as pd
import pytest

from src import dashboard_utils

ROOT = Path(__file__).resolve().parents[1]


def test_prediction_csv_contract_is_valid():
    df = pd.read_csv(ROOT / "predictions.csv")
    validated = dashboard_utils.validate_prediction_frame(df)
    assert list(validated.columns) == ["timestamp", "predicted_ac_power_kw"]
    assert validated["predicted_ac_power_kw"].notna().all()


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        dashboard_utils.load_predictions(ROOT / "missing_predictions.csv")


def test_validation_metrics_are_parsed():
    metrics = dashboard_utils.read_validation_metrics(ROOT / "validation_report.txt")
    assert metrics["cleaned_training_rows"] > 0
    assert "model_mae_kw" in metrics
    assert "model_r2" in metrics
