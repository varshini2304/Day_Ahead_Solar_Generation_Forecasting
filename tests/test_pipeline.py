from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_predict


def test_timestamp_parser_handles_mixed_formats():
    values = pd.Series(["2024-01-01 00:00:00", "02-01-2024 16:00"])
    parsed = train_predict.parse_timestamps(values)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert parsed.iloc[1] == pd.Timestamp("2024-01-02 16:00:00")


def test_cleaned_training_has_unique_timestamps_and_valid_targets():
    raw = pd.read_csv(ROOT / "data" / "train.csv")
    clean = train_predict.clean_and_deduplicate_train(raw)
    assert clean["dt"].is_unique
    assert clean[train_predict.TARGET].notna().all()
    assert clean[train_predict.TARGET].between(-100, 10_000).all()


def test_predictions_format():
    pred = pd.read_csv(ROOT / "predictions.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    assert list(pred.columns) == ["timestamp", "predicted_ac_power_kw"]
    assert len(pred) == len(test) == 1488
    assert pred["timestamp"].equals(test["timestamp"])
    assert pred["predicted_ac_power_kw"].notna().all()
