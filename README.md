# Ellume AI Engineer Take-Home — Day-Ahead Solar Generation Forecasting

## Goal
Forecast hourly AC power export (kW) for the supplied test timestamps using only information available at day-ahead forecast time.

## Files
- `data/train.csv`: historical forecast weather and plant outcomes.
- `data/test.csv`: forecast weather for the period to predict.
- `src/train_predict.py`: reproducible preprocessing, chronological validation, training, and prediction.
- `predictions.csv`: generated test predictions.
- `approach_note.md`: method, validation, limitations, and next steps.
- `validation_report.txt`: run-specific validation metrics and largest errors.

## Run
Use Python 3.10+ in a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/train_predict.py --data-dir data --output predictions.csv
```

The script prints validation metrics and writes `validation_report.txt`.

## Modeling decisions
Only day-ahead weather forecast fields and timestamp-derived features are used. SCADA status, measured irradiance, measured temperatures, rainfall, and target history are not used as model inputs because these are unavailable for the test period. A chronological May–June 2025 validation window is used to mimic prediction into a future period.

## Data note
The supplied training data contains mixed timestamp formats, duplicate timestamps, missing values, and `-999` sentinels. The script parses both timestamp formats, aggregates duplicate timestamps by numeric median, treats `-999` as missing, and removes unusable or implausible target values before training. The original source files are retained unchanged under `data/`.
