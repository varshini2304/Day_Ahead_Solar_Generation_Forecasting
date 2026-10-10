#  Day-Ahead Solar Generation Forecasting

## Goal
Forecast hourly AC power export (kW) for the supplied test timestamps using only information available at day-ahead forecast time.

## Project structure
- `data/train.csv`: historical forecast weather and plant outcomes used to train and validate the model.
- `data/test.csv`: day-ahead forecast weather for the target production window.
- `src/train_predict.py`: reproducible preprocessing, time-based validation, model training, and forecast generation.
- `src/dashboard_utils.py`: helper logic used by the Streamlit app for reading metrics, validating artifacts, and rerunning the forecast pipeline.
- `app.py`: interactive dashboard entry point for exploring the forecast and validation outputs.
- `predictions.csv`: generated forecast artifact that follows the required two-column schema.
- `validation_report.txt`: run-specific validation metrics and the largest absolute errors.
- `README.md`: project overview and usage notes.

## Forecasting methodology
This project keeps the original forecasting pipeline as the source of truth:

- Only day-ahead forecast weather and timestamp-derived features are used.
- Measured SCADA values, measured irradiance, measured temperature, rainfall, and target history are intentionally excluded from the model input set because they are unavailable at forecast time.
- Chronological validation is used: rows before 2025-05-01 are trained on, and May–June 2025 is withheld as a validation window.
- The model is `ExtraTreesRegressor` with median imputation and a day-ahead, no-leakage feature design.

## Data handling and constraints
The supplied data includes mixed timestamp formats, duplicate timestamps, missing values, and `-999` sentinels. The pipeline keeps the original source files under `data/` unchanged and applies the following cleaning logic before training:

- parse ISO and `DD-MM-YYYY HH:MM` timestamps
- treat `-999` as missing
- aggregate duplicate timestamps by median for numeric fields
- drop unusable or physically implausible AC output values
- keep the final output contract exactly as `timestamp,predicted_ac_power_kw`

## Run the forecast pipeline
Use Python 3.11+ in a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/train_predict.py --data-dir data --output predictions.csv
```

The script prints validation metrics and writes both `predictions.csv` and `validation_report.txt`.

## Launch the dashboard locally
The repository includes a Streamlit dashboard that reads the real project artifacts rather than duplicating the forecasting logic.

```bash
.venv\Scripts\activate
streamlit run app.py
```

The dashboard provides:
- an overview of the validation and forecast horizon
- a forecast explorer with date filtering
- validation metrics and actual-vs-predicted plots
- data-quality checks for the training data
- a button to rerun the repository pipeline and refresh the published forecast file

## Dashboard notes
- The dashboard reads the generated prediction CSV and validation report from the repository.
- It does not overwrite raw source datasets in `data/`.
- The July–August 2025 values shown are historical forecast output for the test period, not live plant telemetry.
- The app is designed to fail gracefully on missing or malformed files instead of exposing stack traces to users.

## Validation snapshot
The current validation report shows the ExtraTrees model outperforming the month-hour baseline while preserving the project constraints:

- baseline MAE: approx. 317.62 kW
- baseline RMSE: approx. 809.84 kW
- ExtraTrees MAE: approx. 213.21 kW
- ExtraTrees RMSE: approx. 634.41 kW
- ExtraTrees R²: approx. 0.9619

## Reproducibility and submission checks
The project has strict output checks to protect the submission contract:

- the final CSV must contain exactly two columns: `timestamp` and `predicted_ac_power_kw`
- output length must match the provided test horizon
- timestamps are preserved in order and are not fabricated
- predictions are clipped to a sensible AC-power range and non-null values are required

## Notes and limitations
This solution is intentionally conservative and aligned to the supplied take-home task. It preserves the existing pipeline and adds a lightweight dashboard layer on top of real artifacts rather than rewriting the methodology or altering the data contract.
