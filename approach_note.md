# Approach Note — Day-Ahead Solar Generation Forecasting

## 1. Problem framing
The task is to predict hourly AC power exported by a single solar PV plant for 1,488 timestamps in July–August 2025. At inference time, only the timestamp and four day-ahead weather forecast fields are supplied. I therefore restrict model inputs to those weather forecasts and features derived from timestamp. Plant sensors and SCADA status are useful for diagnosing the historical data but are not available for the test horizon and are not model inputs.

## 2. Data inspection and quality handling
The dataset description gives 12,978 training rows from 2024-01-01 through 2025-06-30 and 1,488 hourly test rows from 2025-07-01 through 2025-08-31. Inspection identified:
- 389 training timestamps in `DD-MM-YYYY HH:MM` format mixed with ISO-like timestamps.
- 36 duplicated timestamps after normalizing formats.
- Missing training target and sensor values.
- `-999` values in several numeric fields, treated as invalid/missing sentinels.
- Some target readings above the documented 10,000 kW AC inverter rating, excluded as implausible for this target.
- Small negative AC readings, retained down to -100 kW because the target is net AC export and small auxiliary consumption may produce negative values.
- Missing values in several test weather fields.

The implementation parses both timestamp formats explicitly, replaces `-999` with missing values, aggregates duplicate timestamps using numeric medians, drops rows without a usable target, and excludes target values outside [-100, 10,000] kW. Test weather missingness is handled with a median imputer fitted on training features. This is a pragmatic treatment; the provenance note says the data is an unprocessed historian export, so further source-level validation would be valuable.

## 3. Features
The model uses:
- Forecast cloud cover, temperature, wind speed, and humidity.
- Hour of day, month, and day of year.
- Cyclical sine/cosine encodings for hour and day of year.
- A simple daytime indicator.

Timestamp features represent the strong diurnal and seasonal structure of solar generation. The model does not use measured plane-of-array irradiance, module temperature, measured ambient temperature, rainfall, SCADA status, or any target-derived lag because those values are not available for the test period. This avoids train/test feature mismatch and leakage.

## 4. Validation design
I used a chronological holdout rather than a random split. Data before 2025-05-01 is used to fit the model, and May–June 2025 is used for validation. This is closer to the operational use case than randomly mixing hours from the future and past. Metrics are MAE and RMSE in kW, plus R² for context.

The baseline is a month-of-year-by-hour climatology: mean training generation for each month/hour pair, with an overall-mean fallback if a group is absent. On the cleaned validation data, the baseline achieved MAE 317.62 kW and RMSE 809.84 kW. The selected ExtraTrees model achieved MAE 211.93 kW, RMSE 634.23 kW, and R² 0.9619 on the same validation window. These are validation results, not measured test-period performance.

## 5. Model selection
I compared the month-hour climatology baseline with ExtraTrees, HistGradientBoosting, and LightGBM using the same time-based split and weather/time feature set. Results:

| Model | Validation MAE (kW) | Validation RMSE (kW) | Validation R² |
|---|---:|---:|---:|
| ExtraTrees | 213.21 | 634.41 | 0.9619 |
| HistGradientBoosting | 247.59 | 657.36 | 0.9591 |
| LightGBM | 252.44 | 663.27 | 0.9584 |
| Month-hour climatology | 317.62 | 809.84 | 0.9379 |

ExtraTrees was selected because it had the lowest validation MAE and RMSE among the tested candidates. The final model is refit on all cleaned historical training rows before producing test predictions. Random seeds and model parameters are fixed in the code.

## 6. Weaknesses and error analysis
The largest validation errors are concentrated in daylight hours, when generation can change sharply or the plant can be unavailable. One severe example is 2025-05-14 around 10:00–14:00: the model predicted several thousand kW while measured output was near zero. Weather and time features alone cannot reliably infer an unannounced plant outage or curtailment. Conversely, rapidly changing cloud conditions can also produce large misses. Nighttime validation errors were much smaller than daytime errors, so aggregate metrics hide the more operationally important daytime weakness.

The model is also limited by the short history (about 18 months), noisy historian records, missing weather values, and imperfect day-ahead forecasts. The validation window is May–June while the test period is July–August, so performance may shift during the monsoon season. The test labels are unavailable and no test performance claim is made.

## 7. What I would do next
1. Verify the `-999` sentinel and duplicate-record semantics with the data owner; consider robust rules for known SCADA fault/outage periods.
2. Evaluate a rolling-origin validation across multiple months, including monsoon-like periods, rather than relying on one two-month holdout.
3. Compare against a solar-geometry/clear-sky baseline and a model that explicitly represents sunrise/sunset.
4. Quantify errors separately for daylight hours, cloudy periods, and high-output periods; consider MAE and normalized plant-capacity error as complementary operational metrics.
5. If operational forecasts are intended to account for outages or curtailment, incorporate only information that would genuinely be available at forecast issuance time.
6. Calibrate prediction intervals so grid operators can understand uncertainty, not just a point forecast.

## 8. Reproducibility and submission checks
Run `python src/train_predict.py --data-dir data --output predictions.csv`. The script prints validation metrics, writes a validation report, fits the selected model on all cleaned historical data, and writes the required two-column CSV. It asserts that there are exactly 1,488 predictions, that the header is correct, that timestamps preserve the supplied test order and values, and that predictions are non-missing.
