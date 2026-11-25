# Depth of anesthesia forecasting

Code for *Calibrated Depth of Anesthesia Forecasting from BIS, EEG and Drug Histories: A Causal Recurrent Model with Episode-Level Alerts*.

The model forecasts the Bispectral Index (BIS) 30, 60 and 120 s ahead from the last 300 s of BIS, EEG spectral features and propofol/remifentanil infusion data. Separate convolutional encoders feed a two-layer GRU with forecast, band and quantile heads. The quantiles are conformally calibrated to 95% intervals, and training simulates sensor gaps.

## Data

Recordings come from the public [VitalDB](https://vitaldb.net) database and are downloaded by the code. `data/cohort_split.csv` lists the 4,501 cases (4,382 patients) with their patient-grouped partition; no patient is in more than one partition.

## Setup

```
pip install -r requirements.txt
```

Training needs a CUDA GPU with about 8 GB and about 16 GB of RAM. The environment variables `DOA_CACHE`, `DOA_ARRAYS` and `DOA_RUNS` change where data and results are stored.

## Steps

| Step | Command |
|---|---|
| 1. Download and build the dataset | `python scripts/build_dataset.py` |
| 2. Tune and train the networks | `python scripts/train_networks.py --stage main` |
| 3. Gradient boosting and persistence | `python scripts/train_xgboost.py --stage main` |
| 4. PK-PD model and classical regressors | `python scripts/train_classical.py` |
| 5. Cross-validation | `python scripts/train_networks.py --stage cv` and `python scripts/train_xgboost.py --stage cv` |
| 6. Accuracy, paired comparisons, phases | `python scripts/evaluate.py --stage main` and `python scripts/evaluate.py --stage cv` |
| 7. Forecasts on complete recordings | `python scripts/predict_streams.py` |
| 8. Prediction intervals | `python scripts/evaluate_intervals.py` |
| 9. Sensor gaps | `python scripts/evaluate_gaps.py` |
| 10. Alerts | `python scripts/evaluate_alerts.py` |
| 11. Replay test and CPU latency | `python scripts/replay_test.py` and `python scripts/latency.py` |

Trained networks are kept in `runs/` and skipped when a script is started again.

## Layout

- `doa/`: configuration, features, splits, models, training and metrics
- `scripts/`: data preparation, training and evaluation
- `data/`: cohort and partition

The code is for research on recorded data and is not intended for clinical use.
