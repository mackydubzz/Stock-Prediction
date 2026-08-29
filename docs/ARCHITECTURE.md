# Architecture Notes

This document explains how the pieces fit together for anyone revisiting the project.

## Data Flow

    yfinance API
        │
        ▼
    StockDataPipeline (utils/preprocessing/data_pipeline.py)
        │  fetch → clean → normalize → split train/test
        ▼
    sliding_window (utils/preprocessing/sliding_window.py)
        │  window_size=100, overlap=90, horizon=10
        │  produces X: (N, 100) inputs, y: (N, 10) targets
        ▼
    ModelTrainer (utils/training/trainer.py)  ── shared loop: Adam, MSE, early stopping
        │
        ▼
    ModelEvaluator (utils/evaluation/evaluator.py)
           metrics + plots + JSON results in results/

## Models

| Model   | File                    | Notes                                            |
|---------|-------------------------|--------------------------------------------------|
| RNN     | `models/rnn/model.py`   | 2-layer vanilla RNN, hidden 64                   |
| LSTM    | `models/lstm/model.py`  | 2-layer LSTM, hidden 64                          |
| CNN     | `models/cnn/model.py`   | 1D conv stack [32, 64, 128] over sliding windows |
| ARIMA   | `models/arima/model.py`| Classical baseline, (p, d, q) default (2, 1, 0)  |

All deep models take a window of 100 prices and predict the next 10.

## Entry-point scripts

- `scripts/train_and_evaluate.py` — one-shot pipeline: fetch data, train RNN/LSTM/CNN/ARIMA, evaluate, plot, save.
- `scripts/train_models.py` — training-focused entry point with checkpoint saving per epoch.
- `scripts/train_all_models.py` — trains each architecture in its own checkpoint dir.
- `scripts/check_cuda.py` — environment sanity check for GPU training.

## Gotchas learned during the 2026 cleanup

- **ARIMA + statsmodels**: differenced series must have a reset (`dropna()` leaves an index starting at 1, which modern statsmodels rejects with `ValueError: No supported index is available`). `model.py` now calls `.reset_index(drop=True)` after `.diff().dropna()`.
- **Relative paths**: scripts assume the working directory is the repo root (`Stock-Prediction/`); `results/` is created on demand.
- `setup.py` exists but the project is used as a source tree, not an installed package.
