# Stock Price Prediction using LSTM, RNN, and CNN-Sliding-Window Models

A PyTorch reproduction of the paper **"Stock Price Prediction using LSTM, RNN and CNN-Sliding Window Model"** (Hiransha et al.), built as a Master's degree final project for the course *Intelligent Algorithms*.

Paper: https://ieeexplore.ieee.org/document/8126078

The project benchmarks three deep-learning architectures — RNN, LSTM, and a 1D-CNN applied over sliding windows of price data — plus a classical ARIMA baseline, on stock data (paper tickers: INFY, TCS.NS, CIPLA.NS; comparison tickers: NVDA, TSLA).

## Project Structure

    Stock-Prediction/
    ├── configs/               # Training hyperparameter configurations
    │   └── training_config.py
    ├── models/               # Model definitions
    │   ├── rnn/model.py      # Vanilla RNN
    │   ├── lstm/model.py     # LSTM
    │   ├── cnn/model.py      # 1D-CNN sliding-window model
    │   └── arima/model.py    # ARIMA statistical baseline
    ├── utils/
    │   ├── preprocessing/    # Data fetching, scaling, sliding windows
    │   │   ├── data_pipeline.py
    │   │   └── sliding_window.py
    │   ├── training/         # Generic training loop w/ early stopping
    │   │   └── trainer.py
    │   └── evaluation/       # Metrics, plots, result serialization
    │       └── evaluator.py
    ├── scripts/              # Entry points (see below)
    ├── tests/                # Smoke-test suite (pytest)
    ├── results/              # Output artifacts (generated; mostly gitignored)
    ├── requirements.txt
    └── environment.yml       # Pinned conda environment (original dev setup)

## Setup

Python 3.10+ is recommended.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Or with `uv`:

    uv venv --python 3.10 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

## Usage

Run the full train + evaluate pipeline (fetches recent intraday data via yfinance, trains all four models, saves results and prediction plots to `results/`):

    python scripts/train_and_evaluate.py

Alternatives:

    python scripts/train_models.py        # Train models, save checkpoints
    python scripts/train_all_models.py   # Train each model separately
    python scripts/check_cuda.py         # Verify PyTorch/CUDA availability

Run the tests:

    pytest tests/ -v

## How It Works

1. **Data** — `utils/preprocessing/data_pipeline.py` fetches recent price history via `yfinance` (paper stocks INFY, TCS.NS, CIPLA.NS, plus NVDA/TSLA for comparison).
2. **Windowing** — `sliding_window.py` slices the price series into overlapping windows (window size 100, overlap 90) and pairs each with the following 10-step horizon as the prediction target.
3. **Training** — `utils/training/trainer.py` provides a shared training loop (Adam optimizer, MSE loss, early stopping with configurable patience). Hyperparameters live in `configs/training_config.py`.
4. **Evaluation** — `utils/evaluation/evaluator.py` computes per-symbol error metrics, writes `results/evaluation_results.json`, and renders prediction plots.

## Results

Example output lives in `results/` (prediction plot: `results/predictions_INFY.png`). Checkpoints and per-run artifacts are generated during training and are not kept in the repo.

## Divergence from the Paper

This is not a strict reproduction. The original paper uses the NSE dataset over multi-year daily closing prices, whereas this project trains on only the last **5 days of 1-minute intraday data** fetched via yfinance — a shortcut chosen so runs complete quickly. Window/horizon parameters (window 100, horizon 10) are also approximations rather than the paper's exact setup. Treat the results as a demonstration of the architecture comparison, not a replication of the paper's figures. Adjust `StockDataPipeline` (start/end dates, interval) for longer histories.

## Notes

- The original development environment is captured in `environment.yml` (conda, Python 3.10, PyTorch 2.5.1 + CUDA 12.1). `requirements.txt` is the lightweight CPU-friendly install.
