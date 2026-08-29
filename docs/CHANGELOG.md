# Development Log

## 2026-08-28 — Repo cleanup

- Fixed ARIMA model for modern statsmodels: `ValueError: No supported index is available` when forecasting on a differenced series whose index started at 1. Fixed by resetting the index after `diff().dropna()` in both `fit()` and `predict_sequences()`.
- Deleted ~2 GB of stale artifacts: 299 `.pt` checkpoints in `results/checkpoints/` and `tests/results/checkpoints/`, training logs, dated result JSONs, `stock_prediction.egg-info/`, all `__pycache__/`.
- Moved root-level `test.py` (a CUDA environment check) to `scripts/check_cuda.py` so pytest doesn't collect it as a broken test module.
- Populated previously-empty `requirements.txt`.
- Added `.gitignore`, rewrote `README.md`, added `docs/ARCHITECTURE.md`.
- Verified: all 8 tests pass (`pytest tests/ -q` → `8 passed`) under Python 3.10 venv with CPU PyTorch; all modules import cleanly.
