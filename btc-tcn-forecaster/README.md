# BTC TCN Forecaster

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Two complementary TCN-based forecasting models for BTC/USDT (1-hour bars):

- **Direction model** — binary classification of next-bar return sign, evaluated by ROC-AUC
- **Volatility model** — regression of next-bar absolute return magnitude, benchmarked against GARCH(1,1)

Both use a **Temporal Convolutional Network** (TCN) with dilated causal convolutions, trained on a fixed 80/20 time-ordered split with a `StandardScaler` fitted exclusively on training data.

---

## Motivation

TCNs offer a middle ground between RNNs and transformers for sequential financial data:

- **Causal convolutions** guarantee no look-ahead within the sequence
- **Dilated layers** exponentially expand the receptive field without depth, enabling the model to capture both short-term (tick-level) and medium-term (multi-day) autocorrelation structure efficiently
- **Parallelism** — unlike LSTMs, the entire sequence is processed in one forward pass, making training and inference fast

The two tasks are deliberately separated: direction and magnitude behave differently statistically, and training separate models with task-appropriate losses (BCE for direction, L1 for volatility) avoids conflating them.

---

## Method

### Direction Model

**Target**: `sign(r_{t+1}) > 0` — binary, next-bar return direction.

**Features** (8 inputs per timestep):

| Feature | Description |
|---|---|
| `return` | Log return: `log(close_t / close_{t-1})` |
| `log_volume` | `log(volume + 1)` |
| `log_volume_delta` | `log(volume + 1) - log(taker_buy_volume + 1)` — sell-side proxy |
| `realized_vol` | Rolling 10-bar return std dev |
| `autocorr_5` | Lag-1 autocorrelation over 5-bar window |
| `autocorr_10` | Lag-1 autocorrelation over 10-bar window |
| `cum_log_volume_delta_10` | Cumulative 10-bar log volume delta |
| `cum_log_volume_10` | Cumulative 10-bar log volume |

**Loss**: `BCEWithLogitsLoss`. **Metric**: ROC-AUC on held-out 20%.

### Volatility Model

**Target**: `|r_{t+1}|` — next-bar absolute return magnitude.

**Features** (13 inputs per timestep):

| Feature | Description |
|---|---|
| `return` | Absolute log return |
| `log_volume` | Absolute log volume |
| `realized_vol_{5,10,20,30,60}` | Rolling std dev at 5 horizons |
| `mean_abs_return_{5,10,20,30,60,120}` | Rolling mean absolute return at 6 horizons |

**Loss**: `L1Loss` (MAE) — robust to the fat tails typical in volatility targets. **Metrics**: RMSE, MAE, R² vs GARCH(1,1).

**GARCH benchmark**: GARCH(1,1) with zero-mean specification, fit on training returns, forecasting the full test horizon in one call via `arch`.

### Architecture

Both models share the same `TCNModel` class:

```
Input (B, T, F) → transpose → (B, F, T)
TCN: [num_channels] × num_layers dilated causal conv blocks
Output head: Linear(num_channels → 1) applied at last timestep
```

Sequences are constructed with a sliding window of length `seq_length`; the label for each window is the value at the final timestep.

---

## Quickstart

```bash
git clone https://github.com/<your-handle>/btc-tcn-forecaster.git
cd btc-tcn-forecaster
pip install -r requirements.txt
```

Place `BTCUSDT_1h.csv` in `data/raw/`. Then:

```bash
python scripts/train_direction.py
python scripts/train_volatility.py
```

---

## Results

### Direction (ROC-AUC vs 0.5 baseline)

| Epoch | ROC-AUC |
|---|---|
| 10 | — |
| 20 | — |
| 30 | — |
| 40 | — |
| 50 | — |

### Volatility (RMSE / MAE / R²)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| TCN | — | — | — |
| GARCH(1,1) | — | — | — |

> Fill in after running scripts. A ROC-AUC above 0.5 indicates directional edge; a lower RMSE than GARCH indicates the TCN captures volatility clustering not explained by the classical model.

---

## Project Structure

```
src/features.py        — feature construction for both models
src/model.py           — shared TCNModel architecture
src/datasets.py        — sequence construction and DataLoader helpers
src/utils.py           — metrics, plotting, GARCH benchmark
config.py              — all hyperparameters in one place
scripts/
  train_direction.py   — direction model training and evaluation
  train_volatility.py  — volatility model training, evaluation, GARCH comparison
tests/
  test_features.py     — unit tests for feature construction
  test_model.py        — unit tests for model forward pass and sequence shapes
```

---

## Dependencies

See `requirements.txt`. Core: `pytorch-tcn`, `arch`, `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`.

---

## License

MIT
