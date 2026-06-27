import numpy as np
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import TCNModel
from src.datasets import make_sequences, fit_scaler_and_scale, build_loaders
from config import DIRECTION_PARAMS, VOLATILITY_PARAMS, DIRECTION_NUM_INPUTS, VOLATILITY_NUM_INPUTS


def _random_features(n: int, num_features: int) -> np.ndarray:
    np.random.seed(42)
    return np.random.randn(n, num_features).astype(np.float32)


def test_direction_model_output_shape():
    model = TCNModel(
        num_inputs=DIRECTION_NUM_INPUTS,
        num_channels=32,
        num_layers=2,
        kernel_size=3,
        dropout=0.0,
    )
    x = torch.randn(16, DIRECTION_PARAMS['seq_length'], DIRECTION_NUM_INPUTS)
    out = model(x)
    assert out.shape == (16,)


def test_volatility_model_output_shape():
    model = TCNModel(
        num_inputs=VOLATILITY_NUM_INPUTS,
        num_channels=32,
        num_layers=2,
        kernel_size=5,
        dropout=0.0,
    )
    x = torch.randn(16, VOLATILITY_PARAMS['seq_length'], VOLATILITY_NUM_INPUTS)
    out = model(x)
    assert out.shape == (16,)


def test_model_no_nan_output():
    model = TCNModel(
        num_inputs=DIRECTION_NUM_INPUTS,
        num_channels=32,
        num_layers=2,
        kernel_size=3,
        dropout=0.0,
    )
    x = torch.randn(8, DIRECTION_PARAMS['seq_length'], DIRECTION_NUM_INPUTS)
    out = model(x)
    assert not torch.any(torch.isnan(out))


def test_make_sequences_shapes():
    n, f = 300, DIRECTION_NUM_INPUTS
    seq_length = DIRECTION_PARAMS['seq_length']
    features = _random_features(n, f)
    targets = np.random.randint(0, 2, n).astype(np.float32)
    X_train, y_train, X_test, y_test = make_sequences(features, targets, seq_length)
    total = n - seq_length
    split = int(total * 0.8)
    assert X_train.shape == (split, seq_length, f)
    assert y_train.shape == (split,)
    assert X_test.shape == (total - split, seq_length, f)
    assert y_test.shape == (total - split,)


def test_scaler_fit_on_train_only():
    n, f = 200, DIRECTION_NUM_INPUTS
    raw = _random_features(n, f)
    train_cutoff = int(n * 0.8)
    scaled, scaler = fit_scaler_and_scale(raw, train_cutoff)
    assert scaled.shape == raw.shape
    train_scaled = scaled[:train_cutoff]
    assert abs(float(train_scaled.mean())) < 0.2


def test_build_loaders_batches():
    n, f, seq_len = 500, DIRECTION_NUM_INPUTS, DIRECTION_PARAMS['seq_length']
    X = np.random.randn(n, seq_len, f).astype(np.float32)
    y = np.random.randint(0, 2, n).astype(np.float32)
    loader = build_loaders(X, y, batch_size=64)
    bx, by = next(iter(loader))
    assert bx.shape == (64, seq_len, f)
    assert by.shape == (64,)
