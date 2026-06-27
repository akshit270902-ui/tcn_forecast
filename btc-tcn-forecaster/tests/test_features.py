import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features import (
    build_direction_features,
    build_volatility_features,
    _rolling_std,
    _rolling_mean_abs,
    _rolling_lag1_autocorr,
    _rolling_cumsum,
)
from config import DIRECTION_NUM_INPUTS, VOLATILITY_NUM_INPUTS


def _make_dummy_df(n: int = 500) -> pd.DataFrame:
    np.random.seed(7)
    close = 30000 * np.exp(np.random.randn(n).cumsum() * 0.002)
    volume = np.abs(np.random.randn(n)) * 1000 + 500
    taker_buy = volume * np.random.uniform(0.3, 0.7, n)
    df = pd.DataFrame({'close': close, 'volume': volume, 'taker_buy_volume': taker_buy})
    df['return'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
    df['log_volume'] = np.log(df['volume'] + 1)
    df['log_volume_delta'] = np.log(df['volume'] + 1) - np.log(df['taker_buy_volume'] + 1)
    return df


def test_direction_features_shape():
    df = _make_dummy_df()
    raw, targets = build_direction_features(df)
    assert raw.shape[0] == len(df) - 1
    assert raw.shape[1] == DIRECTION_NUM_INPUTS
    assert targets.shape[0] == len(df) - 1


def test_direction_features_no_nan():
    df = _make_dummy_df()
    raw, targets = build_direction_features(df)
    assert not np.any(np.isnan(raw))
    assert not np.any(np.isnan(targets))


def test_direction_targets_binary():
    df = _make_dummy_df()
    _, targets = build_direction_features(df)
    assert set(np.unique(targets)).issubset({0.0, 1.0})


def test_volatility_features_shape():
    df = _make_dummy_df()
    raw, targets = build_volatility_features(df)
    assert raw.shape[0] == len(df) - 1
    assert raw.shape[1] == VOLATILITY_NUM_INPUTS
    assert targets.shape[0] == len(df) - 1


def test_volatility_features_non_negative():
    df = _make_dummy_df()
    raw, targets = build_volatility_features(df)
    assert np.all(raw >= 0)
    assert np.all(targets >= 0)


def test_volatility_features_no_nan():
    df = _make_dummy_df()
    raw, targets = build_volatility_features(df)
    assert not np.any(np.isnan(raw))
    assert not np.any(np.isnan(targets))


def test_rolling_std_shape():
    arr = np.random.randn(200)
    out = _rolling_std(arr, window=10)
    assert out.shape == (200,)
    assert np.all(np.isnan(out[:9]))
    assert not np.any(np.isnan(out[9:]))


def test_rolling_std_non_negative():
    arr = np.random.randn(100)
    out = _rolling_std(arr, window=5)
    assert np.all(out[4:] >= 0)


def test_rolling_mean_abs_non_negative():
    arr = np.random.randn(100)
    out = _rolling_mean_abs(arr, window=5)
    assert np.all(out[4:] >= 0)


def test_rolling_lag1_autocorr_range():
    arr = np.random.randn(200)
    out = _rolling_lag1_autocorr(arr, window=10)
    valid = out[~np.isnan(out)]
    assert np.all(valid >= -1.0 - 1e-9) and np.all(valid <= 1.0 + 1e-9)


def test_rolling_cumsum_correctness():
    arr = np.ones(20)
    out = _rolling_cumsum(arr, window=5)
    assert np.all(out[4:] == 5.0)


def test_feature_determinism():
    df = _make_dummy_df()
    a, _ = build_direction_features(df.copy())
    b, _ = build_direction_features(df.copy())
    np.testing.assert_array_equal(a, b)
