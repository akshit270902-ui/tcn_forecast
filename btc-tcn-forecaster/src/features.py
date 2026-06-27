import numpy as np
import pandas as pd


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.std(arr[i - window + 1: i + 1])
    return out


def _rolling_mean_abs(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.mean(np.abs(arr[i - window + 1: i + 1]))
    return out


def _rolling_lag1_autocorr(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        x = arr[i - window + 1: i + 1]
        out[i] = np.corrcoef(x[:-1], x[1:])[0, 1]
    return out


def _rolling_cumsum(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.sum(arr[i - window + 1: i + 1])
    return out


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df['return'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
    df['log_volume'] = np.log(df['volume'] + 1)
    df['log_volume_delta'] = np.log(df['volume'] + 1) - np.log(df['taker_buy_volume'] + 1)
    return df


def build_direction_features(df: pd.DataFrame):
    returns = df['return'].values[:-1]
    log_volume = df['log_volume'].values[:-1]
    log_volume_delta = df['log_volume_delta'].values[:-1]

    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    realized_vol = _rolling_std(returns, 10)
    autocorr_5 = _rolling_lag1_autocorr(returns, 5)
    autocorr_10 = _rolling_lag1_autocorr(returns, 10)
    cum_log_volume_delta_10 = _rolling_cumsum(log_volume_delta, 10)
    cum_log_volume_10 = _rolling_cumsum(log_volume, 10)

    targets = (df['return'].shift(-1) > 0).astype(np.float32).values[:-1]

    raw = np.nan_to_num(np.concatenate([
        returns.reshape(-1, 1),
        log_volume.reshape(-1, 1),
        log_volume_delta.reshape(-1, 1),
        realized_vol.reshape(-1, 1),
        autocorr_5.reshape(-1, 1),
        autocorr_10.reshape(-1, 1),
        cum_log_volume_delta_10.reshape(-1, 1),
        cum_log_volume_10.reshape(-1, 1),
    ], axis=1), nan=0.0).astype(np.float32)

    return raw, targets


def build_volatility_features(df: pd.DataFrame):
    returns = df['return'].values[:-1]
    log_volume = df['log_volume'].values[:-1]

    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    realized_vol_5   = _rolling_std(returns, 5)
    realized_vol_10  = _rolling_std(returns, 10)
    realized_vol_20  = _rolling_std(returns, 20)
    realized_vol_30  = _rolling_std(returns, 30)
    realized_vol_60  = _rolling_std(returns, 60)

    mean_abs_return_5   = _rolling_mean_abs(returns, 5)
    mean_abs_return_10  = _rolling_mean_abs(returns, 10)
    mean_abs_return_20  = _rolling_mean_abs(returns, 20)
    mean_abs_return_30  = _rolling_mean_abs(returns, 30)
    mean_abs_return_60  = _rolling_mean_abs(returns, 60)
    mean_abs_return_120 = _rolling_mean_abs(returns, 120)

    targets = np.abs(df['return'].shift(-1)).astype(np.float32).values[:-1]
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    raw = np.nan_to_num(np.abs(np.concatenate([
        returns.reshape(-1, 1),
        log_volume.reshape(-1, 1),
        realized_vol_5.reshape(-1, 1),
        realized_vol_10.reshape(-1, 1),
        realized_vol_20.reshape(-1, 1),
        realized_vol_30.reshape(-1, 1),
        realized_vol_60.reshape(-1, 1),
        mean_abs_return_5.reshape(-1, 1),
        mean_abs_return_10.reshape(-1, 1),
        mean_abs_return_20.reshape(-1, 1),
        mean_abs_return_30.reshape(-1, 1),
        mean_abs_return_60.reshape(-1, 1),
        mean_abs_return_120.reshape(-1, 1),
    ], axis=1)), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return raw, targets
