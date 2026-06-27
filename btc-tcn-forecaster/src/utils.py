import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model

from config import GARCH_P, GARCH_Q


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def garch_forecast(returns: np.ndarray, train_cutoff: int, n_test: int) -> np.ndarray:
    history = returns[:train_cutoff]
    history = history[np.isfinite(history)]
    am = arch_model(history, vol='Garch', p=GARCH_P, q=GARCH_Q, mean='Zero', rescale=True)
    res = am.fit(disp='off', show_warning=False)
    fc = res.forecast(horizon=n_test, reindex=False)
    return np.sqrt(fc.variance.values[-1]) / res.scale


def print_comparison_table(tcn_metrics: dict, garch_metrics: dict) -> None:
    print("\n" + "═" * 42)
    print("TCN vs GARCH(1,1)")
    print("═" * 42)
    print(f"{'Metric':<10}{'TCN':>16}{'GARCH(1,1)':>16}")
    for key in ('rmse', 'mae', 'r2'):
        print(f"{key.upper():<10}{tcn_metrics[key]:>16.6f}{garch_metrics[key]:>16.6f}")
    winner = "TCN" if tcn_metrics['rmse'] < garch_metrics['rmse'] else "GARCH(1,1)"
    print(f"\n{winner} outperforms on RMSE.")


def print_feature_importances(importances: np.ndarray, feature_names: list) -> None:
    print("\n" + "═" * 42)
    print("FEATURE IMPORTANCE (Permutation)")
    print("═" * 42)
    for rank, idx in enumerate(np.argsort(importances)[::-1], 1):
        print(f"{rank}. {feature_names[idx]:<28} RMSE increase: {importances[idx]:+.6f}")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str, n_plot: int = 100) -> None:
    actual = y_true[-n_plot:]
    predicted = y_pred[-n_plot:]
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label='Actual', color='black', linewidth=1.5)
    plt.plot(predicted, label='Predicted', color='crimson', linewidth=1.5, linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_garch_predictions(y_true: np.ndarray, garch_pred: np.ndarray, save_path: str, n_plot: int = 100) -> None:
    actual = y_true[-n_plot:]
    predicted = garch_pred[-n_plot:]
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label='Actual |Return|', color='black', linewidth=1.5)
    plt.plot(predicted, label='GARCH(1,1) Forecast', color='royalblue', linewidth=1.5, linestyle='--')
    plt.title(f'GARCH(1,1): Predicted vs Actual Return Magnitude (Last {n_plot} Test Samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Return')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
