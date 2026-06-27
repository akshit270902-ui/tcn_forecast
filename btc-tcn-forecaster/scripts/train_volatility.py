import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from config import (
    EPOCHS, VOLATILITY_PARAMS, VOLATILITY_NUM_INPUTS,
    VOLATILITY_FEATURE_NAMES, RAW_DATA_PATH, TRAIN_SPLIT, RESULTS_DIR,
)
from src.features import load_and_prepare, build_volatility_features
from src.model import TCNModel
from src.datasets import make_sequences, fit_scaler_and_scale, build_loaders
from src.utils import (
    compute_regression_metrics, garch_forecast,
    print_comparison_table, print_feature_importances,
    plot_predictions, plot_garch_predictions,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    df = load_and_prepare(RAW_DATA_PATH)
    raw, targets = build_volatility_features(df)

    train_cutoff = int(len(raw) * TRAIN_SPLIT)
    scaled, scaler = fit_scaler_and_scale(raw, train_cutoff)

    p = VOLATILITY_PARAMS
    X_train, y_train, X_test, y_test = make_sequences(scaled, targets, p['seq_length'])
    train_loader = build_loaders(X_train, y_train, p['batch_size'])
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    model = TCNModel(
        num_inputs=VOLATILITY_NUM_INPUTS,
        num_channels=p['hidden_units'],
        num_layers=p['num_layers'],
        kernel_size=p['kernel_size'],
        dropout=p['dropout'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    criterion = nn.L1Loss()

    best_preds, best_rmse = None, float('inf')

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).cpu().numpy()
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                if rmse < best_rmse:
                    best_rmse, best_preds = rmse, preds
            print(f"Epoch {epoch + 1:>3} | RMSE: {rmse:.6f}")

    tcn_metrics = compute_regression_metrics(y_test, best_preds)
    print(f"\nTCN  RMSE: {tcn_metrics['rmse']:.6f}  MAE: {tcn_metrics['mae']:.6f}  R2: {tcn_metrics['r2']:.4f}")

    importances = np.zeros(VOLATILITY_NUM_INPUTS)
    model.eval()
    for feat_idx in range(VOLATILITY_NUM_INPUTS):
        increases = []
        for _ in range(10):
            X_perm = X_test_t.clone()
            X_perm[:, :, feat_idx] = X_perm[torch.randperm(X_perm.shape[0]), :, feat_idx]
            with torch.no_grad():
                perm_preds = model(X_perm).cpu().numpy()
            perm_rmse = float(np.sqrt(mean_squared_error(y_test, perm_preds)))
            increases.append(perm_rmse - best_rmse)
        importances[feat_idx] = float(np.mean(increases))

    print_feature_importances(importances, VOLATILITY_FEATURE_NAMES)

    seq_length = p['seq_length']
    garch_train_cutoff = train_cutoff + seq_length - 1
    n_test = len(y_test)
    returns_raw = df['return'].values[:-1]
    returns_raw = np.nan_to_num(returns_raw, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nFitting GARCH(1,1) on {garch_train_cutoff} training bars, forecasting {n_test} steps...")
    garch_preds = garch_forecast(returns_raw, garch_train_cutoff, n_test)
    print("GARCH forecast complete.")

    garch_metrics = compute_regression_metrics(y_test, garch_preds)
    print(f"GARCH RMSE: {garch_metrics['rmse']:.6f}  MAE: {garch_metrics['mae']:.6f}  R2: {garch_metrics['r2']:.4f}")

    print_comparison_table(tcn_metrics, garch_metrics)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "volatility_model.pt"))

    plot_predictions(
        y_test, best_preds,
        title=f'TCN: Predicted vs Actual |Return| (Last 100 Test Samples)',
        save_path=os.path.join(RESULTS_DIR, "tcn_vs_actual.png"),
    )

    plot_garch_predictions(
        y_test, garch_preds,
        save_path=os.path.join(RESULTS_DIR, "garch_vs_actual.png"),
    )

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
