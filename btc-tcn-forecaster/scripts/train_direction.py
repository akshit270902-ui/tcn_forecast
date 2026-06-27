import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from config import (
    EPOCHS, DIRECTION_PARAMS, DIRECTION_NUM_INPUTS,
    DIRECTION_FEATURE_NAMES, RAW_DATA_PATH, TRAIN_SPLIT, RESULTS_DIR,
)
from src.features import load_and_prepare, build_direction_features
from src.model import TCNModel
from src.datasets import make_sequences, fit_scaler_and_scale, build_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    df = load_and_prepare(RAW_DATA_PATH)
    raw, targets = build_direction_features(df)

    train_cutoff = int(len(raw) * TRAIN_SPLIT)
    scaled, scaler = fit_scaler_and_scale(raw, train_cutoff)

    p = DIRECTION_PARAMS
    X_train, y_train, X_test, y_test = make_sequences(scaled, targets, p['seq_length'])
    train_loader = build_loaders(X_train, y_train, p['batch_size'])
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    model = TCNModel(
        num_inputs=DIRECTION_NUM_INPUTS,
        num_channels=p['hidden_units'],
        num_layers=p['num_layers'],
        kernel_size=p['kernel_size'],
        dropout=p['dropout'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

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
                preds = torch.sigmoid(model(X_test_t)).cpu().numpy()
                auc = roc_auc_score(y_test, preds)
            print(f"Epoch {epoch + 1:>3} | AUC: {auc:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "direction_model.pt"))
    print(f"\nModel saved to {RESULTS_DIR}direction_model.pt")


if __name__ == "__main__":
    main()
