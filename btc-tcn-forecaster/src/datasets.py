import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config import TRAIN_SPLIT


def make_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i: i + seq_length])
        y.append(targets[i + seq_length - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    split = int(len(X) * TRAIN_SPLIT)
    return X[:split], y[:split], X[split:], y[split:]


def fit_scaler_and_scale(raw: np.ndarray, train_cutoff: int) -> np.ndarray:
    scaler = StandardScaler()
    scaler.fit(raw[:train_cutoff])
    return scaler.transform(raw).astype(np.float32), scaler


def build_loaders(X_train: np.ndarray, y_train: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
