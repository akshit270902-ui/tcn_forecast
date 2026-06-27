TRAIN_SPLIT = 0.8
EPOCHS = 50

DIRECTION_PARAMS = {
    'seq_length': 10,
    'batch_size': 128,
    'lr': 1e-3,
    'dropout': 0.1,
    'kernel_size': 3,
    'num_layers': 4,
    'hidden_units': 128,
    'weight_decay': 1e-4,
}

VOLATILITY_PARAMS = {
    'seq_length': 30,
    'batch_size': 128,
    'lr': 1e-3,
    'dropout': 0.2,
    'kernel_size': 5,
    'num_layers': 2,
    'hidden_units': 128,
    'weight_decay': 1e-4,
}

DIRECTION_NUM_INPUTS = 8
VOLATILITY_NUM_INPUTS = 13

DIRECTION_FEATURE_NAMES = [
    'return', 'log_volume', 'log_volume_delta', 'realized_vol',
    'autocorr_5', 'autocorr_10', 'cum_log_volume_delta_10', 'cum_log_volume_10',
]

VOLATILITY_FEATURE_NAMES = [
    'return', 'log_volume',
    'realized_vol_5', 'realized_vol_10', 'realized_vol_20', 'realized_vol_30', 'realized_vol_60',
    'mean_abs_return_5', 'mean_abs_return_10', 'mean_abs_return_20',
    'mean_abs_return_30', 'mean_abs_return_60', 'mean_abs_return_120',
]

GARCH_P = 1
GARCH_Q = 1

RAW_DATA_PATH = 'data/raw/BTCUSDT_1h.csv'
RESULTS_DIR = 'results/'
