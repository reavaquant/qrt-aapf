import pandas as pd
import numpy as np


def compute_features(X):
    """Compute features for the model."""
    X = X.copy()

    ret_cat = ["RET_{i}".format(i=i) for i in range(1, 21)]
    sign_volume_cat = ["SIGNED_VOLUME_{i}".format(i=i) for i in range(1, 21)]
    turnover_cat = ["MEDIAN_DAILY_TURNOVER"]

    for i in [5, 20]:
        history = X[ret_cat[:i]]
        X[f"RET_WIN_RATE_{i}"] = (history > 0).sum(axis=1) / history.count(axis=1)
        X[f'RET_MEAN_{i}'] = history.mean(axis=1)
        X[f'RET_STD_{i}'] = history.std(axis=1)

    volatility = X["RET_STD_20"].replace(0, np.nan)
    X["NORM_RET_1_20"] = X["RET_1"] / volatility
    X["RET_STD_RATIO_5_20"] = X["RET_STD_5"] / volatility

    return X