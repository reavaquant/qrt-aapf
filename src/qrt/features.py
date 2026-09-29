import pandas as pd
import numpy as np


def compute_features(X):
    """Compute features for the model."""
    X = X.copy()

    ret_cat = ["RET_{i}".format(i=i) for i in range(1, 21)]
    sign_volume_cat = ["SIGNED_VOLUME_{i}".format(i=i) for i in range(1, 21)]
    turnover_cat = ["MEDIAN_DAILY_TURNOVER"]

    for i in [5, 20]:
        Xret_cat = X[ret_cat[:i]]
        Xsv_cat = X[sign_volume_cat[:i]]
        X[f"RET_WIN_RATE_{i}"] = (Xret_cat > 0).sum(axis=1) / Xret_cat.count(axis=1)
        X[f'RET_MEAN_{i}'] = Xret_cat.mean(axis=1)
        X[f'RET_STD_{i}'] = Xret_cat.std(axis=1)
        X[f'SIGNED_VOLUME_MEAN_{i}'] = Xsv_cat.mean(axis=1)
        X[f'SIGNED_VOLUME_STD_{i}'] = Xsv_cat.std(axis=1)

    volatility = X["RET_STD_20"].replace(0, np.nan)
    X["NORM_RET_1_20"] = X["RET_1"] / volatility
    X["RET_STD_RATIO_5_20"] = X["RET_STD_5"] / volatility

    # for col in ["RET_1", "RET_MEAN_5", "RET_MEAN_20"]:
    #     date_mean = X.groupby("TS")[col].transform("mean")

    #     X[f"TS_MEAN_{col}"] = date_mean
    #     X[f"TS_DIFF_{col}"] = X[col] - date_mean

    return X