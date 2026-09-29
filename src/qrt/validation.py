import pandas as pd
from sklearn.model_selection import KFold


def make_cv_folds(X, n_splits=8, seed=0):
    dates = X["TS"].unique()
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = pd.Series(-1, index=X.index, name="fold")

    for fold, (_, valid_date_indices) in enumerate(splitter.split(dates)):
        valid_dates = dates[valid_date_indices]
        valid_rows = X[X["TS"].isin(valid_dates)].index

        folds.loc[valid_rows] = fold

    return folds