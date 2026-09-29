import pandas as pd
from sklearn.model_selection import KFold


def make_date_folds(X, n_splits=8, seed=0):
    dates = X["TS"].unique()
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = pd.Series(-1, index=X.index, name="fold")

    for fold, (_, validation_date_indices) in enumerate(splitter.split(dates)):
        validation_dates = dates[validation_date_indices]
        validation_rows = X[X["TS"].isin(validation_dates)].index

        folds.loc[validation_rows] = fold

    return folds