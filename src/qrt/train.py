import pandas as pd
from sklearn.base import clone


def train_and_predict_cv(model, X, y, folds, predict_fn):
    oof = pd.Series(index=X.index, dtype=float, name="prediction")

    for fold in sorted(folds.unique()):
        validation_rows = folds == fold
        train_rows = folds != fold  

        fold_model = clone(model)
        fold_model.fit(X.loc[train_rows], y.loc[train_rows])

        oof.loc[validation_rows] = predict_fn(fold_model, X.loc[validation_rows])

    return oof


def predict_positive_probability(model, X):
    return model.predict_proba(X)[:, 1]


def predict_return(model, X):
    return model.predict(X)