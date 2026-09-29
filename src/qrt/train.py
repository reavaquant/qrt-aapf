import pandas as pd
from sklearn.base import clone


def train_and_predict_cv(model, X, y, folds, predict_fn, return_models=False):
    oof = pd.Series(index=X.index, dtype=float, name="prediction")
    models = []

    for fold in sorted(folds.unique()):
        validation_rows = folds == fold
        train_rows = folds != fold  

        fold_model = clone(model)
        fold_model.fit(X.loc[train_rows], y.loc[train_rows])

        models.append(fold_model)

        oof.loc[validation_rows] = predict_fn(fold_model, X.loc[validation_rows])

    if return_models:
        return oof, models
    
    return oof


def predict_positive_probability(model, X):
    return model.predict_proba(X)[:, 1]


def predict_return(model, X):
    return model.predict(X)