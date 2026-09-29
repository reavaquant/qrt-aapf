import pandas as pd
from qrt.splitters import make_date_folds


def test_make_cv_folds():
    X = pd.DataFrame(
        {"TS": ["A", "A", "B", "B", "C", "C"]},
        index=range(10, 16),
    )

    folds = make_date_folds(X, n_splits=3)

    assert folds.index.equals(X.index)
    assert set(folds) == {0, 1, 2}
    assert (folds.groupby(X["TS"]).nunique() == 1).all()