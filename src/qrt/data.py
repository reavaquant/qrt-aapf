from pathlib import Path

import pandas as pd


def load_tr_data(dir):
    """Load training features and raw targets, indexed by ROW_ID."""

    X_tr = pd.read_csv(dir / "X_train.csv", index_col='ROW_ID')
    y_tr = pd.read_csv(dir / "y_train.csv", index_col='ROW_ID')
    return X_tr, y_tr

def load_te_data(dir):
    """Load test features, indexed by ROW_ID."""

    X_te = pd.read_csv(dir / "X_test.csv", index_col='ROW_ID')   
    return X_te

def load_sample_submission_data(dir):
    """Load sample submission, indexed by ROW_ID."""

    sample_submission = pd.read_csv(dir / "sample_submission.csv", index_col='ROW_ID')
    return sample_submission