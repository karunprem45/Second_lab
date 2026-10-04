from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def get_splits():
    """Breast Cancer Wisconsin dataset (569 samples, 30 features),
    split into train (60%) / calibration (20%) / test (20%)."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test
