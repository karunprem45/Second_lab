from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_splits():
    """Synthetic binary classification data, split into train / calibration / test."""
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10, random_state=42
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test
