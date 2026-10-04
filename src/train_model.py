import argparse
import os

import joblib
from sklearn.ensemble import GradientBoostingClassifier

from data import get_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True, help="Version tag for this model")
    args = parser.parse_args()

    X_train, _, _, y_train, _, _ = get_splits()

    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=3, random_state=0
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    path = f"models/model_{args.timestamp}_gb.joblib"
    joblib.dump(model, path)
    print(f"Model saved to {path}")
