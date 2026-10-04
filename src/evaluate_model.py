import argparse
import json
import os

import joblib
from sklearn.metrics import f1_score

from data import get_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()

    _, _, X_test, _, _, y_test = get_splits()

    model = joblib.load(f"models/model_{args.timestamp}_rf.joblib")
    f1 = f1_score(y_test, model.predict(X_test))

    os.makedirs("metrics", exist_ok=True)
    path = f"metrics/{args.timestamp}_metrics.json"
    with open(path, "w") as f:
        json.dump({"timestamp": args.timestamp, "F1_Score": round(f1, 4)}, f, indent=4)
    print(f"F1 Score: {f1:.4f} -> saved to {path}")
