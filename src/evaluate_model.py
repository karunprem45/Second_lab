import argparse
import json
import os

import joblib
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from data import get_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()

    _, _, X_test, _, _, y_test = get_splits()

    model = joblib.load(f"models/model_{args.timestamp}_gb.joblib")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "timestamp": args.timestamp,
        "model": "GradientBoostingClassifier",
        "dataset": "Breast Cancer Wisconsin",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "F1_Score": round(f1_score(y_test, preds), 4),
        "ROC_AUC": round(roc_auc_score(y_test, probs), 4),
    }

    os.makedirs("metrics", exist_ok=True)
    path = f"metrics/{args.timestamp}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(json.dumps(metrics, indent=4))
