import glob
import json
import os

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from data import get_splits

if __name__ == "__main__":
    # Find the most recent uncalibrated model (timestamps sort in time order)
    models = sorted(glob.glob("models/model_*_rf.joblib"))
    if not models:
        raise SystemExit("No trained model found in models/")
    latest = models[-1]
    timestamp = latest.split("model_")[1].split("_rf")[0]
    print(f"Calibrating {latest}")

    _, X_cal, X_test, _, y_cal, y_test = get_splits()
    model = joblib.load(latest)

    # Platt scaling (sigmoid) on held-out calibration data, without retraining the forest
    try:
        from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    # Brier score: lower = predicted probabilities closer to reality
    before = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
    after = brier_score_loss(y_test, calibrated.predict_proba(X_test)[:, 1])

    out = f"models/model_{timestamp}_calibrated.joblib"
    joblib.dump(calibrated, out)

    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{timestamp}_calibration.json", "w") as f:
        json.dump({"timestamp": timestamp,
                   "brier_before": round(before, 4),
                   "brier_after": round(after, 4)}, f, indent=4)
    print(f"Brier score: before={before:.4f}, after={after:.4f} -> saved {out}")
