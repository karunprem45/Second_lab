# MLOps Lab 2 – Automated Model Training, Versioning & Calibration with GitHub Actions

Every push to `main` automatically trains, evaluates, versions, and calibrates a classifier using GitHub Actions. Trained models and metrics are committed back to the repo by the workflow.

Based on [raminmohammadi/MLOps – Github_Labs/Lab2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2).

## My Modifications

| Area | Original Lab | My Version |
|------|-------------|------------|
| Dataset | Synthetic data (`make_classification`) | **Breast Cancer Wisconsin** (real medical data, 569 samples, 30 features) |
| Model | Random Forest | **Gradient Boosting Classifier** |
| Data split | Train / test | **Train (60%) / calibration (20%) / test (20%)**: calibration uses held-out data so it doesn't overfit |
| Metrics | F1 only | **Accuracy, Precision, Recall, F1, ROC-AUC** |
| Calibration check | Not measured | **Brier score before vs after calibration** saved to `metrics/` |
| Workflow chaining | Both workflows trigger on push (calibration can run before a model exists) | Calibration uses **`workflow_run`**, so it runs only after training succeeds |
| Permissions | Not set | **`permissions: contents: write`** so the bot can commit models back |
| Actions versions | `@v2`, Python 3.8 | `checkout@v4`, `setup-python@v5`, Python 3.11 |

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.9474 |
| Precision | 0.9710 |
| Recall | 0.9437 |
| F1 Score | 0.9571 |
| ROC-AUC | 0.9925 |
| Brier score (before → after calibration) | 0.0462 → 0.0375 |

Calibration reduced the Brier score by about 19%, meaning the model's predicted probabilities better match real outcomes.

## Project Structure

    Second_lab/
    ├── .github/workflows/
    │   ├── model_retraining_on_push.yml    # train -> evaluate -> version -> commit
    │   └── model_calibration_on_push.yml   # runs after training: calibrate -> commit
    ├── src/
    │   ├── data.py              # loads dataset + train/cal/test split
    │   ├── train_model.py       # trains Gradient Boosting, saves models/model_<timestamp>_gb.joblib
    │   ├── evaluate_model.py    # saves metrics/<timestamp>_metrics.json
    │   └── calibrate_model.py   # Platt scaling, saves calibrated model + Brier scores
    ├── models/                  # versioned models (written by GitHub Actions)
    ├── metrics/                 # versioned metrics (written by GitHub Actions)
    └── requirements.txt

Earlier `_rf` model files in `models/` are from the initial Random Forest version, kept as version history.

## How It Works

1. **Push to `main`** triggers *Model Retraining on Push*
2. A **timestamp** is generated as the version ID
3. The model is **trained** and saved as `models/model_<timestamp>_gb.joblib`
4. The model is **evaluated**, with metrics saved to `metrics/<timestamp>_metrics.json`
5. The bot **commits and pushes** the new version
6. *Model Calibration* starts automatically, **calibrates** the latest model with Platt scaling, and commits `model_<timestamp>_calibrated.joblib` plus Brier scores

## Run Locally

    python3 -m venv lab_02
    source lab_02/bin/activate
    pip install -r requirements.txt
    python src/train_model.py --timestamp local
    python src/evaluate_model.py --timestamp local
    python src/calibrate_model.py
