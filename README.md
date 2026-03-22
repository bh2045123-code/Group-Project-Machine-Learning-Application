Project Title
Credit Card Fraud Detection — End-to-End ML Application

Overview
This repository implements a fully reproducible, end-to-end machine learning pipeline on the Kaggle Credit Card Fraud dataset. The pipeline covers:
- EDA (class balance, amount/time distributions, correlation heatmap)
- Preprocessing and imbalance handling
- Modeling: Logistic Regression (baseline) and XGBoost (primary)
- Stratified 5-fold CV with hyperparameter tuning (PR-AUC as the selection metric)
- Threshold tuning (default 0.5 vs best-F1)
- Test evaluation (ROC/PR curves, confusion matrices, metrics)
- Interpretability (feature importances)

Due to the extreme class imbalance (~1:577), PR-AUC (Average Precision) is used for model selection.

Dataset
- Source (Kaggle): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Place creditcard.csv under data/ (see data/README.md)
- Features: Time, Amount, PCA-anonymized V1–V28; Label: Class (0/1)
- Imbalance: 284,807 rows; 492 fraud (≈0.172%); negative/positive ≈ 577:1

Environment
- Python 3.9+ (virtual environment recommended)
- Install dependencies:
  pip install -r requirements.txt
- Optional: lock exact versions:
  pip freeze > requirements-lock.txt

Quick Start
1) EDA (writes images to figures/)
- Generate 4 EDA plots (class balance, amount/time distributions, correlation heatmap):
  python src/eda.py --data_path data/creditcard.csv

2) Train and evaluate (writes artifacts to results/)
- XGBoost (primary):
  python src/train.py --data_path data/creditcard.csv --model xgb --cv 5 --seed 42
- Logistic Regression (baseline):
  python src/train.py --data_path data/creditcard.csv --model logreg --cv 5 --seed 42
- Outputs to results/: metrics.json, roc.png, pr.png, confusion_default.png, confusion_best.png, importances.png (XGB)

One-click (Windows)
- experiments/run_xgb.bat
- experiments/run_logreg.bat

Expected Results (reference ranges)
- XGBoost (seed=42, 5-fold, scoring=average_precision)
  - Test ROC-AUC ≈ 0.975
  - Test PR-AUC ≈ 0.882
  - F1@0.5 ≈ 0.859; F1@best ≈ 0.885
- Logistic Regression (seed=42, 5-fold)
  - Test ROC-AUC ≈ 0.972
  - Test PR-AUC ≈ 0.719
  - F1@0.5 ≈ 0.114; F1@best ≈ 0.825
Note: Your exact numbers should match results/metrics.json within small variance (±0.01 typical).

Reproducibility
- Config and experiment details: experiments/config.md
- Summarized results: experiments/results_summary.md
- One-click scripts: experiments/run_xgb.bat, experiments/run_logreg.bat
- Local artifacts are not versioned: results/ and figures/ contain generated images and metrics (see their README placeholders)
- Data is not versioned: download from Kaggle and place under data/

Directory Structure
- src/
  - eda.py         Generate EDA plots (figures/)
  - train.py       Train/evaluate LR/XGB; save metrics and plots (results/)
- experiments/
  - run_xgb.bat    One-click XGBoost run
  - run_logreg.bat One-click Logistic Regression run
  - config.md      Experiment configuration (splits, metrics, grids)
  - results_summary.md Key results summary
- data/
  - README.md      Where to put creditcard.csv (not versioned)
- results/
  - README.md      Local artifacts description (not versioned)
- figures/
  - README.md      EDA images description (not versioned)
- requirements.txt (+ optional requirements-lock.txt)
- README.md
- .gitignore

Metrics and Thresholding
- Selection metric: Average Precision (PR-AUC)
- Reported metrics: ROC-AUC, PR-AUC, F1, Precision, Recall
- Thresholds: compare default 0.5 vs best-F1 (scanned on validation; reported on test)

Troubleshooting
- “File not found: data/creditcard.csv” → ensure the CSV is placed under data/
- “ModuleNotFoundError” → pip install -r requirements.txt
- No results generated → check console errors; run from the repo root
- Windows path quoting → wrap paths with spaces in double quotes

Academic Notes
- Dataset: ULB/Kaggle Credit Card Fraud
- Libraries: scikit-learn, xgboost, matplotlib, seaborn, imbalanced-learn
- Why PR-AUC? Better reflects minority-class performance in extreme imbalance

License
- MIT (or adapt to your course requirements)

Team
- Group name: Your Group Name
- Group leader: Name (email)
- Questions: open an issue or contact the group leader
