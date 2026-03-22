Title: Credit Card Fraud Detection – Experiment Config

Task: Binary classification, extreme class imbalance

Dataset: Kaggle Credit Card Fraud (URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Split: Stratified train/test = 80/20, random_state=42

Cross-validation: StratifiedKFold, n_splits=5, shuffle=True, random_state=42

Scoring: average_precision (PR-AUC)

Models:

Logistic Regression: StandardScaler + LogisticRegression(class_weight=balanced, max_iter=200)
XGBoost: XGBClassifier(tree_method=hist, random_state=42, n_jobs=-1), scale_pos_weight ≈ neg/pos
Hyperparameter grids:

LR: C ∈ {0.5, 1.0, 2.0}
XGB: n_estimators ∈ {300, 600}, max_depth ∈ {3, 5}, learning_rate ∈ {0.05, 0.1}, subsample ∈ {0.8, 1.0}, colsample_bytree ∈ {0.8, 1.0}, scale_pos_weight ≈ 577
Thresholding: report thr=0.5 and best-F1 threshold (scanned on validation, reported on test)

Test metrics: ROC-AUC, PR-AUC (Average Precision), F1, Precision, Recall, confusion matrix

Reproducibility: random_state=42; requirements.txt (+ optional requirements-lock.txt)
