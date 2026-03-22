import argparse, json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"].values
    return X, y


def make_pipelines(use_smote=False):
    # Logistic Regression
    lr_steps = [
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None))
    ]
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=-1
    )

    if use_smote:
        lr_pipe = ImbPipeline([("smote", SMOTE(random_state=42))] + lr_steps)
        rf_pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", rf)])
        xgb_pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", xgb)])
    else:
        lr_pipe = Pipeline(lr_steps)
        rf_pipe = Pipeline([("clf", rf)])
        xgb_pipe = Pipeline([("clf", xgb)])

    return {"logreg": lr_pipe, "rf": rf_pipe, "xgb": xgb_pipe}


def get_param_grid(model_name, neg_pos_ratio):
    if model_name == "logreg":
        return {"clf__C": [0.5, 1.0, 2.0]}
    if model_name == "rf":
        return {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]}
    if model_name == "xgb":
        return {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__scale_pos_weight": [neg_pos_ratio]
        }
    return {}


def evaluate_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def plot_curves(y_true, y_prob, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc.png"), dpi=200)
    plt.close()

    # PR
    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr.png"), dpi=200)
    plt.close()


def plot_confusion(cm, title, out_path):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_importances(model, feature_names, out_path):
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1][:20]
            plt.figure(figsize=(6, 4))
            sns.barplot(x=imp[idx], y=np.array(feature_names)[idx], orient="h")
            plt.title("Top Feature Importances")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
    except Exception as e:
        print("Importance plot skipped:", e)


def main(args):
    set_seed(args.seed)
    os.makedirs("results", exist_ok=True)

    # Load
    X, y = load_data(args.data_path)
    feature_names = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    # Class ratio
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    ratio = neg / max(1, pos)

    # Pipelines and grid
    pipes = make_pipelines(use_smote=args.use_smote)
    pipe = pipes[args.model]
    param_grid = get_param_grid(args.model, ratio)

    # CV and GridSearch (optimize PR-AUC)
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    grid = GridSearchCV(
        pipe, param_grid=param_grid, scoring="average_precision",
        cv=skf, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # Predict probabilities
    if hasattr(best, "predict_proba"):
        y_prob = best.predict_proba(X_test)[:, 1]
        inner = best
    else:
        y_prob = best.named_steps["clf"].predict_proba(X_test)[:, 1]
        inner = best.named_steps["clf"]

    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    # Threshold scan (report both default 0.5 and best-F1)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    thresholds = np.r_[0.5, thresholds]  # ensure include 0.5
    f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_thr = float(thresholds[int(np.argmax(f1s))])

    m_default = evaluate_threshold(y_test, y_prob, 0.5)
    m_best = evaluate_threshold(y_test, y_prob, best_thr)

    # Plots
    plot_curves(y_test, y_prob, "results")
    plot_confusion(np.array(m_default["confusion_matrix"]), "Confusion (thr=0.5)", "results/confusion_default.png")
    plot_confusion(np.array(m_best["confusion_matrix"]), f"Confusion (best F1 thr={best_thr:.3f})", "results/confusion_best.png")
    plot_importances(inner if hasattr(inner, "feature_importances_") else None, feature_names, "results/importances.png")

    # Save outputs
    out = {
        "model": args.model,
        "use_smote": args.use_smote,
        "cv": args.cv,
        "random_seed": args.seed,
        "class_ratio_neg_pos": ratio,
        "cv_best_score_PR_AUC": float(grid.best_score_),
        "test_metrics": {
            "AUC": float(auc),
            "PR_AUC": float(ap),
            "default_threshold": m_default,
            "best_f1_threshold": m_best
        },
        "best_params": grid.best_params_
    }
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--model", type=str, default="xgb", choices=["xgb", "rf", "logreg"])
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_smote", action="store_true")
    args = p.parse_args()
    main(args)
