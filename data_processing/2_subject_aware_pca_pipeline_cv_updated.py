

"""
Script: 2_subject_aware_pca_pipeline_cv.py (UPDATED)
Purpose: PCA-based dimensionality reduction *inside* subject-aware cross-validation using proper
         sklearn Pipelines (StandardScaler -> PCA -> Classifier) to avoid information leakage.
         Adds imbalance-robust metrics (PR-AUC, class-wise F1, Macro-F1, MCC) and PR curves.
Author: (Antony Morales Cervantes)
Date: 2025-09-29

How to run:
    - Edit DATA_DIR and OUTPUT_DIR_BASE below.
    - Ensure required packages are installed (scikit-learn, xgboost, numpy, pandas, matplotlib).
    - Run: python 2_1_subject_aware_pca_pipeline_cv.py

Outputs (per analysis):
    - full_results_<analysis_name>_subject_cv.csv : per-fold metrics (now incl. PR_AUC, F1s, MCC)
    - summary_<analysis_name>_subject_cv.txt      : mean metrics across folds
    - accuracy_boxplot_<analysis_name>.png        : distribution of accuracies across folds
    - confusion_best_worst_<analysis_name>.png    : normalized confusion matrices (best/worst by accuracy)
    - roc_curves_<analysis_name>.png              : mean ROC curves across folds
    - pr_curves_<analysis_name>.png               : mean Precision–Recall curves across folds
    - roc_data.pkl                                : raw FPR/TPR per fold per model (for reproducibility)
    - pr_data.pkl                                 : raw Precision/Recall per fold per model (for reproducibility)
"""

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve
)

import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

# 1) Paths (EDIT THESE)
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR_BASE = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results2\2_subject_aware_pca_pipeline_cv"

# 2) Cross-validation
N_SPLITS = 5
RANDOM_STATE = 42

# 3) PCA configurations for analyses
ANALYSES = {
    # PCA keeps 95% of variance (fitted ONLY on training folds via Pipeline)
    "PCA_95_variance": dict(pca_n_components=0.95),
    # Fixed small representation (2 PCs) for visualization/comparison
    "PCA_2_components": dict(pca_n_components=2),
}


# =========================
# Utilities
# =========================

def build_models(pca_n_components) -> Dict[str, Pipeline]:
    """
    Construct classifier pipelines with StandardScaler -> PCA(n_components) -> Classifier.
    PCA and scaling are fit ONLY on training data within each fold by design (Pipeline + CV).
    """
    common_steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_n_components, random_state=RANDOM_STATE)),
    ]

    models = {
        "RandomForest": Pipeline(common_steps + [
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline(common_steps + [
            ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline(common_steps + [
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "XGBoost": Pipeline(common_steps + [
            ("clf", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                max_depth=5,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                use_label_encoder=False
            )),
        ]),
        "LogisticRegression": Pipeline(common_steps + [
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)),
        ]),
        "MLP": Pipeline(common_steps + [
            ("clf", MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-4, max_iter=500, random_state=RANDOM_STATE)),
        ]),
    }
    return models


def metrics_from_confusion(cm: np.ndarray) -> Tuple[float, float, float, float, float]:
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return acc, sens, spec, ppv, npv


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_accuracy_boxplot(df_all: pd.DataFrame, analysis_name: str, output_dir: str):
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Model', y='Accuracy', data=df_all)
    plt.title(f'Accuracy Distribution ({analysis_name} - Subject-Aware CV{N_SPLITS})')
    plt.xticks(rotation=15, ha='right')
    plt.savefig(os.path.join(output_dir, f'accuracy_boxplot_{analysis_name}.png'), dpi=300)
    plt.close()


def plot_confusion_best_worst(df_all: pd.DataFrame, conf_mats: Dict[str, List[np.ndarray]], analysis_name: str, output_dir: str):
    import seaborn as sns
    sns.set(style="whitegrid", context="talk")

    model_avg = df_all.groupby("Model")["Accuracy"].mean()
    best = model_avg.idxmax()
    worst = model_avg.idxmin()

    def normalize_cm(cm):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cm / row_sums

    avg_cm_best = np.mean(conf_mats[best], axis=0)
    avg_cm_worst = np.mean(conf_mats[worst], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.3)

    for ax, cm, title in zip(
        axes,
        [normalize_cm(avg_cm_best), normalize_cm(avg_cm_worst)],
        [f"Best Model — {best}", f"Worst Model — {worst}"]
    ):
        sns.heatmap(
            cm, annot=True, fmt=".3f", cmap="Blues",
            cbar=False, square=True, linewidths=0.8,
            annot_kws={"size": 14}, ax=ax
        )
        ax.set_title(f"{title}\n({analysis_name}, CV5)", fontsize=18, pad=16, weight='bold')
        ax.set_xlabel("Predicted Label", fontsize=14)
        ax.set_ylabel("True Label", fontsize=14)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Control", "Post-COVID"], fontsize=13)
        ax.set_yticklabels(["Control", "Post-COVID"], fontsize=13)
        ax.tick_params(axis='both', which='major', labelsize=12)

    sns.despine(left=True, bottom=True)
    plt.suptitle(f"Normalized Confusion Matrices — {analysis_name}", fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_best_worst_{analysis_name}.png"), dpi=400, bbox_inches="tight")
    plt.close()


def plot_mean_roc(df_all: pd.DataFrame, roc_store: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], analysis_name: str, output_dir: str):
    plt.figure(figsize=(8, 7))
    for model_name, roc_folds in roc_store.items():
        if not roc_folds:
            continue
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []
        for fpr, tpr in roc_folds:
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tprs.append(tpr_interp)
        if not tprs:
            continue
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        mean_auc = df_all[df_all["Model"] == model_name]["AUC"].mean()
        plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={mean_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Average ROC Curves ({analysis_name} - Subject-Aware CV{N_SPLITS})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curves_{analysis_name}.png'), dpi=300)
    plt.close()


def plot_mean_pr(df_all: pd.DataFrame, pr_store: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], analysis_name: str, output_dir: str):
    plt.figure(figsize=(8, 7))
    for model_name, pr_folds in pr_store.items():
        if not pr_folds:
            continue
        recall_grid = np.linspace(0, 1, 200)
        precisions = []
        for precision, recall in pr_folds:
            order = np.argsort(recall)
            r = recall[order]
            p = precision[order]
            p_interp = np.interp(recall_grid, r, p)
            precisions.append(p_interp)
        if not precisions:
            continue
        mean_precision = np.mean(np.vstack(precisions), axis=0)
        pr_auc_mean_curve = np.trapz(mean_precision, recall_grid)
        plt.plot(recall_grid, mean_precision, label=f"{model_name} (PR-AUC≈{pr_auc_mean_curve:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Average Precision–Recall Curves ({analysis_name} - Subject-Aware CV{N_SPLITS})")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pr_curves_{analysis_name}.png'), dpi=300)
    plt.close()


# =========================
# Core CV runner
# =========================

def run_subject_aware_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, analysis_name: str):
    output_dir = os.path.join(OUTPUT_DIR_BASE, analysis_name)
    ensure_dir(output_dir)

    models = build_models(pca_n_components=ANALYSES[analysis_name]["pca_n_components"])
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    records = []
    conf_mats: Dict[str, List[np.ndarray]] = {m: [] for m in models.keys()}
    roc_store: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {m: [] for m in models.keys()}
    pr_store: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {m: [] for m in models.keys()}

    for model_name, pipe in models.items():
        fold_idx = 0
        for train_idx, test_idx in cv.split(X, y, groups):
            fold_idx += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:, 1]
            elif hasattr(pipe.named_steps["clf"], "decision_function"):
                scores = pipe.decision_function(X_test)
                ranks = (scores.argsort().argsort()).astype(float)
                y_proba = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)
            else:
                y_proba = y_pred.astype(float)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            acc, sens, spec, ppv, npv = metrics_from_confusion(cm)

            auc = np.nan
            if len(np.unique(y_test)) > 1:
                try:
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    auc = np.nan

            try:
                pr_auc = average_precision_score(y_test, y_proba)
            except Exception:
                pr_auc = np.nan

            f1_pos = f1_score(y_test, y_pred, pos_label=1)
            f1_neg = f1_score(y_test, y_pred, pos_label=0)
            macro_f1 = (f1_pos + f1_neg) / 2.0
            mcc = matthews_corrcoef(y_test, y_pred)

            records.append({
                "Model": model_name,
                "Fold": fold_idx,
                "Accuracy": acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "PPV": ppv,
                "NPV": npv,
                "AUC": auc,
                "PR_AUC": pr_auc,
                "F1_Pos": f1_pos,
                "F1_Neg": f1_neg,
                "Macro_F1": macro_f1,
                "MCC": mcc,
            })
            conf_mats[model_name].append(cm)

            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_store[model_name].append((fpr, tpr))
            except Exception:
                pass

            try:
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_store[model_name].append((precision, recall))
            except Exception:
                pass

    df_all = pd.DataFrame.from_records(records)
    df_all.to_csv(os.path.join(output_dir, f'full_results_{analysis_name}_subject_cv.csv'), index=False)

    with open(os.path.join(output_dir, f'summary_{analysis_name}_subject_cv.txt'), "w", encoding="utf-8") as f:
        f.write(f"--- Summary for {analysis_name} (Subject-Aware CV{N_SPLITS}) ---\n\n")
        f.write(df_all.groupby("Model").mean(numeric_only=True).drop(columns=["Fold"]).to_string())

    plot_accuracy_boxplot(df_all, analysis_name, output_dir)
    plot_confusion_best_worst(df_all, conf_mats, analysis_name, output_dir)
    plot_mean_roc(df_all, roc_store, analysis_name, output_dir)
    plot_mean_pr(df_all, pr_store, analysis_name, output_dir)

    with open(os.path.join(output_dir, "roc_data.pkl"), "wb") as fh:
        pickle.dump(roc_store, fh)
    with open(os.path.join(output_dir, "pr_data.pkl"), "wb") as fh:
        pickle.dump(pr_store, fh)

    print(f"[{analysis_name}] Results saved to: {output_dir}")


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expects multiple TSV files. Adds a subject_id per file.
    Drops columns: 'time', 'status', 'subject_id' from features if present.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.tsv"))
    if not all_files:
        raise FileNotFoundError(f"No .tsv files found under: {data_dir}")

    all_dfs = []
    for i, file_path in enumerate(sorted(all_files)):
        df = pd.read_csv(file_path, sep="\t")
        df["subject_id"] = i
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.dropna().reset_index(drop=True)

    y = df_all["status"].to_numpy()
    groups = df_all["subject_id"].to_numpy()
    X = df_all.drop(columns=["time", "status", "subject_id"], errors="ignore").to_numpy(dtype=float)

    return X, y, groups


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    print("--- Loading dataset ---")
    X, y, groups = load_dataset(DATA_DIR)
    print(f"Dataset loaded with {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(groups))} subjects.")

    for analysis_name in ANALYSES.keys():
        run_subject_aware_cv(X, y, groups, analysis_name)

    print("\nDone. All outputs written under:", OUTPUT_DIR_BASE)

