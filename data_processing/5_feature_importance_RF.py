
"""
Script: 5_feature_importance_RF.py
Purpose:
    - Compute Random Forest feature importances to identify informative fNIRS features.
    - Uses a SUBJECT-LEVEL dataset (one row per subject) built from statistical summaries
      to avoid bias from repeated time-series rows.
    - Performs Stratified 5-Fold CV and averages importances across folds.
    - Saves a CSV with the full ranking and a barplot with the Top-N features.
    - Optionally produces a correlation heatmap for the Top-N features (for SI/appendix).

Inputs:
    - Folder with TSV files, each containing columns like: time, status, and channels.
    - This script mirrors the data loading logic used in scripts 3_ and 4_,
      but condenses to subject-level for clearer interpretability.

Outputs:
    - feature_importances_cv_mean.csv       (Full ranking, mean across folds)
    - feature_importances_cv_std.csv        (Std across folds, same order as mean)
    - top15_feature_importance_barplot.png
    - top15_feature_corr_heatmap.png        (optional; only if >= 15 features exist)
    - run_summary.txt                       (basic run info)

How to run:
    - Edit DATA_DIR and OUTPUT_DIR below according to your environment.
    - Run:  python 5_feature_importance_RF.py

Author: Dr. Antony Morales-Cervantes (pipeline style aligned to scripts 1–4)
Date: 2025-10-09
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import List, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, confusion_matrix

import matplotlib.pyplot as plt

# =========================
# Configuration (EDIT)
# =========================
DATA_DIR  = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results2\5_feature_importance_RF"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
TOP_N = 15
N_TREES = 1000

# =========================
# Utilities
# =========================

def build_subject_level_dataset(tsv_files: List[str]) -> pd.DataFrame:
    """
    Build one-row-per-subject dataset using statistical summaries per channel.
    Keeps interpretability high and avoids duplication bias from time-series rows.
    """
    rows = []
    for file_path in sorted(tsv_files):
        df = pd.read_csv(file_path, sep="\t")
        df = df.dropna()
        if df.empty:
            continue

        # Identify columns
        feat_df = df.drop(columns=["time", "status"], errors="ignore")
        status = int(df["status"].iloc[0]) if "status" in df.columns else None

        # Basic statistical summaries per channel
        summary = {}
        for col in feat_df.columns:
            x = feat_df[col].values
            summary[f"mean_{col}"] = float(np.mean(x))
            summary[f"std_{col}"]  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            summary[f"min_{col}"]  = float(np.min(x))
            summary[f"max_{col}"]  = float(np.max(x))

        # Subject id from filename
        fname = os.path.basename(file_path)
        subj_id = os.path.splitext(fname)[0]
        summary["Subject_ID"] = subj_id
        summary["status"]     = status
        rows.append(summary)

    subjects_df = pd.DataFrame(rows)
    subjects_df = subjects_df.dropna(axis=1, how="all")  # drop empty columns if any
    return subjects_df


def average_importances_across_folds(importances_list: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """Return mean and std of feature importances across folds."""
    M = np.vstack(importances_list)   # shape: (n_folds, n_features)
    return M.mean(axis=0), M.std(axis=0)


# =========================
# Main
# =========================

def main():
    print("[INFO] Loading TSV files...")
    tsv_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found under: {DATA_DIR}")

    print(f"[INFO] Found {len(tsv_files)} files. Building subject-level dataset...")
    df = build_subject_level_dataset(tsv_files)
    print(f"[INFO] Dataset shape: {df.shape} (rows=subjects)")
    df.to_csv(os.path.join(OUTPUT_DIR, "subject_level_dataset.csv"), index=False)

    # Features / labels
    X = df.drop(columns=["Subject_ID", "status"], errors="ignore")
    y = df["status"].astype(int).values
    feature_names = list(X.columns)
    X = X.to_numpy(dtype=float)

    # Cross-validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    fold_importances = []
    metrics_rows = []

    print("[INFO] Running Stratified 5-Fold CV for feature importances...")
    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        rf.fit(Xtr, ytr)
        ypred = rf.predict(Xte)
        yproba = rf.predict_proba(Xte)[:, 1]

        # Store feature importances for this fold
        fold_importances.append(rf.feature_importances_.copy())

        # Basic metrics (for log)
        acc = accuracy_score(yte, ypred)
        try:
            auc = roc_auc_score(yte, yproba)
        except Exception:
            auc = float("nan")
        try:
            pr_auc = average_precision_score(yte, yproba)
        except Exception:
            pr_auc = float("nan")
        f1p = f1_score(yte, ypred, pos_label=1)
        f1n = f1_score(yte, ypred, pos_label=0)
        mcc = matthews_corrcoef(yte, ypred)
        tn, fp, fn, tp = confusion_matrix(yte, ypred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        metrics_rows.append({
            "Fold": fold_idx, "ACC": acc, "AUC": auc, "PR_AUC": pr_auc,
            "F1_Pos": f1p, "F1_Neg": f1n, "MCC": mcc, "SENS": sens, "SPEC": spec
        })

    # Aggregate importances
    mean_imp, std_imp = average_importances_across_folds(fold_importances)

    # Build DataFrame with ranking
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": mean_imp,
        "importance_std": std_imp
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # Save full ranking
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances_cv_mean.csv"), index=False)
    std_sorted = imp_df[["feature", "importance_std"]]
    std_sorted.to_csv(os.path.join(OUTPUT_DIR, "feature_importances_cv_std.csv"), index=False)

    # Save CV metrics log
    met_df = pd.DataFrame(metrics_rows)
    met_df.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics_log.csv"), index=False)

    # --- Plot Top-N barplot
    topk = imp_df.head(TOP_N)
    plt.figure(figsize=(10, max(6, 0.4 * len(topk))))
    plt.barh(topk["feature"][::-1], topk["importance_mean"][::-1])
    plt.xlabel("Mean Importance (Random Forest)")
    plt.title(f"Top-{TOP_N} Informative Features (Subject-Level, CV{N_SPLITS})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top15_feature_importance_barplot.png"), dpi=300)
    plt.close()

    # --- Optional: correlation heatmap for Top-N features
    if len(topk) >= min(TOP_N, 5):
        sel_feats = list(topk["feature"])
        corr = pd.DataFrame(df[sel_feats]).corr().values
        plt.figure(figsize=(8, 7))
        im = plt.imshow(corr, interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(ticks=range(len(sel_feats)), labels=sel_feats, rotation=90)
        plt.yticks(ticks=range(len(sel_feats)), labels=sel_feats)
        plt.title("Correlation Heatmap — Top Features")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "top15_feature_corr_heatmap.png"), dpi=300)
        plt.close()

    # Summary file
    with open(os.path.join(OUTPUT_DIR, "run_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Subject-level rows: {df.shape[0]}\n")
        fh.write(f"Total features: {len(feature_names)}\n")
        fh.write(f"RandomForest trees: {N_TREES}\n")
        fh.write(f"CV splits: {N_SPLITS}\n\n")
        fh.write("CV metrics (per fold):\n")
        fh.write(met_df.to_string(index=False))

    print(f"[DONE] Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
