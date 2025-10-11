
"""
Script: 6_pca_loadings_analysis.py
Purpose:
    - Perform PCA on a SUBJECT-LEVEL dataset (one row per subject, statistical summaries per channel).
    - Use Stratified 5-Fold CV to fit PCA ONLY on training folds (no leakage).
    - Compute per-feature contribution scores from PCA loadings, weighted by explained variance ratio.
    - Aggregate contributions across folds; output ranked features and aggregated per-channel scores.
    - Produce figures: Top-N loadings barplot and (optional) PC1–PC2 scatter/biplot for visualization.

Why subject-level?
    - Matches the subject-aware evaluation strategy of the article.
    - Avoids inflating interpretability by repeated time rows.

Outputs:
    - pca_feature_contributions_mean.csv       (feature-level ranking; mean across folds)
    - pca_feature_contributions_std.csv        (std across folds)
    - pca_channel_contributions_mean.csv       (aggregated by base channel name)
    - top15_pca_feature_loadings_barplot.png
    - (optional) pc1_pc2_scatter.png           (exploratory viz on full data; fit noted in code comments)
    - run_summary.txt

How to run:
    - Edit DATA_DIR and OUTPUT_DIR below.
    - Run: python 6_pca_loadings_analysis.py

Author: Dr. Antony Morales-Cervantes (pipeline style aligned to scripts 1–5)
Date: 2025-10-09
"""

import os
import glob
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# =========================
# Configuration (EDIT)
# =========================
DATA_DIR   = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results2\6_pca_loadings_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
TOP_N = 15
# PCA components selection: keep 95% variance; also report first K for reference
PCA_N_COMPONENTS = 0.95
TOP_K_FOR_SCORE  = None  # if None, use all components retained by PCA; else use first K

# =========================
# Utilities
# =========================

def build_subject_level_dataset(tsv_files: List[str]) -> pd.DataFrame:
    """Create one-row-per-subject with basic stats for each time-series channel."""
    rows = []
    for file_path in sorted(tsv_files):
        df = pd.read_csv(file_path, sep="\t").dropna()
        if df.empty:
            continue
        feat_df = df.drop(columns=["time", "status"], errors="ignore")
        status = int(df["status"].iloc[0]) if "status" in df.columns else None

        summary = {}
        for col in feat_df.columns:
            x = feat_df[col].values
            summary[f"mean_{col}"] = float(np.mean(x))
            summary[f"std_{col}"]  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            summary[f"min_{col}"]  = float(np.min(x))
            summary[f"max_{col}"]  = float(np.max(x))

        fname = os.path.basename(file_path)
        subj_id = os.path.splitext(fname)[0]
        summary["Subject_ID"] = subj_id
        summary["status"]     = status
        rows.append(summary)

    subjects_df = pd.DataFrame(rows)
    subjects_df = subjects_df.dropna(axis=1, how="all")
    return subjects_df


def parse_base_channel(feature_name: str) -> str:
    """
    Map feature names like 'mean_Ch01' or 'std_FP2' to base channel 'Ch01' or 'FP2'.
    If no '_' found, return the original feature name.
    """
    if "_" in feature_name:
        return feature_name.split("_", 1)[1]
    return feature_name


def contribution_from_pca(components_: np.ndarray, evr_: np.ndarray, top_k: int = None) -> np.ndarray:
    """
    Compute per-feature contribution score from PCA loadings weighted by explained variance ratio.
    Score_j = sum_{i in PCs} |loading_{i,j}| * EVR_i
    Using absolute loadings captures magnitude of participation irrespective of sign.
    """
    n_pc = components_.shape[0]
    if top_k is not None:
        n_use = min(top_k, n_pc)
    else:
        n_use = n_pc
    # components_ shape: (n_components, n_features)
    loadings = np.abs(components_[:n_use, :])  # abs loadings
    weights  = evr_[:n_use].reshape(-1, 1)     # EVR weights
    scores = (loadings * weights).sum(axis=0)
    return scores


# =========================
# Main
# =========================

def main():
    print("[INFO] Loading TSV files...")
    tsv_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found under: {DATA_DIR}")

    print(f"[INFO] Building subject-level dataset from {len(tsv_files)} files...")
    df = build_subject_level_dataset(tsv_files)
    df.to_csv(os.path.join(OUTPUT_DIR, "subject_level_dataset.csv"), index=False)
    print(f"[INFO] Dataset shape: {df.shape} (subjects x features+meta)")

    # Prepare X/y
    X_df = df.drop(columns=["Subject_ID", "status"], errors="ignore")
    y = df["status"].astype(int).to_numpy()
    feature_names = list(X_df.columns)

    X = X_df.to_numpy(dtype=float)

    # CV setup
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_scores = []
    fold_evr_list = []
    fold_ncomp = []
    fold_pc1_pc2 = []  # store for optional later plots

    print("[INFO] Running Stratified 5-Fold CV for PCA loadings...")
    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        scaler = StandardScaler()
        Xtr_z = scaler.fit_transform(Xtr)
        Xte_z = scaler.transform(Xte)

        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
        Xtr_p = pca.fit_transform(Xtr_z)
        Xte_p = pca.transform(Xte_z)

        # Contribution scores for this fold
        scores = contribution_from_pca(pca.components_, pca.explained_variance_ratio_, TOP_K_FOR_SCORE)
        fold_scores.append(scores)
        fold_evr_list.append(pca.explained_variance_ratio_)
        fold_ncomp.append(pca.n_components_)

        # Save first 2 PCs scores (test only) for potential per-fold scatter (not plotted by default)
        if pca.n_components_ >= 2:
            fold_pc1_pc2.append({
                "fold": fold_idx,
                "PC1_test": Xte_p[:, 0],
                "PC2_test": Xte_p[:, 1],
                "y_test": yte
            })

    # Aggregate scores across folds
    S = np.vstack(fold_scores)  # (n_folds, n_features)
    mean_scores = S.mean(axis=0)
    std_scores  = S.std(axis=0)

    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "pca_contrib_mean": mean_scores,
        "pca_contrib_std": std_scores
    }).sort_values("pca_contrib_mean", ascending=False).reset_index(drop=True)

    contrib_df.to_csv(os.path.join(OUTPUT_DIR, "pca_feature_contributions_mean.csv"), index=False)
    contrib_df[["feature", "pca_contrib_std"]].to_csv(os.path.join(OUTPUT_DIR, "pca_feature_contributions_std.csv"), index=False)

    # Aggregate at channel level
    channel_scores: Dict[str, float] = {}
    channel_scores_list: Dict[str, List[float]] = {}

    for feat, score in zip(contrib_df["feature"], contrib_df["pca_contrib_mean"]):
        ch = parse_base_channel(feat)
        channel_scores[ch] = channel_scores.get(ch, 0.0) + score
        channel_scores_list.setdefault(ch, []).append(score)

    channel_mean = {ch: np.mean(vals) for ch, vals in channel_scores_list.items()}
    channel_std  = {ch: np.std(vals) for ch, vals in channel_scores_list.items()}
    channel_df = pd.DataFrame({
        "channel": list(channel_mean.keys()),
        "pca_channel_contrib_mean": list(channel_mean.values()),
        "pca_channel_contrib_std":  list(channel_std.values())
    }).sort_values("pca_channel_contrib_mean", ascending=False).reset_index(drop=True)

    channel_df.to_csv(os.path.join(OUTPUT_DIR, "pca_channel_contributions_mean.csv"), index=False)

    # Plot Top-N features
    topk = contrib_df.head(TOP_N)
    plt.figure(figsize=(10, max(6, 0.4 * len(topk))))
    plt.barh(topk["feature"][::-1], topk["pca_contrib_mean"][::-1])
    plt.xlabel("PCA Contribution (|loading| × EVR, mean across folds)")
    plt.title(f"Top-{TOP_N} PCA Feature Contributions (Subject-Level, CV{N_SPLITS})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top15_pca_feature_loadings_barplot.png"), dpi=300)
    plt.close()

    # Optional: PC1–PC2 scatter (fit on full standardized data for a compact visualization)
    # Note: This is exploratory; interpret with caution regarding leakage in visualization only.
    scaler_full = StandardScaler()
    Xz_full = scaler_full.fit_transform(X)
    pca_full = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp_full = pca_full.fit_transform(Xz_full)

    plt.figure(figsize=(7, 6))
    for cls, marker in zip([0, 1], ["o", "s"]):
        idx = np.where(y == cls)[0]
        plt.scatter(Xp_full[idx, 0], Xp_full[idx, 1], marker=marker, label=("Control" if cls == 0 else "Post-COVID"))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1–PC2 Scatter (Exploratory, full fit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pc1_pc2_scatter.png"), dpi=300)
    plt.close()

    # Summary
    with open(os.path.join(OUTPUT_DIR, "run_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Subjects: {X.shape[0]}\n")
        fh.write(f"Features: {X.shape[1]}\n")
        fh.write(f"CV folds: {N_SPLITS}\n")
        fh.write(f"PCA n_components setting: {PCA_N_COMPONENTS}\n")
        fh.write(f"Mean retained components across folds: {np.mean(fold_ncomp):.2f}\n")
        # Mean EVR for first 5 components (if available)
        max_len = max(len(evr) for evr in fold_evr_list)
        mean_evr = np.zeros(max_len)
        counts = np.zeros(max_len)
        for evr in fold_evr_list:
            for i, v in enumerate(evr):
                mean_evr[i] += v
                counts[i] += 1
        mean_evr = np.divide(mean_evr, counts, out=np.zeros_like(mean_evr), where=counts>0)
        top_show = min(5, len(mean_evr))
        fh.write("Mean explained variance ratio (first PCs):\n")
        for i in range(top_show):
            fh.write(f"  PC{i+1}: {mean_evr[i]:.4f}\n")

    print(f"[DONE] Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
