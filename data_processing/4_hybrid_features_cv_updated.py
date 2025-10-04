
import os
import glob
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score, f1_score, matthews_corrcoef,
    precision_recall_curve, auc as auc_func
)

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Script: 4_hybrid_features_cv.py (UPDATED)
# Adds imbalance-robust metrics (PR-AUC, F1 per class, Macro-F1, MCC),
# generates Precision–Recall curves, and saves PR data.
# ------------------------------

# 1. Paths
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results2\4_hybrid_features_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- PART 1: Load data and create hybrid features ---
print("--- Part 1: Creating Hybrid Features (Time-Series + Statistics) ---")
all_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
all_dfs = []

# Precompute per-subject stats
subject_stats = {}
for i, file_path in enumerate(sorted(all_files)):
    df = pd.read_csv(file_path, sep='\t')
    df.dropna(inplace=True)
    if df.empty:
        continue
    
    features_df = df.drop(columns=['time', 'status'], errors='ignore')
    stats = {
        'mean': features_df.mean(axis=0),
        'std': features_df.std(axis=0),
        'min': features_df.min(axis=0),
        'max': features_df.max(axis=0)
    }
    subject_stats[i] = stats

# Build hybrid dataframe (time-series rows + subject-level stats duplicated as features)
for i, file_path in enumerate(sorted(all_files)):
    df = pd.read_csv(file_path, sep='\t')
    df.dropna(inplace=True)
    if df.empty:
        continue
    
    df['subject_id'] = i
    stats_to_add = subject_stats[i]
    for stat_name, stat_series in stats_to_add.items():
        for col_name, value in stat_series.items():
            df[f'{stat_name}_{col_name}'] = value
    all_dfs.append(df)

global_df = pd.concat(all_dfs, ignore_index=True)
print("Hybrid dataset created.")

# 2. Define X, y, groups
X = global_df.drop(columns=['time', 'status', 'subject_id'], errors='ignore')
y = global_df['status']
groups = global_df['subject_id']
n_samples = len(y)
n_subjects = len(groups.unique())
print(f"Data loaded. Total samples: {n_samples} from {n_subjects} subjects.")

# --- PART 2: ML analysis ---
print("\n--- Part 2: Running Machine Learning Analysis on Hybrid Data ---")

# 3. Pipelines & CV
HYPERPARAMS = {
    'RandomForest': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
    'SVM': {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42},
    'KNN': {'n_neighbors': 5},
    'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42},
    'LogisticRegression': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
    'MLP': {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42}
}
models = {
    name: Pipeline([('scaler', StandardScaler()), ('clf', cls(**HYPERPARAMS[name]))])
    for name, cls in [('RandomForest', RandomForestClassifier), ('SVM', SVC), ('KNN', KNeighborsClassifier),
                      ('XGBoost', XGBClassifier), ('LogisticRegression', LogisticRegression), ('MLP', MLPClassifier)]
}
N_SPLITS = 5
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
print(f"\nUsing StratifiedGroupKFold with {N_SPLITS} splits.")

# 3.5. Inspect fold distributions (optional report)
print("\n--- Part 3.5: Analyzing data distribution in CV folds ---")
fold_distributions = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    fold_distributions.append({
        'Fold': fold,
        'Train_Controls': train_counts.get(0, 0),
        'Train_PostCOVID': train_counts.get(1, 0),
        'Test_Controls': test_counts.get(0, 0),
        'Test_PostCOVID': test_counts.get(1, 0)
    })
df_fold_dist = pd.DataFrame(fold_distributions)
print(df_fold_dist)

# 4. Evaluation
all_records = []
all_conf_mats = {}
all_roc_data = {}
all_pr_data = {}

for model_name, pipe in models.items():
    print(f"Processing model: {model_name}...")
    conf_matrices_per_model = []
    roc_data_per_model = []
    pr_data_per_model = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Probabilities for ROC/PR
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps['clf'], 'decision_function'):
            scores = pipe.decision_function(X_test)
            ranks = (scores.argsort().argsort()).astype(float)
            y_proba = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)
        else:
            y_proba = y_pred.astype(float)

        auc_val = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        acc = accuracy_score(y_test, y_pred)
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv  = tp / (tp + fp) if (tp + fp) else 0.0
        npv  = tn / (tn + fn) if (tn + fn) else 0.0

        # --- Imbalance-robust metrics ---
        try:
            pr_auc = average_precision_score(y_test, y_proba)
        except Exception:
            pr_auc = float('nan')

        f1_pos  = f1_score(y_test, y_pred, pos_label=1)
        f1_neg  = f1_score(y_test, y_pred, pos_label=0)
        macro_f1 = (f1_pos + f1_neg) / 2.0
        mcc     = matthews_corrcoef(y_test, y_pred)

        all_records.append({
            'Model': model_name, 'Fold': fold,
            'Accuracy': acc, 'Sensitivity': sens, 'Specificity': spec,
            'PPV': ppv, 'NPV': npv, 'AUC': auc_val,
            'PR_AUC': pr_auc, 'F1_Pos': f1_pos, 'F1_Neg': f1_neg,
            'Macro_F1': macro_f1, 'MCC': mcc
        })
        conf_matrices_per_model.append(np.array([[tn, fp], [fn, tp]]))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data_per_model.append({'fpr': fpr, 'tpr': tpr})

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_data_per_model.append({'precision': precision, 'recall': recall})

    all_conf_mats[model_name] = conf_matrices_per_model
    all_roc_data[model_name] = roc_data_per_model
    all_pr_data[model_name]  = pr_data_per_model

# 5. Reports
df_all = pd.DataFrame(all_records)
df_all.to_csv(os.path.join(OUTPUT_DIR, 'full_results_hybrid_cv_with_imbalance_metrics.csv'), index=False)

with open(os.path.join(OUTPUT_DIR, 'summary_hybrid_cv_with_imbalance_metrics.txt'), 'w') as f:
    f.write(f"--- Summary for Hybrid Features (Subject-Aware CV{N_SPLITS}) ---\n\n")
    f.write("--- Data Distribution per Fold ---\n")
    f.write(df_fold_dist.to_string(index=False))
    f.write("\n\n--- Average Model Performance Metrics ---\n")
    avg_metrics = df_all.groupby('Model').mean(numeric_only=True).drop(columns='Fold')
    f.write(avg_metrics.to_string())

# Boxplot (Accuracy)
plt.figure(figsize=(14, 10))
sns.boxplot(x='Model', y='Accuracy', data=df_all)
plt.xlabel(None)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(rotation=15, ha='right', fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_boxplot_hybrid_cv.png'), dpi=300)
plt.close()

# Best vs. Worst Confusion Matrix
PRIMARY_RANK = ['AUC', 'PR_AUC', 'Accuracy']  # orden de desempate

metric_means = (df_all
                .groupby('Model')[PRIMARY_RANK]
                .mean()
                .sort_values(by=PRIMARY_RANK, ascending=False))

best_model_name  = metric_means.index[0]
worst_model_name = metric_means.index[-1]

best_val  = metric_means.loc[best_model_name, 'AUC']
worst_val = metric_means.loc[worst_model_name, 'AUC']


avg_cm_best = np.mean(all_conf_mats[best_model_name], axis=0)
norm_cm_best = avg_cm_best / avg_cm_best.sum(axis=1, keepdims=True)

avg_cm_worst = np.mean(all_conf_mats[worst_model_name], axis=0)
norm_cm_worst = avg_cm_worst / avg_cm_worst.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
tick_labels = ['Control', 'Long COVID']
sns.heatmap(norm_cm_best, annot=True, fmt='.3f', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 20})
axes[0].set_title(f'Best — {best_model_name} (by ROC–AUC = {best_val:.3f}, CV{N_SPLITS})', fontsize=20)
axes[0].set_xlabel('Predicted class', fontsize=20)
axes[0].set_ylabel('True class', fontsize=20)
axes[0].set_xticklabels(tick_labels, fontsize=16)
axes[0].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')

sns.heatmap(norm_cm_worst, annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=False, annot_kws={"size": 20})
axes[1].set_title(f'Worst — {worst_model_name} (by ROC–AUC = {worst_val:.3f}, CV{N_SPLITS})', fontsize=20)
axes[1].set_xlabel('Predicted class', fontsize=20)
axes[1].set_ylabel('True class', fontsize=20)
axes[1].set_xticklabels(tick_labels, fontsize=16)
axes[1].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_best_worst_hybrid_cv.png'), dpi=300)
plt.close()

# ROC Curves
linestyles = ['-', '--', ':', '-.', (0, (5, 5)), (0, (3, 5, 1, 5))]
model_names = list(models.keys())

plt.figure(figsize=(12, 10))
for i, model_name in enumerate(model_names):
    roc_folds = all_roc_data[model_name]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = [np.interp(mean_fpr, fold['fpr'], fold['tpr']) for fold in roc_folds]
    if not tprs:
        continue
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0, 1
    mean_auc = df_all[df_all['Model'] == model_name]['AUC'].mean()
    plt.plot(mean_fpr, mean_tpr, linestyle=linestyles[i % len(linestyles)], label=f'{model_name} (AUC = {mean_auc:.3f})')

plt.plot([0, 1], [0, 1])
plt.title(f'Average ROC Curves (Hybrid Features, Subject-Aware CV{N_SPLITS})', fontsize=18)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_hybrid_cv.png'), dpi=300)
plt.close()

# Precision–Recall Curves
plt.figure(figsize=(12, 10))
for model_name, pr_folds in all_pr_data.items():
    recall_grid = np.linspace(0, 1, 100)
    precisions = []
    for fold in pr_folds:
        precision = fold['precision']
        recall = fold['recall']
        order = np.argsort(recall)
        r = recall[order]
        p = precision[order]
        p_interp = np.interp(recall_grid, r, p)
        precisions.append(p_interp)
    if not precisions:
        continue
    mean_precision = np.mean(np.vstack(precisions), axis=0)
    mean_pr_auc = auc_func(recall_grid, mean_precision)
    plt.plot(recall_grid, mean_precision, label=f'{model_name} (PR-AUC = {mean_pr_auc:.3f})')

plt.title(f'Average Precision–Recall Curves (Hybrid Features, Subject-Aware CV{N_SPLITS})', fontsize=18)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower left', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curves_hybrid_cv.png'), dpi=300)
plt.close()

# Save curve data
with open(os.path.join(OUTPUT_DIR, 'roc_data.pkl'), 'wb') as f:
    pickle.dump(all_roc_data, f)
with open(os.path.join(OUTPUT_DIR, 'pr_data.pkl'), 'wb') as f:
    pickle.dump(all_pr_data, f)

print(f"\nScript 4 execution complete. Outputs saved in: {OUTPUT_DIR}")
