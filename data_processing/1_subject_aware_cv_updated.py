
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
# Script: 1_subject_aware_cv.py (UPDATED)
# Adds PR-AUC, class-wise F1, Macro-F1, and MCC.
# Also generates Precision-Recall curves and saves PR data.
# ------------------------------

# 1. Hyperparameters
HYPERPARAMS = {
    'RandomForest': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
    'SVM': {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42},
    'KNN': {'n_neighbors': 5},
    'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1,
                'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42},
    'LogisticRegression': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
    'MLP': {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42}
}

# 2. File paths
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results2\1_subject_aware_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Load data and assign subject IDs
print("Loading data and assigning subject IDs...")
all_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
all_dfs = []
for i, file_path in enumerate(all_files):
    df = pd.read_csv(file_path, sep='\t')
    df['subject_id'] = i
    all_dfs.append(df)

global_df = pd.concat(all_dfs, ignore_index=True)
global_df.dropna(inplace=True)

# 4. Define X, y, and groups for CV
X = global_df.drop(columns=['time', 'status', 'subject_id'], errors='ignore')
y = global_df['status']
groups = global_df['subject_id']
n_samples = len(y)
n_subjects = len(groups.unique())
print(f"Data loaded. Total samples: {n_samples} from {n_subjects} subjects.")

# 5. Build pipelines
models = {
    name: Pipeline([('scaler', StandardScaler()), ('clf', cls(**HYPERPARAMS[name]))])
    for name, cls in [('RandomForest', RandomForestClassifier), ('SVM', SVC),
                      ('KNN', KNeighborsClassifier), ('XGBoost', XGBClassifier),
                      ('LogisticRegression', LogisticRegression), ('MLP', MLPClassifier)]
}

# 6. Stratified Group K-Fold
N_SPLITS = 5
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
print(f"\nUsing StratifiedGroupKFold with {N_SPLITS} splits.")

# 7. Evaluation loop
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

        # Probabilities for positive class if available
        if hasattr(pipe.named_steps['clf'], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps['clf'], "decision_function"):
            # Map decision_function scores to [0,1] via rank normalization
            scores = pipe.decision_function(X_test)
            ranks = (scores.argsort().argsort()).astype(float)
            y_proba = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)
        else:
            # Fallback (not ideal): use predictions as pseudo-probabilities
            y_proba = y_pred.astype(float)

        # ROC AUC
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        acc = accuracy_score(y_test, y_pred)
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0

        # --- Imbalance-robust metrics ---
        try:
            pr_auc = average_precision_score(y_test, y_proba)
        except Exception:
            pr_auc = float('nan')

        f1_pos = f1_score(y_test, y_pred, pos_label=1)
        f1_neg = f1_score(y_test, y_pred, pos_label=0)
        macro_f1 = (f1_pos + f1_neg) / 2.0
        mcc = matthews_corrcoef(y_test, y_pred)

        all_records.append({
            'Model': model_name,
            'Fold': fold,
            'Accuracy': acc,
            'Sensitivity': sens,
            'Specificity': spec,
            'PPV': ppv,
            'NPV': npv,
            'AUC': auc,
            'PR_AUC': pr_auc,
            'F1_Pos': f1_pos,
            'F1_Neg': f1_neg,
            'Macro_F1': macro_f1,
            'MCC': mcc
        })

        conf_matrices_per_model.append(np.array([[tn, fp], [fn, tp]]))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data_per_model.append({'fpr': fpr, 'tpr': tpr})

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_data_per_model.append({'precision': precision, 'recall': recall})

    all_conf_mats[model_name] = conf_matrices_per_model
    all_roc_data[model_name] = roc_data_per_model
    all_pr_data[model_name] = pr_data_per_model

# 8. Reports
df_all = pd.DataFrame(all_records)
df_all.to_csv(os.path.join(OUTPUT_DIR, 'full_results_subject_cv_with_imbalance_metrics.csv'), index=False)

# Summary TXT (means across folds)
with open(os.path.join(OUTPUT_DIR, 'summary_subject_cv_with_imbalance_metrics.txt'), 'w') as f:
    f.write(f"--- Summary for Subject-Aware Cross-Validation (CV{N_SPLITS}) ---\n\n")
    avg_metrics = df_all.groupby('Model').mean(numeric_only=True).drop(columns='Fold')
    f.write(avg_metrics.to_string())

# Boxplot (Accuracy)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Model', y='Accuracy', data=df_all)
plt.title(f'Accuracy Distribution (Subject-Aware CV{N_SPLITS})', fontsize=20)
plt.xlabel(None); plt.ylabel('Accuracy', fontsize=20)
plt.xticks(rotation=15, ha='right', fontsize=16); plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_boxplot_subject_cv.png'), dpi=300)
plt.close()

# Best vs. Worst Confusion Matrix (by Accuracy)
model_avg_metrics = df_all.groupby('Model')['Accuracy'].mean()
best_model_name = model_avg_metrics.idxmax()
worst_model_name = model_avg_metrics.idxmin()

avg_cm_best = np.mean(all_conf_mats[best_model_name], axis=0)
norm_cm_best = avg_cm_best / avg_cm_best.sum(axis=1, keepdims=True)

avg_cm_worst = np.mean(all_conf_mats[worst_model_name], axis=0)
norm_cm_worst = avg_cm_worst / avg_cm_worst.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
tick_labels = ['Control', 'Long COVID']
sns.heatmap(norm_cm_best, annot=True, fmt='.3f', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 20})
axes[0].set_title(f'Best CM — {best_model_name} (CV{N_SPLITS})', fontsize=20)
axes[0].set_xlabel('Predicted class', fontsize=20); axes[0].set_ylabel('True class', fontsize=20)
axes[0].set_xticklabels(tick_labels, fontsize=16); axes[0].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')

sns.heatmap(norm_cm_worst, annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=False, annot_kws={"size": 20})
axes[1].set_title(f'Worst CM — {worst_model_name} (CV{N_SPLITS})', fontsize=20)
axes[1].set_xlabel('Predicted class', fontsize=20); axes[1].set_ylabel('True class', fontsize=20)
axes[1].set_xticklabels(tick_labels, fontsize=16); axes[1].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_best_worst_subject_cv.png'), dpi=300)
plt.close()

# Average ROC Curves
plt.figure(figsize=(10, 8))
for model_name, roc_folds in all_roc_data.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = [np.interp(mean_fpr, fold['fpr'], fold['tpr']) for fold in roc_folds]
    if not tprs:
        continue
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0, 1
    mean_auc = df_all[df_all['Model'] == model_name]['AUC'].mean()
    plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f'Average ROC Curves (Subject-Aware CV{N_SPLITS})')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc='lower right'); plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_subject_cv.png'), dpi=300)
plt.close()

# Average Precision-Recall Curves
plt.figure(figsize=(10, 8))
for model_name, pr_folds in all_pr_data.items():
    recall_grid = np.linspace(0, 1, 100)
    precisions = []
    for fold in pr_folds:
        precision = fold['precision']
        recall = fold['recall']
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        prec_interp = np.interp(recall_grid, recall_sorted, precision_sorted)
        precisions.append(prec_interp)
    if not precisions:
        continue
    mean_precision = np.mean(np.vstack(precisions), axis=0)
    mean_pr_auc = auc_func(recall_grid, mean_precision)
    plt.plot(recall_grid, mean_precision, label=f'{model_name} (PR-AUC = {mean_pr_auc:.3f})')
plt.title(f'Average Precision–Recall Curves (Subject-Aware CV{N_SPLITS})')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.legend(loc='lower left'); plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curves_subject_cv.png'), dpi=300)
plt.close()

# Save ROC & PR data
with open(os.path.join(OUTPUT_DIR, 'roc_data.pkl'), 'wb') as f:
    pickle.dump(all_roc_data, f)
with open(os.path.join(OUTPUT_DIR, 'pr_data.pkl'), 'wb') as f:
    pickle.dump(all_pr_data, f)

print(f"Outputs saved in: {OUTPUT_DIR}")
