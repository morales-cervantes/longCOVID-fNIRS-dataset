
import os
import glob
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

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
# Script: 3_statistical_features_cv_full_report.py (UPDATED)
# Adds imbalance-robust metrics (PR-AUC, F1 per class, Macro-F1, MCC),
# generates Precision–Recall curves, and saves PR data.
# ------------------------------

# --- PART 1: FLATTEN DATA BY EXTRACTING STAT FEATURES ---

# 1. Paths
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results2\3_statistical_features_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Process each file and extract statistics (per subject)
print("--- Part 1: Flattening Data by Extracting Statistical Features ---")
all_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
all_summaries = []

for file_path in sorted(all_files):
    df = pd.read_csv(file_path, sep='\t')
    df.dropna(inplace=True)
    if df.empty:
        print(f"Warning: File {os.path.basename(file_path)} is empty after dropping NaNs. Skipping.")
        continue
    
    filename = os.path.basename(file_path)
    subject_id = filename.replace('_data.tsv', '')
    features_df = df.drop(columns=['time', 'status'], errors='ignore')
    status = df['status'].iloc[0]

    summary_data = {'Subject_ID': subject_id}
    # Aggregate stats per column
    summary_data.update({f'mean_{c}': features_df[c].mean() for c in features_df.columns})
    summary_data.update({f'std_{c}':  features_df[c].std()  for c in features_df.columns})
    summary_data.update({f'min_{c}':  features_df[c].min()  for c in features_df.columns})
    summary_data.update({f'max_{c}':  features_df[c].max()  for c in features_df.columns})
    summary_data['status'] = status

    all_summaries.append(summary_data)

# 3. Create flattened dataset
flattened_df = pd.DataFrame(all_summaries)
flattened_path = os.path.join(OUTPUT_DIR, 'flattened_statistical_dataset.csv')
flattened_df.to_csv(flattened_path, index=False)
print(f"Flattened dataset created with {len(flattened_df)} subjects. Saved to: {flattened_path}")

# --- PART 2: MACHINE LEARNING ANALYSIS ---

print("\n--- Part 2: Running Machine Learning Analysis on Flattened Data ---")

# 4. Define X and y
X = flattened_df.drop(columns=['Subject_ID', 'status'])
y = flattened_df['status']
n_samples = len(y)
print(f"Number of samples for ML: {n_samples}")

# 5. Models
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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nRunning analysis for 5-fold Cross-Validation...")

# 6. Evaluation loop
all_records = []
all_conf_mats = {}
all_roc_data = {}
all_pr_data  = {}

for model_name, pipe in models.items():
    conf_matrices_per_model = []
    roc_data_per_model = []
    pr_data_per_model  = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Probabilities (if available) for ROC/PR; fallback to decision_function or preds
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

# 7. Reports
df_all = pd.DataFrame(all_records)
csv_path = os.path.join(OUTPUT_DIR, 'full_results_stats_cv5_with_imbalance_metrics.csv')
df_all.to_csv(csv_path, index=False)

# Summary TXT
summary_path = os.path.join(OUTPUT_DIR, 'summary_stats_cv5_with_imbalance_metrics.txt')
with open(summary_path, 'w') as f:
    f.write("--- Summary for Statistical Features (CV5) ---\n\n")
    avg_metrics = df_all.groupby('Model').mean(numeric_only=True).drop(columns='Fold')
    f.write(avg_metrics.to_string())

print(f"Saved per-fold CSV to: {csv_path}")
print(f"Saved summary to: {summary_path}")

# Boxplot
plt.figure(figsize=(12, 7))
sns.boxplot(x='Model', y='Accuracy', data=df_all)
plt.title('Accuracy Distribution (Statistical Features, CV5)', fontsize=20)
plt.xlabel(None); plt.ylabel('Accuracy', fontsize=20)
plt.xticks(rotation=15, ha='right', fontsize=16); plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_boxplot_stats_cv5.png'), dpi=300)
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
axes[0].set_title(f'Best — {best_model_name} (by ROC–AUC = {best_val:.3f}, CV5', fontsize=20)
axes[0].set_xlabel('Predicted class', fontsize=20); axes[0].set_ylabel('True class', fontsize=20)
axes[0].set_xticklabels(tick_labels, fontsize=16); axes[0].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')

sns.heatmap(norm_cm_worst, annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=False, annot_kws={"size": 20})
axes[1].set_title(f'Worst — {worst_model_name} (by ROC–AUC = {worst_val:.3f}, CV5)', fontsize=20)
axes[1].set_xlabel('Predicted class', fontsize=20); axes[1].set_ylabel('True class', fontsize=20)
axes[1].set_xticklabels(tick_labels, fontsize=16); axes[1].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_best_worst_stats.png'), dpi=300)
plt.close()

# ROC Curves
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
plt.title('Average ROC Curves (Statistical Features - CV5)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc='lower right'); plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_stats_cv5.png'), dpi=300)
plt.close()

# Precision–Recall Curves
plt.figure(figsize=(10, 8))
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
plt.title('Average Precision–Recall Curves (Statistical Features - CV5)')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.legend(loc='lower left'); plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curves_stats_cv5.png'), dpi=300)
plt.close()

# Save PR data
pr_output_path = os.path.join(OUTPUT_DIR, 'pr_data.pkl')
with open(pr_output_path, 'wb') as f:
    pickle.dump(all_pr_data, f)

print(f"PR curve data saved to: {pr_output_path}")
print(f"\nScript 3 execution complete. Full reports saved in: {OUTPUT_DIR}")
