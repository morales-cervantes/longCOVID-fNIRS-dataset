
import os
import glob
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
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Script: 8_statistical_features_cv_full_report.py
# ------------------------------

# --- PARTE 1: APLANAMIENTO AUTOMÁTICO DE DATOS ---

# 1. Paths
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results\8_statistical_features_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Bucle para procesar cada archivo y extraer estadísticas
print("--- Part 1: Flattening Data by Extracting Statistical Features ---")
all_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
all_summaries = []

for file_path in all_files:
    df = pd.read_csv(file_path, sep='\t')
    df.dropna(inplace=True)
    if df.empty:
        print(f"Warning: File {os.path.basename(file_path)} is empty after dropping NaNs. Skipping.")
        continue
    
    filename = os.path.basename(file_path)
    subject_id = filename.replace('_data.tsv', '')
    features_df = df.drop(columns=['time', 'status'], errors='ignore')
    status = df['status'].iloc[0]

    mean_values = features_df.mean(axis=0)
    std_values = features_df.std(axis=0)
    min_values = features_df.min(axis=0)
    max_values = features_df.max(axis=0)

    summary_data = {'Subject_ID': subject_id}
    for col in features_df.columns:
        summary_data[f'mean_{col}'] = mean_values[col]
        summary_data[f'std_{col}'] = std_values[col]
        summary_data[f'min_{col}'] = min_values[col]
        summary_data[f'max_{col}'] = max_values[col]
    
    summary_data['status'] = status
    all_summaries.append(summary_data)

# 3. Crear el DataFrame aplanado final
flattened_df = pd.DataFrame(all_summaries)
flattened_df.to_csv(os.path.join(OUTPUT_DIR, 'flattened_statistical_dataset.csv'), index=False)
print(f"Flattened dataset created with {len(flattened_df)} subjects.")

# --- PARTE 2: ANÁLISIS CON MACHINE LEARNING ---

print("\n--- Part 2: Running Machine Learning Analysis on Flattened Data ---")

# 4. Definir X e y desde el nuevo dataset
X = flattened_df.drop(columns=['Subject_ID', 'status'])
y = flattened_df['status']
n_samples = len(y)
print(f"Number of samples for ML: {n_samples}")

# 5. Configurar modelos
HYPERPARAMS = {
    'RandomForest': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
    'SVM': {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42},
    'KNN': {'n_neighbors': 5},
    'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42},
    'LogisticRegression': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
    'MLP': {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42}
}
models = {name: Pipeline([('scaler', StandardScaler()), ('clf', cls(**HYPERPARAMS[name]))]) for name, cls in [('RandomForest', RandomForestClassifier), ('SVM', SVC), ('KNN', KNeighborsClassifier), ('XGBoost', XGBClassifier), ('LogisticRegression', LogisticRegression), ('MLP', MLPClassifier)]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nRunning analysis for 5-fold Cross-Validation...")

# 6. Bucle de Evaluación
all_records = []
all_conf_mats = {}
all_roc_data = {} # <-- AÑADIDO: para guardar datos de ROC

for model_name, pipe in models.items():
    conf_matrices_per_model = []
    roc_data_per_model = [] # <-- AÑADIDO
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        all_records.append({'Model': model_name, 'Fold': fold, 'Accuracy': accuracy_score(y_test, y_pred), 'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0, 'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0, 'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0, 'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0, 'AUC': auc})
        conf_matrices_per_model.append(np.array([[tn, fp], [fn, tp]]))
        
        fpr, tpr, _ = roc_curve(y_test, y_proba) # <-- AÑADIDO
        roc_data_per_model.append({'fpr': fpr, 'tpr': tpr}) # <-- AÑADIDO

    all_conf_mats[model_name] = conf_matrices_per_model
    all_roc_data[model_name] = roc_data_per_model # <-- AÑADIDO

# 7. Generación de Gráficos y Reportes
df_all = pd.DataFrame(all_records)
df_all.to_csv(os.path.join(OUTPUT_DIR, 'full_results_stats_cv5.csv'), index=False) # <-- AÑADIDO: CSV con resultados por pliegue

# --- AÑADIDO: Resumen en TXT ---
with open(os.path.join(OUTPUT_DIR, 'summary_stats_cv5.txt'), 'w') as f:
    f.write("--- Summary for Statistical Features (CV5) ---\n\n")
    avg_metrics = df_all.groupby('Model').mean(numeric_only=True).drop(columns='Fold')
    f.write(avg_metrics.to_string())

# Boxplot
plt.figure(figsize=(12, 7))
sns.boxplot(x='Model', y='Accuracy', data=df_all)
plt.title('Accuracy Distribution (Statistical Features, CV5)', fontsize=20)
plt.xlabel(None); plt.ylabel('Accuracy', fontsize=20)
plt.xticks(rotation=15, ha='right', fontsize=16); plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_boxplot_stats_cv5.png'), dpi=300)
plt.close()

# Mejor vs. Peor Matriz de Confusión
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
axes[0].set_title(f'Best CM — {best_model_name} (CV5)', fontsize=20)
axes[0].set_xlabel('Predicted class', fontsize=20); axes[0].set_ylabel('True class', fontsize=20)
axes[0].set_xticklabels(tick_labels, fontsize=16); axes[0].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')
sns.heatmap(norm_cm_worst, annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=False, annot_kws={"size": 20})
axes[1].set_title(f'Worst CM — {worst_model_name} (CV5)', fontsize=20)
axes[1].set_xlabel('Predicted class', fontsize=20); axes[1].set_ylabel('True class', fontsize=20)
axes[1].set_xticklabels(tick_labels, fontsize=16); axes[1].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_best_worst_stats.png'), dpi=300)
plt.close()

# --- AÑADIDO: Curva ROC ---
plt.figure(figsize=(10, 8))
for model_name, roc_folds in all_roc_data.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = [np.interp(mean_fpr, fold['fpr'], fold['tpr']) for fold in roc_folds]
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

print(f"\nScript 8 execution complete. Full reports saved in: {OUTPUT_DIR}")
