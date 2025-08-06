import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold # <-- Importante
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Script: 7_1_subject_aware_pca_cv.py
# ------------------------------

# --- Función Auxiliar para ejecutar un análisis completo de CV por Grupos ---
def run_full_subject_cv_analysis(X_data, y_data, groups_data, analysis_name, output_dir_base):
    output_dir = os.path.join(output_dir_base, analysis_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Running Full Analysis for: {analysis_name} ---")
    print(f"Results will be saved in: {output_dir}")

    # Configurar modelos
    HYPERPARAMS = {
        'RandomForest': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
        'SVM': {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42},
        'KNN': {'n_neighbors': 5},
        'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42},
        'LogisticRegression': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
        'MLP': {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42}
    }
    models = {name: cls(**HYPERPARAMS[name]) for name, cls in [('RandomForest', RandomForestClassifier), ('SVM', SVC), ('KNN', KNeighborsClassifier), ('XGBoost', XGBClassifier), ('LogisticRegression', LogisticRegression), ('MLP', MLPClassifier)]}
    
    # Configurar CV por Grupos (n_splits debe ser <= al número de sujetos en la clase minoritaria)
    N_SPLITS = 5
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Bucle de Evaluación
    all_records = []
    all_conf_mats = {}
    all_roc_data = {}

    for model_name, model in models.items():
        conf_matrices_per_model = []
        roc_data_per_model = []
        # --- IMPORTANTE: Se pasa 'groups_data' al método split ---
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_data, y_data, groups_data), start=1):
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            all_records.append({'Model': model_name, 'Fold': fold, 'Accuracy': accuracy_score(y_test, y_pred), 'Sensitivity': tp / (tp + fn) if (tp + fn) else 0, 'Specificity': tn / (tn + fp) if (tn + fp) else 0, 'PPV': tp / (tp + fp) if (tp + fp) else 0, 'NPV': tn / (tn + fn) if (tn + fn) else 0, 'AUC': auc})
            conf_matrices_per_model.append(np.array([[tn, fp], [fn, tp]]))
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data_per_model.append({'fpr': fpr, 'tpr': tpr})

        all_conf_mats[model_name] = conf_matrices_per_model
        all_roc_data[model_name] = roc_data_per_model
    
    df_all = pd.DataFrame(all_records)
    
    # Generar todos los reportes y gráficos
    # Resumen TXT y CSV
    with open(os.path.join(output_dir, f'summary_{analysis_name}_subject_cv.txt'), 'w') as f:
        f.write(f"--- Summary for {analysis_name} (Subject-Aware CV{N_SPLITS}) ---\n\n")
        avg_metrics = df_all.groupby('Model').mean(numeric_only=True).drop(columns='Fold')
        f.write(avg_metrics.to_string())
    df_all.to_csv(os.path.join(output_dir, f'full_results_{analysis_name}_subject_cv.csv'), index=False)

    # Boxplot
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Model', y='Accuracy', data=df_all)
    plt.title(f'Accuracy Distribution ({analysis_name} - Subject-Aware CV{N_SPLITS})')
    plt.xticks(rotation=15, ha='right')
    plt.savefig(os.path.join(output_dir, f'accuracy_boxplot_{analysis_name}.png'), dpi=300)
    plt.close()

    # Matriz de Confusión Mejor vs. Peor
    model_avg_metrics = df_all.groupby('Model')['Accuracy'].mean()
    best_model_name = model_avg_metrics.idxmax()
    worst_model_name = model_avg_metrics.idxmin()
    avg_cm_best = np.mean(all_conf_mats[best_model_name], axis=0)
    norm_cm_best = avg_cm_best / avg_cm_best.sum(axis=1, keepdims=True)
    avg_cm_worst = np.mean(all_conf_mats[worst_model_name], axis=0)
    norm_cm_worst = avg_cm_worst / avg_cm_worst.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(norm_cm_best, annot=True, fmt='.3f', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f'Best CM — {best_model_name} (CV{N_SPLITS})')
    sns.heatmap(norm_cm_worst, annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title(f'Worst CM — {worst_model_name} (CV{N_SPLITS})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_best_worst_{analysis_name}.png'), dpi=300)
    plt.close()

    # Curva ROC
    plt.figure(figsize=(10, 8))
    for model_name, roc_folds in all_roc_data.items():
        mean_fpr = np.linspace(0, 1, 100)
        tprs = [np.interp(mean_fpr, fold['fpr'], fold['tpr']) for fold in roc_folds]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0], mean_tpr[-1] = 0, 1
        mean_auc = df_all[df_all['Model'] == model_name]['AUC'].mean()
        plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Average ROC Curves ({analysis_name} - Subject-Aware CV{N_SPLITS})')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.grid()
    plt.savefig(os.path.join(output_dir, f'roc_curves_{analysis_name}.png'), dpi=300)
    plt.close()
    
    # --- GUARDAR ROC DATA COMO PKL ---
    import pickle
    
    roc_output_path = os.path.join(output_dir, 'roc_data.pkl')
    with open(roc_output_path, 'wb') as f:
        pickle.dump(all_roc_data, f)
    
    print(f"ROC curve data saved to: {roc_output_path}")


# --- SCRIPT PRINCIPAL ---

# 1. Paths y Carga de Datos con IDs de sujeto
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR_BASE = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results\2_subject_aware_pca_cv"
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

print("--- Loading and Preparing Data with Subject IDs ---")
all_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
all_dfs = []
for i, file_path in enumerate(all_files):
    df = pd.read_csv(file_path, sep='\t')
    df['subject_id'] = i
    all_dfs.append(df)
global_df = pd.concat(all_dfs, ignore_index=True)
global_df.dropna(inplace=True)

X = global_df.drop(columns=['time', 'status', 'subject_id'], errors='ignore')
y = global_df['status']
groups = global_df['subject_id']
print(f"Dataset loaded with {len(X)} samples from {len(groups.unique())} subjects.")

# 2. Escalar datos y aplicar PCA
print("\n--- Performing PCA to determine components ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
n_components_95 = np.where(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95)[0][0] + 1
print(f"\n>>>>>> Number of components for 95% variance: {n_components_95} <<<<<<\n")

# 3. Crear los datasets de PCA
X_pca_95 = PCA(n_components=n_components_95, random_state=42).fit_transform(X_scaled)
X_pca_2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

# 4. Ejecutar los dos análisis completos con validación por grupos
run_full_subject_cv_analysis(X_pca_95, y, groups, "PCA_95_percent_variance", OUTPUT_DIR_BASE)
run_full_subject_cv_analysis(X_pca_2, y, groups, "PCA_2_components", OUTPUT_DIR_BASE)

# 5. Generar gráficos generales de PCA
print("\nGenerating final PCA plots...")
# (El código para los gráficos de varianza y scatter plot se mantiene igual que en el script 7_1)
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='.', linestyle='-')
plt.xlabel('Number of Components'); plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
plt.axvline(x=n_components_95 - 1, color='g', linestyle=':', label=f'{n_components_95} Components for 95%')
plt.legend(); plt.grid(axis='y')
plt.savefig(os.path.join(OUTPUT_DIR_BASE, 'pca_explained_variance.png'), dpi=300)
plt.close()

df_pca_2 = pd.DataFrame(data=X_pca_2, columns=['PCA1', 'PCA2'])
df_pca_2['status'] = y.values
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='status', data=df_pca_2, palette='viridis', s=60, alpha=0.7)
plt.title('Data Visualization in First 2 Principal Components')
plt.xlabel('Principal Component 1'); plt.ylabel('Principal Component 2')
plt.legend(title='Status (0=Control, 1=post-COVID)'); plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR_BASE, 'pca_scatter_plot.png'), dpi=300)
plt.close()


print(f"\nScript 2 execution complete. All outputs saved in: {OUTPUT_DIR_BASE}")
