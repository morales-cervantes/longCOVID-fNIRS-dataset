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
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Script: 3_2_global_timeseries_cv.py
# ------------------------------

# 1. Hyperparameters (se mantienen)
HYPERPARAMS = {
    'RandomForest': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
    'SVM': {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42},
    'KNN': {'n_neighbors': 5},
    'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1,
                'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42},
    'LogisticRegression': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
    'MLP': {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42}
}

# 2. File paths (actualizado para 3_2)
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results\6_global_timeseries_cv" # <-- Directorio de salida actualizado
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Read and concatenate all .tsv files (se mantiene)
all_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
global_df = pd.concat([pd.read_csv(fp, sep='\t') for fp in all_files], ignore_index=True)

# 4. Define features X and label y (se mantiene)
X = global_df.drop(columns=['time', 'status'], errors='ignore')
y = global_df['status']
n_samples = len(y)


# Después de definir X e y:
y = global_df['status']
n_missing = y.isna().sum()
print(f"Muestras sin etiqueta en 'status': {n_missing}")

# Eliminar filas con status faltante
mask = y.notna()
X = X[mask]
y = y[mask]

n_samples = len(y)
print(f"Número de muestras tras limpieza: {n_samples}")


# 5. Configure pipelines (se mantiene)
models = {
    name: Pipeline([('scaler', StandardScaler()), ('clf', cls(**HYPERPARAMS[name]))])
    for name, cls in [('RandomForest', RandomForestClassifier), ('SVM', SVC),
                      ('KNN', KNeighborsClassifier), ('XGBoost', XGBClassifier),
                      ('LogisticRegression', LogisticRegression), ('MLP', MLPClassifier)]
}

# 6. Stratified CV settings (se mantiene)
cvs = {
    'cv5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'cv10': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
}

# 7. Evaluation loop
# --- INICIO DE MODIFICACIÓN: Acumuladores globales ---
all_records = []
all_conf_mats = {}
# --- FIN DE MODIFICACIÓN ---

for cv_name, cv in cvs.items():
    records = []
    roc_points = {m: [] for m in models}
    conf_matrices = {m: [] for m in models}
    fold_dists = []
    for _, test_idx in cv.split(X, y):
        fold_dists.append(y.iloc[test_idx].value_counts().to_dict())

    for model_name, pipe in models.items():
        # Limpiar listas para el modelo actual dentro del bucle de CV
        conf_matrices[model_name] = []
        roc_points[model_name] = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            records.append({
                'Model': model_name, 'CV': cv_name, 'Fold': fold,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Sensitivity': tp / (tp + fn) if (tp + fn) else 0,
                'Specificity': tn / (tn + fp) if (tn + fp) else 0,
                'PPV': tp / (tp + fp) if (tp + fp) else 0,
                'NPV': tn / (tn + fn) if (tn + fn) else 0,
                'AUC': roc_auc_score(y_test, y_proba)
            })
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_points[model_name].append((fpr, tpr))
            conf_matrices[model_name].append(np.array([[tn, fp], [fn, tp]]))

        # --- INICIO DE MODIFICACIÓN: Guardar matrices de confusión por modelo y CV ---
        all_conf_mats[(model_name, cv_name)] = conf_matrices[model_name]
        # --- FIN DE MODIFICACIÓN ---

    # El resto del bucle para generar archivos y gráficos individuales se mantiene
    df_metrics = pd.DataFrame(records)
    all_records.extend(records) # Acumular registros para el DataFrame final

    # Save summary txt
    txt_path = os.path.join(OUTPUT_DIR, f'metrics_summary_{cv_name}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Global samples: {n_samples}\n")
        f.write("Fold label distributions:\n")
        for i, d in enumerate(fold_dists, 1):
            f.write(f" Fold {i}: {d}\n")
        f.write("\nAverage metrics per model:\n")
        for m in models:
            avg = df_metrics[df_metrics['Model'] == m].mean(numeric_only=True)
            f.write(f"\nModel: {m} ({cv_name})\n")
            for metric in ['Accuracy','Sensitivity','Specificity','PPV','NPV','AUC']:
                f.write(f"  {metric}: {avg[metric]:.3f}\n")

    # Plot ROC averaged
    plt.figure(figsize=(10, 7))
    mean_fpr = np.linspace(0, 1, 200)
    for m, curves in roc_points.items():
        interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in curves]
        interp_tprs = [tpr/ np.max(tpr) if np.max(tpr)>0 else tpr for tpr in interp_tprs]
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = df_metrics[(df_metrics['Model']==m)&(df_metrics['CV']==cv_name)]['AUC'].mean()
        plt.plot(mean_fpr, mean_tpr, lw=2, label=f"{m} (AUC={mean_auc:.2f})")
    plt.plot([0,1],[0,1],'--',color='grey')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'Average ROC Curve ({cv_name})', fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_{cv_name}.png'), dpi=300)
    plt.close()

    # Boxplots: Accuracy & AUC
    for metric in ['Accuracy','AUC']:
        plt.figure(figsize=(10,6))
        sns.boxplot(x='Model', y=metric, data=df_metrics[df_metrics['CV']==cv_name])
        plt.title(f'{metric} Distribution ({cv_name})', fontsize=18)
        plt.xlabel('Model', fontsize=14); plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, fontsize=12); plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{metric.lower()}_{cv_name}.png'), dpi=300)
        plt.close()

    # Average confusion matrix heatmaps
    for m in models:
        avg_cm = np.mean(conf_matrices[m], axis=0)
        plt.figure(figsize=(6,5))
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
        plt.title(f'Average Confusion Matrix ({m}, {cv_name})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'confmat_{m}_{cv_name}.png'), dpi=300)
        plt.close()


# --- INICIO DE NUEVA SECCIÓN: Plots combinados y finales ---

# 8. Crear un DataFrame final con todos los resultados
df_all = pd.DataFrame(all_records)

# 9. Generar Boxplot combinado de Accuracy (CV5 vs CV10)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Model', y='Accuracy', hue='CV', data=df_all)
plt.title('Accuracy Distribution (CV5 vs CV10) — All Models', fontsize=20)
plt.xlabel(None)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(rotation=15, ha='right', fontsize=16)
plt.yticks(fontsize=16)
#plt.ylim(0.90, 1.001) # Ajusta este límite si es necesario para tus datos
plt.legend(title="CV", title_fontsize=18, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_boxplots_all.png'), dpi=300)
plt.close()

# 10. Generar Heatmaps de la Mejor y Peor Matriz de Confusión
model_avg_metrics = df_all.groupby(['Model', 'CV'])['Accuracy'].mean()
best_combo = model_avg_metrics.idxmax()
worst_combo = model_avg_metrics.idxmin()
best_model_name, best_cv_name = best_combo
worst_model_name, worst_cv_name = worst_combo

# Obtener, promediar y normalizar las matrices
avg_cm_best = np.mean(all_conf_mats[best_combo], axis=0)
norm_cm_best = avg_cm_best / avg_cm_best.sum(axis=1, keepdims=True)
avg_cm_worst = np.mean(all_conf_mats[worst_combo], axis=0)
norm_cm_worst = avg_cm_worst / avg_cm_worst.sum(axis=1, keepdims=True)

# Crear la figura con 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
tick_labels = ['Control', 'Long COVID']

# Plot Mejor Matriz
sns.heatmap(norm_cm_best, annot=True, fmt='.3f', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 20})
axes[0].set_title(f'Best CM — {best_model_name} ({best_cv_name})', fontsize=20)
axes[0].set_xlabel('Predicted class', fontsize=20)
axes[0].set_ylabel('True class', fontsize=20)
axes[0].set_xticklabels(tick_labels, fontsize=16)
axes[0].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')

# Plot Peor Matriz
sns.heatmap(norm_cm_worst, annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=False, annot_kws={"size": 20})
axes[1].set_title(f'Worst CM — {worst_model_name} ({worst_cv_name})', fontsize=20)
axes[1].set_xlabel('Predicted class', fontsize=20)
axes[1].set_ylabel('True class', fontsize=20)
axes[1].set_xticklabels(tick_labels, fontsize=16)
axes[1].set_yticklabels(tick_labels, fontsize=16, rotation=90, va='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_best_worst.png'), dpi=300)
plt.close()

# --- FIN DE NUEVA SECCIÓN ---

print("3_2 script execution complete. Outputs in:", OUTPUT_DIR)