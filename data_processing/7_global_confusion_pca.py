import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Script: 4_2_global_timeseries_pca_cv.py
# ------------------------------

# 1. Paths
DATA_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Database_all"
OUTPUT_DIR = r"G:\Mi unidad\TecNM ITM\Artículo fNIRS\Results\7_global_timeseries_pca_cv" # <-- Directorio de salida actualizado
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load all TSV files
all_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
dfs = [pd.read_csv(fp, sep='\t') for fp in all_files]
global_df = pd.concat(dfs, ignore_index=True)

# 3. Prepare X and y
X = global_df.drop(columns=['time', 'status'], errors='ignore')
y = global_df['status']
n_samples = len(y)

# 4. Standardize and apply PCA (95% variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 5. Define models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1,
                             use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# 6. Cross-validation schemes
cvs = {
    'cv5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'cv10': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
}

# 7. Evaluation loop
# --- INICIO DE MODIFICACIÓN 1: Acumular resultados ---
all_records = []
all_conf_mats = {}
# --- FIN DE MODIFICACIÓN 1 ---

for cv_name, cv in cvs.items():
    records = []
    roc_data = {m: [] for m in models}
    conf_mats = {m: [] for m in models}

    # (El resto del bucle de evaluación se mantiene igual, solo se guardan los resultados)
    fold_dists = []
    for _, test_idx in cv.split(X_pca, y):
        fold_dists.append(y.iloc[test_idx].value_counts().to_dict())

    for m_name, model in models.items():
        fold_num = 1
        for train_idx, test_idx in cv.split(X_pca, y):
            X_tr, X_te = X_pca[train_idx], X_pca[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:,1]
            acc = accuracy_score(y_te, y_pred)
            auc = roc_auc_score(y_te, y_proba)
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
            sens = tp / (tp + fn) if tp + fn else 0
            spec = tn / (tn + fp) if tn + fp else 0
            ppv = tp / (tp + fp) if tp + fp else 0
            npv = tn / (tn + fn) if tn + fn else 0
            records.append({'CV': cv_name, 'Model': m_name, 'Fold': fold_num, 'Accuracy': acc,
                            'Sensitivity': sens, 'Specificity': spec, 'PPV': ppv, 'NPV': npv, 'AUC': auc})
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            roc_data[m_name].append((fpr, tpr))
            conf_mats[m_name].append(np.array([[tn, fp], [fn, tp]]))
            fold_num += 1
        # --- INICIO DE MODIFICACIÓN 2: Guardar matrices de confusión ---
        all_conf_mats[(m_name, cv_name)] = conf_mats[m_name]
        # --- FIN DE MODIFICACIÓN 2 ---

    dfm = pd.DataFrame(records)
    all_records.extend(records) # Acumular registros de cada CV
    
    # Save summary TXT
    with open(os.path.join(OUTPUT_DIR, f'metrics_pca_{cv_name}.txt'), 'w') as f:
        f.write(f"Global samples: {n_samples}\n")
        f.write("Fold distributions (status: count):\n")
        for i, dist in enumerate(fold_dists, 1):
            f.write(f" Fold {i}: {dist}\n")
        f.write("\nAverage metrics per model:\n")
        for m in models:
            avg = dfm[dfm['Model']==m].mean(numeric_only=True)
            f.write(f"{m} ({cv_name}):\n")
            for metric in ['Accuracy','Sensitivity','Specificity','PPV','NPV','AUC']:
                f.write(f"  {metric}: {avg[metric]:.3f}\n")
            f.write("\n")

    # Plot average ROC curves
    plt.figure(figsize=(10, 7))
    mean_fpr = np.linspace(0, 1, 200)
    for m, curves in roc_data.items():
        interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in curves]
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = dfm[(dfm['Model']==m)&(dfm['CV']==cv_name)]['AUC'].mean()
        plt.plot(mean_fpr, mean_tpr, lw=2, label=f"{m} (AUC={mean_auc:.2f})")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'Average ROC with PCA ({cv_name})', fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_pca_{cv_name}.png'), dpi=300)
    plt.close()

    # Boxplots for Accuracy and AUC
    for metric in ['Accuracy','AUC']:
        plt.figure(figsize=(10, 7))
        sns.boxplot(x='Model', y=metric, data=dfm[dfm['CV']==cv_name])
        plt.title(f'{metric} Distribution PCA ({cv_name})', fontsize=18)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{metric.lower()}_pca_{cv_name}.png'), dpi=300)
        plt.close()

    # Average confusion matrix heatmaps
    for m in models:
        avg_cm = np.mean(conf_mats[m], axis=0)
        plt.figure(figsize=(5, 4))
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
        plt.title(f'Avg Confusion Matrix\n{m} ({cv_name})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'confmat_pca_{m}_{cv_name}.png'), dpi=300)
        plt.close()

# --- INICIO DE NUEVA SECCIÓN: Plots combinados y finales ---

# 8. Crear un DataFrame final con todos los resultados
df_all = pd.DataFrame(all_records)

# 9. Generar Boxplot combinado de Accuracy (CV5 vs CV10)
plt.figure(figsize=(12, 7)) # <-- MODIFICADO: Ajusta el tamaño para que sea más ancha que alta
sns.boxplot(x='Model', y='Accuracy', hue='CV', data=df_all)
plt.title('Accuracy Distribution (CV5 vs CV10) — All Models', fontsize=20)
plt.xlabel(None)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(rotation=15, ha='right', fontsize=16) # <-- MODIFICADO: Rota las etiquetas del eje X
plt.yticks(fontsize=16)
#plt.ylim(0.97, 1.001) # <-- AÑADIDO: Establece el límite del eje Y para "hacer zoom"
plt.legend(title="CV", title_fontsize=18, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_boxplots_pca_all.png'), dpi=300)
plt.close()

# 10. Generar Heatmaps de la Mejor y Peor Matriz de Confusión
# Calcular promedios para encontrar el mejor y peor modelo
model_avg_metrics = df_all.groupby(['Model', 'CV'])['Accuracy'].mean()

# Identificar mejor y peor
best_combo = model_avg_metrics.idxmax()
worst_combo = model_avg_metrics.idxmin()

best_model_name, best_cv_name = best_combo
worst_model_name, worst_cv_name = worst_combo

# Obtener, promediar y normalizar las matrices de confusión
# Mejor
avg_cm_best = np.mean(all_conf_mats[best_combo], axis=0)
norm_cm_best = avg_cm_best / avg_cm_best.sum(axis=1, keepdims=True)
# Peor
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
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_best_worst_pca.png'), dpi=300)
plt.close()

# --- FIN DE NUEVA SECCIÓN ---

print("4_2 script complete. Results in:", OUTPUT_DIR)