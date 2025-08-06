# Long COVID Neural Correlates Identification Using fNIRS Data

**This repository supports the scientific article:**

## Exploring New Horizons: fNIRS and Machine Learning in Understanding postCOVID-19


**Antony Morales-Cervantes** [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--3669--2638-green)](https://orcid.org/0000-0003-3669-2638)  
**Victor Herrera** [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--1367--8622-green)](https://orcid.org/0000-0003-1367-8622)  
**Blanca Nohemí Zamora-Mendoza** [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--0093--7752-green)](https://orcid.org/0000-0003-0093-7752)  
**Rogelio Flores-Ramírez** [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--2263--6280-green)](https://orcid.org/0000-0003-2263-6280)  
**Aaron A. López-Cano**  
**Edgar Guevara** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--2313--2810-green)](https://orcid.org/0000-0002-2313-2810)

---
This repository contains the **expanded version** of the functional Near-Infrared Spectroscopy (fNIRS) dataset and the complete **Python scripts** used for analyzing the neural correlates of postCOVID-19. The dataset consists of **29,737 time-series samples from 37 participants** (9 postCOVID-19, 28 controls). We applied several machine learning algorithms with and without dimensionality reduction to evaluate their performance in classifying long COVID patients.

## Contents

- `Long_COVID_fNIRS_Raw_Data/`: Raw fNIRS recordings (.tsv) from 37 participants (9 post-COVID, 28 controls).
- `data_processing/`: Python scripts implementing subject-aware preprocessing, feature extraction, dimensionality reduction, and classification using six machine learning models.


## Methodology

- **Acquisition**: Brite MKII (Artinis), bilateral motor cortex, finger-tapping task.
- **Preprocessing**: standardization of HbO time-series signals; PCA where applicable.
- **Feature Representation Strategies**:
  1. **Time-Series**: full-resolution HbO signals from 132 features per sample.
  2. **PCA-based**: reduced to 35 components retaining 95% of variance.
  3. **Statistical Features**: 528 subject-level descriptors (mean, std, min, max).
  4. **Hybrid**: time-series samples augmented with per-subject statistics.
- **Validation**: strict **subject-aware cross-validation** using `StratifiedGroupKFold` (5-fold), avoiding data leakage.
- **Metrics**: Accuracy, Sensitivity, Specificity, PPV, NPV, and AUC-ROC.

## Key Findings

- The **hybrid representation** combining time-series and per-subject statistics achieved the most robust classification, with **SVM yielding AUC = 0.91**.
- PCA-based models also showed strong performance, while **statistical-only models were limited** (sensitivity < 0.30).
- Results emphasize the importance of preserving **temporal structure** and preventing **data leakage across subjects**.
- All scripts rely on standard Python libraries: `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `numpy`, `pandas`.


## Purpose

This project aims to promote reproducibility and encourage further research into **non-invasive, portable neuroimaging** tools for the diagnosis and monitoring of long COVID. The results support the development of robust biomarkers using fNIRS and machine learning.

## How to Use

1. **Clone the Repository**
   ```bash
   git clone https://github.com/usuario/longCOVID-fNIRS-dataset.git
   cd longCOVID-fNIRS-dataset
Install Python Dependencies

bash
Copiar
Editar
pip install -r requirements.txt
Run Scripts

1_subject_aware_cv.py: time-series classification using group-stratified cross-validation

2_subject_aware_pca_cv.py: PCA-based dimensionality reduction and classification

3_statistical_features_cv.py: analysis based on statistical features per subject

4_hybrid_features_cv.py: hybrid model combining time-series and statistical features

Explore Results

Performance metrics in results/

Figures: ROC curves, confusion matrices, boxplots

📊 Dataset Summary
Participants: 37 (9 post-COVID, 28 control)



Ethics Statement
The study was approved by the State Research Ethics Committee in Health of San Luis Potosí, Mexico (SLP/08-2020). All participants provided informed consent in accordance with the Declaration of Helsinki.

Authors' Contributions
Edgar Guevara and Aaron López-Cano contributed to data acquisition. Antony Morales-Cervantes led data analysis and software development. All authors participated in the design, interpretation, and manuscript writing. All authors reviewed and approved this version.
