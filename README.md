# Long COVID Neural Correlates Identification Using fNIRS Data

**This repository supports the scientific article:**

##Exploring New Horizons: fNIRS and Machine Learning in Understanding postCOVID-19*


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
- `data_processing/`: Python scripts for preprocessing, feature extraction, dimensionality reduction, and classification using six ML models.



## Methodology

- Acquisition: Brite MKII (Artinis), bilateral motor cortex, finger-tapping task.
- Preprocessing: time-series standardization, PCA (95% variance retained).
- Machine Learning:
  - **Time-series models**: direct use of 132-channel HbO features (per sample).
  - **PCA-based models**: reduced to 35 components.
  - **Statistical features**: 528 features per subject (mean, std, min, max), for comparison.
- Evaluation: 5-fold and 10-fold **stratified cross-validation** with metrics including:
  - Accuracy, Sensitivity, Specificity, PPV, NPV, and AUC-ROC.

## Key Findings

- Time-series models using **PCA** retained nearly perfect classification (up to 100% accuracy for KNN, MLP, RF, XGBoost).
- **Flattened statistical features** resulted in significantly lower sensitivity (<30%), highlighting the critical role of temporal dynamics.
- All scripts use standard Python libraries: `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `numpy`, `pandas`.

## Purpose

This project aims to promote reproducibility and encourage further research into **non-invasive, portable neuroimaging** tools for the diagnosis and monitoring of long COVID. The results support the development of robust biomarkers using fNIRS and machine learning.

## ⚙How to Use

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

6_global_timeseries_cv.py: full time-series classification

7_global_confusion_pca.py: PCA-reduced classification

8_statistical_features_cv.py: statistical feature analysis

Explore Results

Performance metrics in results/

Figures: ROC curves, confusion matrices, boxplots

📊 Dataset Summary
Participants: 37 (9 post-COVID, 28 control)



Ethics Statement
The study was approved by the State Research Ethics Committee in Health of San Luis Potosí, Mexico (SLP/08-2020). All participants provided informed consent in accordance with the Declaration of Helsinki.

Authors' Contributions
Edgar Guevara and Aaron López-Cano contributed to data acquisition. Antony Morales-Cervantes led data analysis and software development. All authors participated in the design, interpretation, and manuscript writing. All authors reviewed and approved this version.
