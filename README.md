# Machine Learning-Based Prediction of Nanoparticle-Induced Hepatotoxicity and Nephrotoxicity in Mice

**Date:** February 2026  
**Authors:** Qiran Chen, Kun Mi, Xinyue Chen, Zhoumeng Lin  
**Institution:** University of Florida  
**Contact:** linzhoumeng@ufl.edu

## 📋 Project Description
This repository contains comprehensive machine learning models for predicting nanoparticle-induced hepatotoxicity and nephrotoxicity in mice. The models were developed using a curated Nano-Tox Database comprising 2,144 datasets from 214 different nanoparticle-experimental setting combinations across 104 publications (2021-2024). The models predict changes in four key biomarkers:

- **Hepatic biomarkers:** Alanine aminotransferase (ALT) and Aspartate aminotransferase (AST)
- **Renal biomarkers:** Blood urea nitrogen (BUN) and Creatinine (CREA)

The project provides computational tools to guide nanoparticle design by predicting potential liver and kidney toxicity, thereby supporting safer clinical translation of nanomedicine.

## 📁 Repository Contents

### Best Model Implementation Notebooks
#### Each notebook contains the optimized pipeline for predicting toxicity of specific biomarkers:

- `ALT_best_model.ipynb` - Best performing model for ALT prediction (CatBoost)
- `AST_best_model.ipynb` - Best performing model for AST prediction (Random Forest)
- `BUN_best_model.ipynb` - Best performing model for BUN prediction (CatBoost)
- `CREA_best_model.ipynb` - Best performing model for CREA prediction (LightGBM)

#### Why these are the "best" models:

- Selected from 9 different algorithms (Random Forest, LightGBM, XGBoost, CatBoost, SVR, KNN, Bayesian Ridge, Gaussian Process Regression, DNN)
- Optimized through extensive hyperparameter tuning with 5-fold cross-validation
- Outperformed all other models in rigorous testing with independent validation sets
- Include ensemble approaches (DNN-LGBM hybrid and stacked ensembles) that showed superior performance
- Feature comprehensive SHAP analysis and interpretability components


### Processed Data Files for Each Biomarker
- `SMOGN dataset_ALT.csv`
- `SMOGN dataset_AST.csv`
- `SMOGN dataset_BUN.csv`
- `SMOGN dataset_CREA.csv`

## 🔧 Core Components in Each Notebook

### Data Loading and Cleaning
- Import nanoparticle physicochemical properties and biomarker data
- Handle missing values
- Standardize categorical variables (e.g., tumor model standardization)

### Feature Engineering
- Calculate biomarker percentage changes: Biomarker% = (C_DLNP - C_CTRL)/C_CTRL × 100%
- Compute cumulative dose: AccumDose = Dose × Dose frequency
- Transform features: log10(HD) for hydrodynamic size, Z-score for zeta potential
- One-hot encoding for categorical variables (9 categorical features → 32 binary features)

### Data Augmentation (SMOGN)
- Apply Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise (SMOGN)
- Addresses imbalance between non-toxic and toxic responses
- Generates synthetic samples for underrepresented toxicity ranges

### Machine Learning Model Development
- Individual models: Random Forest, LightGBM, XGBoost, CatBoost, SVR, KNN, Bayesian Ridge, Gaussian Process Regression, Deep Neural Network
- Ensemble models: DNN-LGBM hybrid and stacked ensemble models
- Hyperparameter optimization via 5-fold cross-validation

### Model Interpretation
- SHAP (SHapley Additive exPlanations) analysis for feature importance
- Dependence plots for top physicochemical features
- Statistical significance testing with Bonferroni-adjusted Dunn's post hoc tests

## ⚙️ Prerequisites

### 🐍 Python Requirements
```
# Core packages
pip install numpy pandas scikit-learn matplotlib seaborn
pip install scikit-optimize keras-tuner iterative-stratification
pip install pyreadr lifelines smogn scikit_posthocs
pip install tensorflow==2.18.0

# Machine learning libraries
pip install xgboost lightgbm catboost
pip install shap  # For model interpretation
```

### 📦 Required Python Packages
```python
# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning models
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep learning
import tensorflow as tf
from tensorflow import keras

# Model interpretation
import shap

# Data augmentation
import smogn
```

### 💻 System Requirements
- Python 3.8+
- Minimum 8GB RAM (16GB recommended for DNN training)
- GPU support recommended for deep learning models

## 🚀 Usage

### ▶️ Running Individual Biomarker Models
Each .ipynb file is self-contained and runs sequentially:

```python
# 1. Install required packages (if not already installed)
# 2. Mount Google Drive (if using Colab) or set local directory
# 3. Run all cells sequentially

# The notebook will:
# - Load and preprocess data
# - Perform feature engineering
# - Apply SMOGN data augmentation
# - Train and evaluate multiple ML models
# - Generate SHAP analysis and feature importance plots
# - Save best models and results
```

### 🔄 Model Training Workflow Using the Best Model for ALT as an Example
1. **Data Preparation** (ALT_best_model.ipynb cells 1-35)
   - Load nanoparticle info and biomarker data
   - Merge datasets and handle missing values
   - Calculate biomarker percentage changes

2. **Feature Transformation** (cells 36-50)
   - Apply log10 transformation to hydrodynamic size
   - Standardize zeta potential and cumulative dose
   - One-hot encode categorical variables

3. **Data Augmentation** (cells 51-70)
   - Apply SMOGN algorithm to balance dataset
   - Generate synthetic samples for extreme toxicity values

4. **Model Development** (cells 71-100)
   - Split data into training (80%) and test (20%) sets
   - Train 9 individual ML models
   - Optimize hyperparameters via 5-fold cross-validation
   - Build ensemble models (DNN-LGBM hybrid and stacked ensemble)

5. **Model Evaluation** (cells 101-120)
   - Calculate R² and RMSE for all models
   - Compare performance across algorithms
   - Identify best-performing model for each biomarker

6. **Model Interpretation** (cells 121-150)
   - Perform SHAP analysis on best models
   - Generate feature importance rankings
   - Create dependence plots for top features
   - Statistical analysis of material-specific effects

### 🔍 Accessing Results
After running any notebook, results are available in:

```python
# Model performance metrics
performance_df = pd.DataFrame({
    'Model': model_names,
    'R2_train': train_scores,
    'R2_test': test_scores,
    'RMSE_train': train_rmse,
    'RMSE_test': test_rmse
})

# Best model (saved automatically)
best_model = joblib.load('best_model_ALT.pkl')

# SHAP values
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
```

## 📈 Model Performance

### 🏆 Best Performing Models
| Biomarker | Best Model | Test R² | Test RMSE | Key Features         |
|----------|------------|---------|-----------|----------------------|
| ALT      | CatBoost   | 0.72    | 23.57     | ZP, logHD, TS        |
| AST      | CatBoost   | 0.79    | 23.29     | ZP, logHD, AccumDose |
| BUN      | CatBoost   | 0.96    | 11.04     | AccumDose, ZP, logHD |
| CREA     | LightGBM   | 0.64    | 24.21     | AccumDose, logHD, ZP |

### 📊 Ensemble Model Performance
- DNN-LGBM Hybrid: R² = 0.62-0.96 across biomarkers
- Stacked Ensemble: Test R² = 0.67-0.97 across biomarkers
- Consistent outperformance of individual models by ensembles

## 🔑 Key Features

### Input Features (19 total, expanded to 32 via encoding)
**1. Physicochemical Properties**
- Particle Type (PT): Inorganic, Organic, Hybrid
- Core Material (MAT): 10 categories (Polymeric, Gold, Liposome, etc.)
- Shape: Spherical, Rod, Plate
- Targeting Strategy (TS): Active, Passive
- Hydrodynamic Size (logHD): log10-transformed
- Zeta Potential (ZP): Standardized

**2. Experimental Conditions**
- Cumulative Dose (AccumDose): Dose × Frequency
- Tumor Model (TM): 4 categories
- Cancer Type: 19 categories
- Assistive Technology (AT): Yes/No
- Mouse Model (MM): TB, healthy
- Co-delivery: Yes/No

**3. Derived Features**
- Biomarker%: Percentage change from control
- Maximum absolute change across time points

### 🔧 Feature Engineering Details
- **Log Transformation:** logHD = log10(HD) reduces skewness in hydrodynamic size
- **Standardization:** ZP_std = (ZP - μ_ZP) / σ_ZP enables comparison across features
- **One-Hot Encoding:** Converts 9 categorical variables to 32 binary features
- **SMOGN Augmentation:** Balances dataset with Gaussian noise and SMOTE techniques

## 📊 Data Sources

### 📚 Experimental Data Collection
- Time Period: 2021-2024 publications
- Number of Studies: 104 publications
- Total Datasets: 2,144 across 214 NP-experimental setting combinations
- Biomarker Measurements: 731 DLNP, 726 control, 514 free drug, 173 other treatments

### 🔍 Data Categories
**1. Nanoparticle Groups:**
- Drug-Loaded Nanoparticles (DLNP)
- Control (PBS/saline)
- Free Drug
- Other (blank NPs, assisted free drug, healthy controls)

**2. Biomarker Availability:**
- ALT: 200 DLNP, 201 control, 139 free drug, 50 other
- AST: 197 DLNP, 193 control, 141 free drug, 45 other
- BUN: 169 DLNP, 168 control, 115 free drug, 39 other
- CREA: 165 DLNP, 164 control, 119 free drug, 39 other

## ⚙️Model Details

### 🤖 Machine Learning Algorithms
**1. Tree-Based Models**
- Random Forest: 100-500 trees, max depth 10-30
- XGBoost: Learning rate 0.01-0.3, max depth 3-10
- LightGBM: Num leaves 31-127, learning rate 0.01-0.1
- CatBoost: Iterations 100-1000, depth 4-10

**2. Other Algorithms**
- Support Vector Regressor (SVR): RBF kernel, C=1-100
- K-Nearest Neighbors (KNN): k=3-15, distance weighting
- Bayesian Ridge Regression: Alpha_1=1e-6, lambda_1=1e-6
- Gaussian Process Regression: RBF kernel

**3. Deep Learning**
- DNN Architecture: 3-5 hidden layers (64-256 neurons each)
- Activation: ReLU, Output: Linear
- Optimizer: Adam, Learning Rate: 0.001-0.01
- Regularization: Dropout (0.2-0.5), L2 (1e-4)

### 🎯 Hyperparameter Optimization
- Method: 5-fold cross-validation with Bayesian optimization
- Search Space: Pre-defined ranges for each algorithm
- Objective: Maximize R² on validation set
- Iterations: 50-100 iterations per model

### 📈 Performance Metrics
- Primary: Coefficient of Determination (R²)
- Secondary: Root Mean Square Error (RMSE)
- Validation: 5-fold cross-validation consistency
- Testing: Independent test set (20% of data)

## 📝 Citation
If you use these models in your research, please cite:

Chen, Q., Mi, K., Chen, X., & Lin, Z. (2026). Machine Learning-Based Prediction of Nanoparticle-Induced Hepatotoxicity and Nephrotoxicity in Mice. (Manuscript submitted for publication)

## 📞 Contact
**Authors:** Qiran Chen, Kun Mi, Xinyue Chen, Zhoumeng Lin  
**Email:** linzhoumeng@ufl.edu  
**Institution:** University of Florida  
- Department of Environmental and Global Health, College of Public Health and Health Professions  
- Center for Environmental and Human Toxicology  
- Center for Pharmacometrics and Systems Pharmacology

## 🙏 Acknowledgments
- **Funding:** National Institute of Biomedical Imaging and Bioengineering of the National Institutes of Health (Grant Number: R01EB031022)
- **Data Contributors:** Research groups of the 104 publications whose published data were included in the Nano-Tox Database
- **Technical Support:** University of Florida Research Computing

**⚠️Note:** Each .ipynb file is designed to run independently. When running locally, adjust the file paths in the data loading sections (cells 3-5) to match your directory structure. The notebooks were originally developed in Google Colab with Google Drive mounting for data access.
