# steam-coating-A6061-ML
Code and data for interpretable ML analysis of steam-coated A6061 pitting resistance
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18876390.svg)](https://doi.org/10.5281/zenodo.18876390)

## Analysis scripts

This repository contains Python scripts used for the machine learning analysis and uncertainty quantification in the associated corrosion study.

- `full_analysis_pipeline.py`  
  Nested cross-validation of multiple regression models (GPR, Random Forest, Gradient Boosting, SVR, Elastic Net), out-of-fold prediction analysis, learning curve calculation, and model retraining on the full dataset. It also performs 1D/2D ALE, bootstrap-based interaction strength estimation, and SHAP analysis for model interpretability.

- `mc_error_propagation.py`  
  Monte Carlo simulation for propagating experimental uncertainties in electrochemical measurements (e.g., pitting potential) to estimated model inputs and outputs, providing empirical confidence intervals for the predicted quantities.

### Input data

- `A4.csv`  
  Preliminary experimental dataset used for exploratory model development. It contains numerical predictor variables describing material and electrochemical conditions, with the target variable (e.g., pitting potential) stored in the last column.

- `A5.csv`  
  Main experimental dataset used for the final machine learning analysis and figures reported in the manuscript. As with A4, all predictor variables are stored in the first columns and the target variable in the last column; this file is used as the default training data in `full_analysis_pipeline.py`.

All scripts assume that non-numeric columns have been removed and that missing values are either absent or handled during preprocessing.

### Requirements

- Python 3.9 or later  
- numpy, pandas, matplotlib, scikit-learn, joblib  
- shap, PyALE  

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn joblib shap PyALE

### Citation

If you use this code or data, please cite:

Kei Masuhara, et al. (2026). steam-coating-A6061-ML: Code and data for interpretable ML analysis of steam-coated A6061 pitting resistance (Version 1.0.1). Zenodo. https://doi.org/10.5281/zenodo.18876390
