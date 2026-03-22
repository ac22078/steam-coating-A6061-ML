# steam-coating-A6061-ML
Code and data for interpretable ML analysis of steam-coated A6061 pitting resistance

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18876390.svg)](https://doi.org/10.5281/zenodo.18876390)

## Analysis scripts

Run the scripts in order. Each script saves outputs to `./results/`.

- `01_nested_cv.py`  
  Repeated nested cross-validation (5-fold outer, 10 repeats) across five regression models (GPR, Random Forest, Gradient Boosting, SVR, Elastic Net). Generates out-of-fold parity plot (Fig. 4) and retrains the best model on the full dataset.

- `02_learning_curve.py`  
  Learning curve analysis using the best Random Forest configuration selected in step 01.

- `03_shap.py`  
  SHAP analysis on the final trained model. Computes mean |SHAP| feature importance (Fig. 5a) and beeswarm plot (Fig. 5b).

- `04_ale.py`  
  1D-ALE with bootstrap 95% CI (Fig. 6) and 2D-ALE interaction strength analysis with bootstrap CI (Fig. 7, 8). Requires the model saved by `01_nested_cv.py`.

- `05_monte_carlo.py`  
  Monte Carlo uncertainty propagation for predicted pitting potential. Applies relative measurement errors to each input feature across 1000 trials and reports empirical prediction uncertainty.

## Usage

```bash
python 01_nested_cv.py
python 02_learning_curve.py
python 03_shap.py
python 04_ale.py
python 05_monte_carlo.py
```

All outputs are saved to `./results/`.

## Input data

- `A4.csv`  
  Preliminary experimental dataset used for exploratory model development. It contains numerical predictor variables describing material and electrochemical conditions, with the target variable (e.g., pitting potential) stored in the last column.

- `A5.csv`  
  Main experimental dataset used for the final machine learning analysis and figures reported in the manuscript. All predictor variables are stored in the first columns and the target variable in the last column; this file is used as the default training data in all scripts.

All scripts assume that non-numeric columns have been removed and that missing values are either absent or handled during preprocessing.

## Requirements

- Python 3.9 or later  
- numpy, pandas, matplotlib, scikit-learn, joblib  
- shap, PyALE  

```bash
pip install numpy pandas matplotlib scikit-learn joblib shap PyALE
```

## Citation

If you use this code or data, please cite:

Kei Masuhara, et al. (2026). steam-coating-A6061-ML: Code and data for interpretable ML analysis of steam-coated A6061 pitting resistance (Version 1.0.1). Zenodo. [https://doi.org/10.5281/zenodo.18876390](https://doi.org/10.5281/zenodo.18876390)
