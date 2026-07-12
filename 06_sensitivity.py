"""
Descriptor-Exclusion Sensitivity Analysis using Repeated Nested Cross-Validation
Author: Kei Masuhara
Date: July 2026

This script explicitly quantifies physical descriptor redundancy and validates 
model interpretability profiles under remaining multicollinearity (high VIF) 
using a mathematically rigorous repeated nested cross-validation framework.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import r2_score

# =============================================================================
# 1. CONFIGURATION AND CONFIG BOUNDARIES
# =============================================================================
DATA_PATH = "A5.csv"
OUTPUT_DIR = Path(".")
OUTPUT_FILE = "Sensitivity_Analysis_NestedCV.csv"

# Hyperparameter optimization grid synchronized with primary model architecture
RF_MODEL = RandomForestRegressor(random_state=42, n_estimators=500)
PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "min_samples_leaf": [3, 5],
    "max_features": ["sqrt", None]
}

# 5-fold, 10-repeats structure (Total 50 outer folds for strict validation)
RKF_OUTER = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# Definition of systematic physical feature-exclusion alternative subsets
FEATURE_SETS = {
    "Full Model": ['FT', 'SQ', 'CS', 'FS'],
    "Without FT": ['SQ', 'CS', 'FS'],
    "Without SQ": ['FT', 'CS', 'FS'],
    "Without CS": ['FT', 'SQ', 'FS'],
    "Without FS": ['FT', 'SQ', 'CS']
}

# =============================================================================
# 2. DATA LOADING AND INITIALIZATION
# =============================================================================
def load_and_verify_dataset(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Target verification dataset '{path}' not found.")
    df = pd.read_csv(path)
    X_full = df[['FT', 'SQ', 'CS', 'FS']]
    y = df['Epit']
    return X_full, y

# =============================================================================
# 3. REPEATED NESTED CROSS-VALIDATION LOOP
# =============================================================================
def execute_sensitivity_nested_cv(X_full, y):
    records = []
    
    for model_name, feature_list in FEATURE_SETS.items():
        print(f"Executing Repeated Nested CV space for: {model_name}")
        X_sub = X_full[feature_list].values
        outer_r2_scores = []
        
        # Outer Loop: Generalization capability evaluation
        for train_idx, test_idx in RKF_OUTER.split(X_sub, y):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Inner Loop: Rigorous hyperparameter re-optimization per fold
            inner_cv = GridSearchCV(
                estimator=RF_MODEL,
                param_grid=PARAM_GRID,
                cv=5, 
                scoring='r2',
                n_jobs=-1
            )
            inner_cv.fit(X_train, y_train)
            
            # Prediction validation using optimized parameters
            best_estimator = inner_cv.best_estimator_
            y_pred = best_estimator.predict(X_test)
            outer_r2_scores.append(r2_score(y_test, y_pred))
            
        mean_r2 = np.mean(outer_r2_scores)
        std_r2 = np.std(outer_r2_scores)
        print(f"Result -> {model_name}: Mean R² = {mean_r2:.4f} ± {std_r2:.4f}\n")
        
        records.append({
            "Model": model_name,
            "Features": ", ".join(feature_list),
            "CV_R2_Mean": mean_r2,
            "CV_R2_Std": std_r2
        })
        
    return pd.DataFrame(records)

# =============================================================================
# 4. EXECUTION AND DATA EXPORT
# =============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("REPEATED NESTED CROSS-VALIDATION SENSITIVITY ANALYSIS ROUTINE")
    print("=" * 75)
    
    X_full, y = load_and_verify_dataset(DATA_PATH)
    sensitivity_results = execute_sensitivity_nested_cv(X_full, y)
    
    # Save the finalized statistical benchmark to disk
    output_path = OUTPUT_DIR / OUTPUT_FILE
    sensitivity_results.to_csv(output_path, index=False)
    
    print("=" * 75)
    print(f"Analysis successfully completed. Statistics saved to: '{output_path}'")
    print("=" * 75)
