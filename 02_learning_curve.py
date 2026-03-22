"""
02_learning_curve.py
Learning curve analysis using the best Random Forest configuration.
"""
import warnings
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_CSV   = "A5.csv"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


FIG_WIDTH, FIG_HEIGHT = 3.46, 3.46
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def load_data(path):
    for enc in ["utf-8-sig", "cp932"]:
        try:
            df = pd.read_csv(path, encoding=enc, index_col=0)
            df = df.select_dtypes(include=[np.number]).dropna()
            return df.iloc[:, :-1], df.iloc[:, -1]
        except Exception:
            continue
    raise ValueError(f"Could not read {path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("02 LEARNING CURVE")
    print("="*60)


    X, y = load_data(TRAIN_CSV)
    print(f"Data: X={X.shape}, features={list(X.columns)}")


    pipe = Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ))
    ])


    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator=pipe,
        X=X, y=y,
        train_sizes=np.linspace(0.14, 1.0, 8),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores,  axis=1, ddof=1)
    test_scores_mean  = np.mean(test_scores,  axis=1)
    test_scores_std   = np.std(test_scores,   axis=1, ddof=1)


    pd.DataFrame({
        "train_size":       train_sizes_abs,
        "train_score_mean": train_scores_mean,
        "train_score_std":  train_scores_std,
        "cv_score_mean":    test_scores_mean,
        "cv_score_std":     test_scores_std,
    }).to_csv(RESULTS_DIR / "SourceData_Fig_learning_curve.csv", index=False)


    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(train_sizes_abs, train_scores_mean, "o-",
            color="#1f78b4", lw=1.2, ms=4, label="Training score")
    ax.fill_between(train_sizes_abs,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.2, color="#1f78b4")
    ax.plot(train_sizes_abs, test_scores_mean, "s-",
            color="#e31a1c", lw=1.2, ms=4, label="CV score")
    ax.fill_between(train_sizes_abs,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.2, color="#e31a1c")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("$R^2$")
    ax.set_xlim(5, 75)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Fig_learning_curve.pdf")
    plt.close()
    print("  ✓ Fig_learning_curve.pdf")
