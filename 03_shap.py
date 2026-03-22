"""
03_shap.py
SHAP analysis on the final trained model.
Computes mean |SHAP| feature importance and beeswarm plot.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import shap
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_CSV   = "A5.csv"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH  = RESULTS_DIR / "model_artifacts" / "final_best_pipe_Random Forest.joblib"

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
    print("03 SHAP ANALYSIS")
    print("="*60)

    X, y = load_data(TRAIN_CSV)
    print(f"Data: X={X.shape}, features={list(X.columns)}")

    pipe  = joblib.load(MODEL_PATH)
    model = pipe.named_steps["model"]
    print(f"Model loaded: {type(model).__name__}")

    # Transform if scaler exists
    if "scaler" in pipe.named_steps:
        scaler = pipe.named_steps["scaler"]
        X_in = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        print("Preprocessing: RobustScaler applied")
    else:
        X_in = X.copy()
        print("Preprocessing: None")

    # SHAP values
    explainer   = shap.Explainer(model, X_in)
    shap_values = explainer(X_in)
    shap_vals   = shap_values.values
    feat_names  = X.columns.tolist()
    print(f"SHAP computed for {len(X)} samples")

    # Importance
    imp = pd.DataFrame({
        "feature":       feat_names,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(RESULTS_DIR / "SourceData_Fig5a_SHAP_importance.csv", index=False)

    # Raw SHAP values
    shap_df = pd.DataFrame(shap_vals, columns=feat_names, index=X.index)
    shap_df.insert(0, "Sample_ID", X.index)
    shap_df.to_csv(RESULTS_DIR / "SourceData_Fig5b_SHAP_values.csv", index=False)

    # Fig 5a: bar
    imp_s = imp.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.barh(np.arange(len(imp_s)), imp_s["mean_abs_shap"],
            color="#1f78b4", edgecolor="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(imp_s)))
    ax.set_yticklabels(imp_s["feature"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Fig5a_SHAP_importance_bar.pdf")
    plt.close()
    print("  ✓ Fig5a_SHAP_importance_bar.pdf")

    # Fig 5b: beeswarm
    plt.figure(figsize=(8, 10))
    shap.summary_plot(shap_values, features=X_in, feature_names=feat_names,
                      max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Fig5b_SHAP_beeswarm.pdf")
    plt.close()
    print("  ✓ Fig5b_SHAP_beeswarm.pdf")

    print("\nFeature Importance:")
    print(imp.to_string(index=False))
