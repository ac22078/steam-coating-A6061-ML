"""
Monte Carlo uncertainty propagation for E_pit predictions.

- Uses the final trained Random Forest pipeline saved as:
    final_analysis_results/model_artifacts/final_best_pipe_Random Forest.joblib
- Input data:
    A5.csv  (FT, SQ, CS, FS, Epit)
- Output:
    final_analysis_results/mc_predictions.csv
    final_analysis_results/mc_summary.txt
    final_analysis_results/mc_pred_with_uncertainty.pdf
    final_analysis_results/mc_std_hist.pdf
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MODEL_PATH = Path(
    "final_analysis_results/model_artifacts/"
    "final_best_pipe_Random Forest.joblib"
)
DATA_CSV = "A5.csv"  # main analysis dataset (4 descriptors + Epit)
RESULTS_DIR = Path("final_analysis_results")
RESULTS_DIR.mkdir(exist_ok=True)

N_MC = 1000  # number of Monte Carlo trials

# Relative measurement errors
ERR_FT = 0.0852  # 8.52 %
ERR_SQ = 0.125   # 12.5 %
ERR_CS = 0.05    # 5 %
ERR_FS = 0.01    # 1 %


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data(file_path: str):
    """Load numeric data and split into X (features) and y (target)."""
    for enc in ["utf-8-sig", "cp932"]:
        try:
            df = pd.read_csv(file_path, encoding=enc, index_col=0)
            df = df.select_dtypes(include=[np.number]).dropna()
            return df.iloc[:, :-1], df.iloc[:, -1]
        except Exception:
            continue
    raise ValueError("CSV reading failed.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Load data
    X, y = load_data(DATA_CSV)
    print("Columns:", list(X.columns))

    # Load trained pipeline
    pipe = load(MODEL_PATH)

    X_base = X.copy()
    n_samples = len(X_base)

    # Monte Carlo predictions
    pred_mc = np.zeros((n_samples, N_MC))

    for k in range(N_MC):
        X_perturbed = X_base.copy()
        X_perturbed["FT"] = X_base["FT"] * (1 + ERR_FT * np.random.randn(n_samples))
        X_perturbed["SQ"] = X_base["SQ"] * (1 + ERR_SQ * np.random.randn(n_samples))
        X_perturbed["CS"] = X_base["CS"] * (1 + ERR_CS * np.random.randn(n_samples))
        X_perturbed["FS"] = X_base["FS"] * (1 + ERR_FS * np.random.randn(n_samples))

        pred_mc[:, k] = pipe.predict(X_perturbed)

    # Statistics for each sample
    pred_mean = pred_mc.mean(axis=1)
    pred_std = pred_mc.std(axis=1, ddof=1)

    # Overall RMSE due to measurement error only
    rmse_mc = float(np.sqrt(np.mean((pred_mc - pred_mean[:, None]) ** 2)))
    mean_std = float(pred_std.mean())
    max_std = float(pred_std.max())

    # Save detailed predictions
    out_df = pd.DataFrame(
        {
            "Epit_true": y.values,
            "Epit_pred_base": pipe.predict(X_base),
            "Epit_pred_mc_mean": pred_mean,
            "Epit_pred_mc_std": pred_std,
        }
    )
    out_df.to_csv(RESULTS_DIR / "mc_predictions.csv", index=False)

    # Save text summary
    with open(RESULTS_DIR / "mc_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Monte Carlo trials: {N_MC}\n")
        f.write(f"Mean MC std over samples: {mean_std:.4f} V\n")
        f.write(f"Max  MC std over samples: {max_std:.4f} V\n")
        f.write(
            f"RMSE due to measurement error only: {rmse_mc:.4f} V\n"
        )

    print("Monte Carlo summary:")
    print(f"  Mean std = {mean_std:.4f} V")
    print(f"  Max  std = {max_std:.4f} V")
    print(f"  RMSE(measurement error only) = {rmse_mc:.4f} V")

    # Plot: per-sample uncertainty
    x_idx = np.arange(n_samples)

    plt.figure(figsize=(3.46, 3.46))
    plt.errorbar(
        x_idx,
        pred_mean,
        yerr=pred_std,
        fmt="o",
        ms=3,
        ecolor="#fb9a99",
        color="#1f78b4",
        elinewidth=0.5,
        capsize=2,
    )
    plt.xlabel("Sample index")
    plt.ylabel("Predicted $E_{pit}$ / V")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mc_pred_with_uncertainty.pdf")
    plt.close()

    # Plot: histogram of MC std
    plt.figure(figsize=(3.46, 3.46))
    plt.hist(
        pred_std,
        bins=15,
        color="#e31a1c",
        edgecolor="k",
        alpha=0.7,
    )
    plt.xlabel("MC std of predicted $E_{pit}$ / V")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mc_std_hist.pdf")
    plt.close()


if __name__ == "__main__":
    main()
