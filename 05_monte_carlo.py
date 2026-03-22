"""
05_monte_carlo.py
Monte Carlo uncertainty propagation for E_pit predictions.

- Uses the final trained Random Forest pipeline saved by 01_nested_cv.py
- Input data: A5.csv (FT, SQ, CS, FS, Epit)
- Output:
    results/SourceData_SuppFig2_MonteCarlo.csv
    results/mc_summary.txt
    results/mc_pred_with_uncertainty.pdf
    results/mc_std_hist.pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_CSV    = "A5.csv"
RESULTS_DIR = Path("./results")
MODEL_PATH  = RESULTS_DIR / "model_artifacts" / "final_best_pipe_Random Forest.joblib"
RESULTS_DIR.mkdir(exist_ok=True)

N_MC   = 1000
ERR_FT = 0.0852
ERR_SQ = 0.125
ERR_CS = 0.05
ERR_FS = 0.01

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def load_data(file_path: str):
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
    X, y = load_data(DATA_CSV)
    print(f"Data: X={X.shape}, features={list(X.columns)}")

    pipe      = load(MODEL_PATH)
    n_samples = len(X)
    pred_mc   = np.zeros((n_samples, N_MC))

    print(f"Running {N_MC} Monte Carlo iterations...")
    for k in range(N_MC):
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{N_MC}...", flush=True)
        X_perturbed = X.copy()
        X_perturbed["FT"] = X["FT"] * (1 + ERR_FT * np.random.randn(n_samples))
        X_perturbed["SQ"] = X["SQ"] * (1 + ERR_SQ * np.random.randn(n_samples))
        X_perturbed["CS"] = X["CS"] * (1 + ERR_CS * np.random.randn(n_samples))
        X_perturbed["FS"] = X["FS"] * (1 + ERR_FS * np.random.randn(n_samples))
        pred_mc[:, k] = pipe.predict(X_perturbed)

    pred_mean = pred_mc.mean(axis=1)
    pred_std  = pred_mc.std(axis=1, ddof=1)
    rmse_mc   = float(np.sqrt(np.mean((pred_mc - pred_mean[:, None]) ** 2)))
    mean_std  = float(pred_std.mean())
    max_std   = float(pred_std.max())

    # CSV
    pd.DataFrame({
        "Epit_true":         y.values,
        "Epit_pred_base":    pipe.predict(X),
        "Epit_pred_mc_mean": pred_mean,
        "Epit_pred_mc_std":  pred_std,
    }).to_csv(RESULTS_DIR / "SourceData_SuppFig2_MonteCarlo.csv", index=False)
    print("  ✓ SourceData_SuppFig2_MonteCarlo.csv")

    # Summary
    with open(RESULTS_DIR / "mc_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Monte Carlo trials: {N_MC}\n")
        f.write(f"Mean MC std over samples: {mean_std:.4f} V\n")
        f.write(f"Max  MC std over samples: {max_std:.4f} V\n")
        f.write(f"RMSE due to measurement error only: {rmse_mc:.4f} V\n")
    print(f"  Mean std = {mean_std:.4f} V  |  Max std = {max_std:.4f} V")

    # Plot: per-sample uncertainty
    plt.figure(figsize=(3.46, 3.46))
    plt.errorbar(np.arange(n_samples), pred_mean, yerr=pred_std,
                 fmt="o", ms=3, ecolor="#fb9a99", color="#1f78b4",
                 elinewidth=0.5, capsize=2)
    plt.xlabel("Sample index")
    plt.ylabel(r"Predicted $E_\mathrm{pit}$ / V")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mc_pred_with_uncertainty.pdf")
    plt.close()
    print("  ✓ mc_pred_with_uncertainty.pdf")

    # Plot: std histogram
    plt.figure(figsize=(3.46, 3.46))
    plt.hist(pred_std, bins=15, color="#e31a1c", edgecolor="k", alpha=0.7)
    plt.xlabel(r"MC std of predicted $E_\mathrm{pit}$ / V")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mc_std_hist.pdf")
    plt.close()
    print("  ✓ mc_std_hist.pdf")


if __name__ == "__main__":
    print("=" * 60)
    print("05 MONTE CARLO UNCERTAINTY")
    print("=" * 60)
    main()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
