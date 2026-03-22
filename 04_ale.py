"""
04_ale.py
1D-ALE and 2D-ALE analysis with bootstrap 95% CI.
Requires: final model saved by 01_nested_cv.py
"""
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
from PyALE import ale
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, DotProduct, ConstantKernel as C, WhiteKernel,
)
from sklearn.utils import resample

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_CSV            = "A5.csv"
RESULTS_DIR          = Path("./results")
MODEL_PATH           = RESULTS_DIR / "model_artifacts" / "final_best_pipe_Random Forest.joblib"

ALE_GRID_1D          = 10
ALE_GRID_2D          = 5
TOP_N_2D             = 4
DO_2D                = True
BOOTSTRAP_ITERATIONS = 1000
TARGET_FEATURES_1D   = ["SQ", "FT", "FS", "CS"]

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "1D_ALE").mkdir(exist_ok=True)
(RESULTS_DIR / "2D_ALE").mkdir(exist_ok=True)

FIG_WIDTH, FIG_HEIGHT = 3.46, 3.46
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "ytick.right":      True,
    "savefig.dpi":      600,
    "savefig.bbox":     "tight",
})
warnings.filterwarnings("ignore")

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


def get_models():
    kernel1 = C(1.0) * Matern(nu=1.5) + WhiteKernel()
    kernel2 = DotProduct() + RBF() + WhiteKernel()
    kernel3 = C(1.0) * RBF() + WhiteKernel()
    return {
        "GPR": {
            "model": GaussianProcessRegressor(random_state=42, n_restarts_optimizer=5),
            "params": {"kernel": [kernel1, kernel2, kernel3], "alpha": [1e-10, 1e-6]},
            "use_scaler": True,
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_estimators=500),
            "params": {
                "max_depth":        [3, 4, 5],
                "min_samples_leaf": [3, 5],
                "max_features":     ["sqrt", None],
            },
            "use_scaler": False,
        },
        "GBR": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "loss":          ["huber"],
                "alpha":         [0.9],
                "learning_rate": [0.03, 0.1],
                "n_estimators":  [200, 500, 1000],
                "max_depth":     [1, 2],
            },
            "use_scaler": False,
        },
        "SVR": {
            "model": SVR(kernel="rbf"),
            "params": {"C": [1, 10, 100], "epsilon": [0.05, 0.1], "gamma": ["scale", 0.1]},
            "use_scaler": True,
        },
        "Elastic Net": {
            "model": ElasticNet(random_state=42),
            "params": {"alpha": [1e-4, 1e-2, 1.0], "l1_ratio": [0.1, 0.5, 0.9, 1.0]},
            "use_scaler": True,
        },
    }


def make_pipeline(model_info):
    steps = []
    if model_info.get("use_scaler", True):
        steps.append(("scaler", RobustScaler()))
    steps.append(("model", model_info["model"]))
    return Pipeline(steps)


def compute_interaction_strength_statistics(ale_res):
    num_cols = ale_res.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return None
    vals     = ale_res[num_cols[0]].values
    abs_vals = np.abs(vals)
    return {
        "mean_abs": float(np.mean(abs_vals)),
        "range":    float(np.max(vals) - np.min(vals)),
        "std":      float(np.std(vals)),
        "max_abs":  float(np.max(abs_vals)),
    }


def compute_confidence_intervals(bootstrap_results, confidence=0.95):
    alpha      = 1 - confidence
    ci_records = []
    for (f1, f2), values in bootstrap_results.items():
        if len(values) == 0:
            continue
        values = np.array(values)
        ci_records.append({
            "Feature_1": f1,
            "Feature_2": f2,
            "mean":      float(np.mean(values)),
            "std":       float(np.std(values, ddof=1)),
            "ci_lower":  float(np.percentile(values, 100 * alpha / 2)),
            "ci_upper":  float(np.percentile(values, 100 * (1 - alpha / 2))),
            "n_samples": len(values),
        })
    return pd.DataFrame(ci_records)


def plot_interaction_with_ci(ci_df, results_dir, top_k=6):
    ci_df_sorted         = ci_df.sort_values("mean", ascending=False).head(top_k).reset_index(drop=True)
    ci_df_sorted["pair"] = ci_df_sorted["Feature_1"] + "-" + ci_df_sorted["Feature_2"]
    fig, ax  = plt.subplots(figsize=(FIG_WIDTH * 1.2, FIG_HEIGHT))
    y_pos    = np.arange(len(ci_df_sorted))[::-1]
    means    = ci_df_sorted["mean"].values[::-1]
    ci_lower = ci_df_sorted["ci_lower"].values[::-1]
    ci_upper = ci_df_sorted["ci_upper"].values[::-1]
    labels   = ci_df_sorted["pair"].values[::-1]
    ax.barh(y_pos, means, color="#1f78b4", edgecolor="black", linewidth=0.8, alpha=0.7)
    ax.errorbar(means, y_pos, xerr=[means - ci_lower, ci_upper - means],
                fmt="none", ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("2D-ALE interaction strength (mean |effect|)")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(results_dir / "Fig7_ALE_2D_interaction_with_CI.pdf")
    plt.close()
    print(f"   ✓ Saved: Fig7_ALE_2D_interaction_with_CI.pdf")


# ----------------------------------------------------------------------
# 1D-ALE + Bootstrap 95% CI (Fig. 6)
# ----------------------------------------------------------------------
def run_1d_ale_with_bootstrap(best_pipe, X, y, model_info, results_dir: Path,
                               target_features: list,
                               ale_grid=10, bootstrap_n=1000, seed=42):
    results_dir = Path(results_dir)
    ale_1d_dir  = results_dir / "1D_ALE"
    ale_1d_dir.mkdir(exist_ok=True)

    np.random.seed(seed)
    X_ale = X.reset_index(drop=True).copy()
    y_ale = y.reset_index(drop=True).copy()

    boot_ale_store = {col: [] for col in target_features}
    print(f"\n[1D-ALE Bootstrap] {bootstrap_n} iterations × {len(target_features)} features...")

    for i in range(bootstrap_n):
        if (i + 1) % 200 == 0:
            print(f"   Iteration {i+1}/{bootstrap_n}...", flush=True)
        X_boot, y_boot = resample(X_ale, y_ale, random_state=seed + i)
        X_boot = X_boot.reset_index(drop=True)
        y_boot = y_boot.reset_index(drop=True)
        pipe = make_pipeline(model_info)
        try:
            pipe.fit(X_boot, y_boot)
        except Exception:
            continue
        for col in target_features:
            try:
                ale_res = ale(X=X_boot, model=pipe, feature=[col],
                              grid_size=ale_grid, plot=False)
                num_col = ale_res.select_dtypes(include=[np.number]).columns[0]
                boot_ale_store[col].append(ale_res[num_col].values)
            except Exception:
                continue

    main_ale = {}
    for col in target_features:
        try:
            main_ale[col] = ale(X=X_ale, model=best_pipe, feature=[col],
                                grid_size=ale_grid, plot=False)
        except Exception:
            main_ale[col] = None

    n_feat = len(target_features)
    ncols  = 2
    nrows  = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIG_WIDTH * ncols, FIG_HEIGHT * nrows))
    axes = axes.flatten()
    source_records = []

    for ax_idx, col in enumerate(target_features):
        ax      = axes[ax_idx]
        ale_res = main_ale.get(col)
        if ale_res is None:
            ax.set_visible(False)
            continue
        num_col  = ale_res.select_dtypes(include=[np.number]).columns[0]
        x_vals   = ale_res.index.values
        y_center = ale_res[num_col].values

        boots = boot_ale_store[col]
        if len(boots) > 10:
            min_len = min(len(b) for b in boots)
            boots   = np.array([b[:min_len] for b in boots])
            x_v     = x_vals[:min_len]
            y_c     = y_center[:min_len]
            ci_lo   = np.percentile(boots, 2.5,  axis=0)
            ci_hi   = np.percentile(boots, 97.5, axis=0)
        else:
            x_v, y_c     = x_vals, y_center
            ci_lo, ci_hi = y_center, y_center

        ax.plot(x_v, y_c, color="#1f78b4", lw=1.2, zorder=3)
        ax.fill_between(x_v, ci_lo, ci_hi,
                        alpha=0.3, color="gray", zorder=2, label="95% CI")
        ax.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
        rug_y = ci_lo.min() - (ci_hi.max() - ci_lo.min()) * 0.05
        ax.plot(X_ale[col].values, np.full(len(X_ale), rug_y),
                "|", color="black", alpha=0.4, ms=4, lw=0.5, zorder=4)
        ax.set_xlabel(col)
        ax.set_ylabel(r"ALE (V)")
        ax.legend(fontsize=6, frameon=True)
        ax.grid(alpha=0.3, linestyle="--")
        label = chr(ord("a") + ax_idx)
        ax.text(-0.12, 1.02, f"({label})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")

        for xi, yi, lo, hi in zip(x_v, y_c, ci_lo, ci_hi):
            source_records.append({"feature": col, "x": xi,
                                   "ale_mean": yi, "ci_lower": lo, "ci_upper": hi})
        pd.DataFrame({"x": x_v, "ale_mean": y_c,
                      "ci_lower": ci_lo, "ci_upper": ci_hi}).to_csv(
            ale_1d_dir / f"SourceData_ALE_1D_{col}.csv", index=False)

    for ax_idx in range(n_feat, len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(results_dir / "Fig6_ALE_1D_panel.pdf")
    plt.close()
    print(f"   ✓ Saved: Fig6_ALE_1D_panel.pdf")

    src_df = pd.DataFrame(source_records)
    src_df.to_csv(results_dir / "SourceData_Fig6_ALE_1D_bootstrap.csv", index=False)
    print(f"   ✓ Saved: SourceData_Fig6_ALE_1D_bootstrap.csv")
    return src_df


# ----------------------------------------------------------------------
# 2D-ALE + Bootstrap (Fig. 7 & 8)
# ----------------------------------------------------------------------
def bootstrap_2d_ale_interaction_strength(X, y, model_info, feature_pairs,
                                           n_iterations=1000, ale_grid_size=5, seed=42):
    np.random.seed(seed)
    bootstrap_results = {pair: [] for pair in feature_pairs}
    print(f"\n[Bootstrap 2D-ALE] {n_iterations} iterations × {len(feature_pairs)} pairs...")

    for i in range(n_iterations):
        if (i + 1) % 100 == 0:
            print(f"   Iteration {i+1}/{n_iterations}...", flush=True)
        X_boot, y_boot = resample(X, y, random_state=seed + i)
        X_boot = X_boot.reset_index(drop=True)
        y_boot = y_boot.reset_index(drop=True)
        pipe = make_pipeline(model_info)
        try:
            pipe.fit(X_boot, y_boot)
        except Exception:
            continue
        for f1, f2 in feature_pairs:
            try:
                ale_res    = ale(X=X_boot, model=pipe, feature=[f1, f2],
                                 grid_size=ale_grid_size, plot=False)
                stats_dict = compute_interaction_strength_statistics(ale_res)
                if stats_dict is not None:
                    bootstrap_results[(f1, f2)].append(stats_dict["mean_abs"])
            except Exception:
                continue
    return bootstrap_results


def run_comprehensive_interpretation(best_pipe, X, y, model_info, results_dir: Path,
                                      ale_grid_2d=5, do_2d=True, top_n_2d=4,
                                      do_bootstrap=True, bootstrap_n=1000):
    results_dir = Path(results_dir)
    ale_2d_dir  = results_dir / "2D_ALE"
    ale_2d_dir.mkdir(exist_ok=True)
    X_ale = X.reset_index(drop=True).copy()

    if not do_2d:
        return None

    top_feats     = list(X_ale.columns)[:top_n_2d]
    print(f"\n[2D-ALE] Selected features (column order): {top_feats}")
    feature_pairs = list(itertools.combinations(top_feats, 2))
    interaction_records = []

    for idx, (f1, f2) in enumerate(feature_pairs, start=1):
        print(f"   [{idx}/{len(feature_pairs)}] {f1} × {f2}...", end="", flush=True)
        try:
            ale_res = ale(X=X_ale, model=best_pipe, feature=[f1, f2],
                          grid_size=ale_grid_2d, plot=True)
            plt.savefig(ale_2d_dir / f"Fig8_ALE_2D_{f1}_{f2}.pdf")
            plt.close()
            ale_res.to_csv(ale_2d_dir / f"SourceData_ALE_2D_{f1}_{f2}.csv", index=False)
            stats = compute_interaction_strength_statistics(ale_res)
            if stats:
                interaction_records.append({"Feature_1": f1, "Feature_2": f2, **stats})
                print(" OK")
            else:
                print(" SKIP")
        except Exception:
            plt.close()
            print(" FAIL")

    if do_bootstrap and interaction_records:
        boot_results = bootstrap_2d_ale_interaction_strength(
            X=X_ale, y=y, model_info=model_info, feature_pairs=feature_pairs,
            n_iterations=bootstrap_n, ale_grid_size=ale_grid_2d, seed=42,
        )
        ci_df = compute_confidence_intervals(boot_results, confidence=0.95)
        ci_df.to_csv(results_dir / "SourceData_Fig7_ALE_2D_bootstrap_CI.csv", index=False)
        print(f"   ✓ Saved: SourceData_Fig7_ALE_2D_bootstrap_CI.csv")
        plot_interaction_with_ci(ci_df, results_dir, top_k=len(feature_pairs))

    if interaction_records:
        inter_df = pd.DataFrame(interaction_records)
        inter_df.sort_values("mean_abs", ascending=False).to_csv(
            results_dir / "SourceData_ALE_2D_interaction_strength.csv", index=False
        )
        return inter_df
    return None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("ALE ANALYSIS (Fig. 6, 7, 8)")
    print("=" * 70)

    X, y = load_data(TRAIN_CSV)
    print(f"Data: X={X.shape}, features={list(X.columns)}")

    best_pipe = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH.name}")

    m_dict    = get_models()
    best_info = m_dict["Random Forest"]

    run_1d_ale_with_bootstrap(
        best_pipe=best_pipe, X=X, y=y,
        model_info=best_info,
        results_dir=RESULTS_DIR,
        target_features=TARGET_FEATURES_1D,
        ale_grid=ALE_GRID_1D, bootstrap_n=BOOTSTRAP_ITERATIONS, seed=42,
    )

    run_comprehensive_interpretation(
        best_pipe, X, y, best_info, RESULTS_DIR,
        ale_grid_2d=ALE_GRID_2D, do_2d=DO_2D, top_n_2d=TOP_N_2D,
        do_bootstrap=True, bootstrap_n=BOOTSTRAP_ITERATIONS,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
