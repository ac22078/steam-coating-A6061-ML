"""
04_ale.py
1D-ALE and 2D-ALE analysis with bootstrap 95% CI.
Requires: final model saved by 01_nested_cv.py
"""
import os
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

from PyALE import ale
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel as C, WhiteKernel
from sklearn.utils import resample
from sklearn.base import clone

# ==========================================
# Configuration
# ==========================================
TRAIN_CSV = "A5.csv"
MODEL_PATH = Path("model_artifacts/final_best_pipe_Random Forest.joblib")
RESULTS_DIR = Path(os.path.join(os.path.expanduser("~"), "Desktop", "ALE_Results"))

ALE_GRID_1D = 10
ALE_GRID_2D = 3
DO_2D = True
BOOTSTRAP_ITERATIONS = 1000
TARGET_FEATURES_1D = ["SQ", "FT", "FS", "CS"]
INDIVIDUAL_1D_FIGS = True
SCATTER_OVERLAY_2D = True
MARK_LOW_DENSITY_2D = True
LOW_DENSITY_THRESHOLD = 5

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
(RESULTS_DIR / "1D_ALE").mkdir(exist_ok=True)
(RESULTS_DIR / "2D_ALE").mkdir(exist_ok=True)
print("Resolved RESULTS_DIR:", RESULTS_DIR.resolve())

FIG_WIDTH, FIG_HEIGHT = 3.46, 3.46
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})
mpl.rcParams["hatch.linewidth"] = 1
warnings.filterwarnings("ignore")

def labels_to_edges(labels):
    labels = np.asarray(labels, dtype=float)

    if len(labels) == 1:
        delta = max(abs(labels[0]) * 0.05, 1e-6)
        return np.array([labels[0] - delta, labels[0] + delta])

    mids = (labels[:-1] + labels[1:]) / 2.0
    first = labels[0] - (mids[0] - labels[0])
    last = labels[-1] + (labels[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


# ==========================================
# Utilities
# ==========================================
def load_data(file_path: str):
    for enc in ["utf-8-sig", "cp932"]:
        try:
            df = pd.read_csv(file_path, encoding=enc, index_col=0)
            df = df.select_dtypes(include=[np.number]).dropna()
            if df.empty:
                continue
            return df.iloc[:, :-1], df.iloc[:, -1]
        except Exception:
            continue
    raise ValueError(f"CSV reading failed for path: {file_path}")


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
            "model": RandomForestRegressor(random_state=42, n_estimators=500, n_jobs=-1),
            "params": {
                "max_depth": [3, 4, 5],
                "min_samples_leaf": [3, 5],
                "max_features": ["sqrt", None],
            },
            "use_scaler": False,
        },
        "GBR": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "loss": ["huber"],
                "alpha": [0.9],
                "learning_rate": [0.03, 0.1],
                "n_estimators": [200, 500, 1000],
                "max_depth": [1, 2],
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
    vals = ale_res.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None
    abs_vals = np.abs(vals)
    return {
        "mean_abs": float(np.mean(abs_vals)),
        "range": float(np.max(vals) - np.min(vals)),
        "std": float(np.std(vals)),
        "max_abs": float(np.max(abs_vals)),
    }


def compute_confidence_intervals(bootstrap_results, confidence=0.95):
    alpha = 1 - confidence
    ci_records = []
    for (f1, f2), values in bootstrap_results.items():
        if len(values) == 0:
            continue
        values = np.array(values)
        ci_records.append({
            "Feature_1": f1,
            "Feature_2": f2,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "ci_lower": float(np.percentile(values, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
            "n_samples": len(values),
        })
    return pd.DataFrame(ci_records)


def plot_interaction_with_ci(ci_df, results_dir, top_k=6):
    ci_df_sorted = ci_df.sort_values("mean", ascending=False).head(top_k).reset_index(drop=True)
    ci_df_sorted["pair"] = ci_df_sorted["Feature_1"] + "-" + ci_df_sorted["Feature_2"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.2, FIG_HEIGHT))
    y_pos = np.arange(len(ci_df_sorted))[::-1]
    means = ci_df_sorted["mean"].values[::-1]
    ci_lower = ci_df_sorted["ci_lower"].values[::-1]
    ci_upper = ci_df_sorted["ci_upper"].values[::-1]
    labels = ci_df_sorted["pair"].values[::-1]

    ax.barh(y_pos, means, color="#1f78b4", edgecolor="black", linewidth=0.8, alpha=0.7)
    ax.errorbar(
        means,
        y_pos,
        xerr=[means - ci_lower, ci_upper - means],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("2D-ALE interaction strength (mean |effect|)")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(results_dir / "Fig7_ALE_2D_interaction_with_CI.pdf")
    plt.close(fig)


def _save_single_1d_figure(col, x_v, y_c, ci_lo, ci_hi, X_ale, outpath):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(x_v, y_c, color="#2a6f97", lw=1.2, zorder=3)
    ax.fill_between(x_v, ci_lo, ci_hi, alpha=0.22, color="#8ea6b4", zorder=2, label="95% confidence interval")
    ax.axhline(0, color="black", lw=0.6, ls="--", zorder=1)

    span = max(ci_hi.max() - ci_lo.min(), 1e-9)
    rug_y = ci_lo.min() - span * 0.05
    ax.plot(X_ale[col].values, np.full(len(X_ale), rug_y), "|", color="#9a9a9a", alpha=0.45, ms=4, lw=0.5, zorder=4)

    ax.set_xlabel(f"{col} (nm)" if col in ["SQ", "FT"] else col)
    ax.set_ylabel("ALE for Epit")
    ax.legend(loc="upper left", fontsize=6, frameon=True)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


# ==========================================
# 1D ALE Analysis
# ==========================================
def run_1d_ale_with_bootstrap(best_pipe, X, y, results_dir: Path, target_features: list, ale_grid=10, bootstrap_n=1000, seed=42, individual_figs=True):
    results_dir = Path(results_dir)
    ale_1d_dir = results_dir / "1D_ALE"

    np.random.seed(seed)
    X_ale = X.reset_index(drop=True).copy()
    y_ale = y.reset_index(drop=True).copy()
    boot_ale_store = {col: [] for col in target_features}
    success_counts = {col: 0 for col in target_features}

    print("\n--- Starting 1D ALE Bootstrap ---")
    for i in range(bootstrap_n):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1} / {bootstrap_n} iterations")

        X_boot, y_boot = resample(X_ale, y_ale, random_state=seed + i)
        X_boot = X_boot.reset_index(drop=True)
        y_boot = y_boot.reset_index(drop=True)

        pipe = clone(best_pipe)
        try:
            pipe.fit(X_boot, y_boot)
        except Exception:
            continue

        for col in target_features:
            try:
                ale_res = ale(X=X_boot, model=pipe, feature=[col], grid_size=ale_grid, plot=False)
                num_col = ale_res.select_dtypes(include=[np.number]).columns[0]
                boot_ale_store[col].append(ale_res[num_col].values)
                success_counts[col] += 1
            except Exception:
                continue

    print("\n[1D Bootstrap Success Rates]")
    for col, count in success_counts.items():
        print(f"  - {col}: {count} / {bootstrap_n} successful iterations")

    main_ale = {}
    for col in target_features:
        try:
            main_ale[col] = ale(X=X_ale, model=best_pipe, feature=[col], grid_size=ale_grid, plot=False)
        except Exception:
            main_ale[col] = None

    n_feat = len(target_features)
    ncols = 2
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH * ncols, FIG_HEIGHT * nrows))
    axes = axes.flatten()
    source_records = []

    for ax_idx, col in enumerate(target_features):
        ax = axes[ax_idx]
        ale_res = main_ale.get(col)
        if ale_res is None:
            ax.set_visible(False)
            continue

        num_col = ale_res.select_dtypes(include=[np.number]).columns[0]
        x_vals = ale_res.index.values
        y_center = ale_res[num_col].values
        boots = boot_ale_store[col]

        if len(boots) > 10:
            min_len = min(len(b) for b in boots)
            boots = np.array([b[:min_len] for b in boots])
            x_v = x_vals[:min_len]
            y_c = y_center[:min_len]
            ci_lo = np.percentile(boots, 2.5, axis=0)
            ci_hi = np.percentile(boots, 97.5, axis=0)
        else:
            x_v, y_c = x_vals, y_center
            ci_lo, ci_hi = y_center.copy(), y_center.copy()

        ax.plot(x_v, y_c, color="#2a6f97", lw=1.2, zorder=3)
        ax.fill_between(x_v, ci_lo, ci_hi, alpha=0.22, color="#8ea6b4", zorder=2, label="95% CI")
        ax.axhline(0, color="black", lw=0.6, ls="--", zorder=1)

        rug_y = ci_lo.min() - (max(ci_hi.max() - ci_lo.min(), 1e-9)) * 0.05
        ax.plot(X_ale[col].values, np.full(len(X_ale), rug_y), "|", color="#9a9a9a", alpha=0.45, ms=4, lw=0.5, zorder=4)

        ax.set_xlabel(col)
        ax.set_ylabel("ALE (V)")
        ax.legend(fontsize=6, frameon=True)
        label = chr(ord("a") + ax_idx)
        ax.text(-0.12, 1.02, f"({label})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

        for xi, yi, lo, hi in zip(x_v, y_c, ci_lo, ci_hi):
            source_records.append({"feature": col, "x": xi, "ale_mean": yi, "ci_lower": lo, "ci_upper": hi})

        pd.DataFrame({
            "x": x_v,
            "ale_mean": y_c,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi
        }).to_csv(ale_1d_dir / f"SourceData_ALE_1D_{col}.csv", index=False)

        if individual_figs:
            _save_single_1d_figure(col, x_v, y_c, ci_lo, ci_hi, X_ale, ale_1d_dir / f"Fig6_ALE_1D_{col}.pdf")

    for ax_idx in range(n_feat, len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(results_dir / "Fig6_ALE_1D_panel.pdf")
    plt.close(fig)

    src_df = pd.DataFrame(source_records)
    src_df.to_csv(results_dir / "SourceData_Fig6_ALE_1D_bootstrap.csv", index=False)
    return src_df


# ==========================================
# 2D ALE Analysis
# ==========================================
def summarize_2d_ale_density_filtered(source_csv_path, feature_1, feature_2, min_samples=5):
    df = pd.read_csv(source_csv_path)
    df["abs_ALE"] = df["ALE"].abs()
    df_filt = df[df["n_samples"] >= min_samples].copy()

    summary = {
        "Feature_1": feature_1,
        "Feature_2": feature_2,
        "threshold_n": min_samples,
        "total_cells": int(len(df)),
        "retained_cells": int(len(df_filt)),
        "retained_sample_sum": int(df_filt["n_samples"].sum()) if len(df_filt) else 0,
        "mean_abs_all_cells": float(df["abs_ALE"].mean()),
        "mean_abs_filtered": float(df_filt["abs_ALE"].mean()) if len(df_filt) else np.nan,
        "sample_weighted_abs_filtered": float((df_filt["abs_ALE"] * df_filt["n_samples"]).sum() / df_filt["n_samples"].sum()) if len(df_filt) and df_filt["n_samples"].sum() > 0 else np.nan,
        "max_abs_filtered": float(df_filt["abs_ALE"].max()) if len(df_filt) else np.nan,
    }
    return df, df_filt, summary


def bootstrap_2d_ale_interaction_strength(best_pipe, X, y, feature_pairs, n_iterations=1000, ale_grid_size=3, seed=42, min_samples_filter=5):
    np.random.seed(seed)
    bootstrap_results = {pair: [] for pair in feature_pairs}
    success_counts = {pair: 0 for pair in feature_pairs}

    print("\n--- Starting 2D ALE Bootstrap ---")
    for i in range(n_iterations):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1} / {n_iterations} iterations")

        X_boot, y_boot = resample(X, y, random_state=seed + i)
        X_boot, y_boot = X_boot.reset_index(drop=True), y_boot.reset_index(drop=True)
        pipe = clone(best_pipe)

        try:
            pipe.fit(X_boot, y_boot)
        except Exception as e:
            print(f"BOOT FIT FAIL iter {i}: {e}")
            continue

        for f1, f2 in feature_pairs:
            try:
                ale_res = ale(
                    X=X_boot,
                    model=pipe,
                    feature=[f1, f2],
                    grid_size=ale_grid_size,
                    plot=False
                )

                yedges = labels_to_edges(ale_res.index.values)
                xedges = labels_to_edges(ale_res.columns.values)

                H, _, _ = np.histogram2d(
                    X_boot[f2].values,
                    X_boot[f1].values,
                    bins=[xedges, yedges]
                )

                if H.T.shape != ale_res.shape:
                    print(f"BOOT SHAPE MISMATCH {f1}-{f2}: H.T={H.T.shape}, ALE={ale_res.shape}")
                    continue

                mask = H.T >= min_samples_filter
                valid_ales = ale_res.values[mask]

                if len(valid_ales) > 0:
                    mean_abs_filtered = float(np.mean(np.abs(valid_ales)))
                    bootstrap_results[(f1, f2)].append(mean_abs_filtered)
                    success_counts[(f1, f2)] += 1

            except Exception as e:
                print(f"BOOT ALE FAIL {f1}-{f2} iter {i}: {e}")
                continue

    print("\n[2D Bootstrap Success Rates]")
    for pair, count in success_counts.items():
        print(f"  - {pair[0]}-{pair[1]}: {count} / {n_iterations} successful iterations")

    return bootstrap_results


def run_comprehensive_interpretation(best_pipe, X, y, results_dir: Path, target_features: list, ale_grid_2d=3, do_bootstrap=True, bootstrap_n=1000, min_samples_filter=5):
    results_dir = Path(results_dir)
    ale_2d_dir = results_dir / "2D_ALE"
    X_ale = X.reset_index(drop=True).copy()

    top_feats = [f for f in target_features if f in X_ale.columns]
    feature_pairs = list(itertools.combinations(top_feats, 2))
    interaction_records = []
    filtered_records = []

    for f1, f2 in feature_pairs:
        fig = None
        try:
            ale_res = ale(
                X=X_ale,
                model=best_pipe,
                feature=[f1, f2],
                grid_size=ale_grid_2d,
                plot=False
            )

            fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.1, FIG_HEIGHT))

            yedges = labels_to_edges(ale_res.index.values)
            xedges = labels_to_edges(ale_res.columns.values)

            mesh = ax.pcolormesh(
                xedges,
                yedges,
                ale_res.values,
                cmap="viridis",
                shading="flat",
                alpha=0.92,
                zorder=1
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Effect on prediction", rotation=270, labelpad=15)

            H, _, _ = np.histogram2d(
                X_ale[f2].values,
                X_ale[f1].values,
                bins=[xedges, yedges]
            )

            if H.T.shape != ale_res.shape:
                print(f"Skipping {f1}-{f2}: Histogram shape mismatch (expected {ale_res.shape}, got {H.T.shape})")
                plt.close(fig)
                continue

            # 0セルは白塗り
            for i in range(len(xedges) - 1):
                for j in range(len(yedges) - 1):
                    count = int(H[i, j])
                    xmin, xmax = xedges[i], xedges[i + 1]
                    ymin, ymax = yedges[j], yedges[j + 1]

                    if count == 0:
                        rect = patches.Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            linewidth=0,
                            facecolor="white",
                            zorder=4
                        )
                        ax.add_patch(rect)

            # 低密度セルはハッチ
            if MARK_LOW_DENSITY_2D:
                for i in range(len(xedges) - 1):
                    for j in range(len(yedges) - 1):
                        count = int(H[i, j])
                        xmin, xmax = xedges[i], xedges[i + 1]
                        ymin, ymax = yedges[j], yedges[j + 1]

                        if 1 <= count < LOW_DENSITY_THRESHOLD:
                            rect = patches.Rectangle(
                                (xmin, ymin),
                                xmax - xmin,
                                ymax - ymin,
                                linewidth=0,
                                edgecolor="black",
                                facecolor="white",
                                alpha=0.12,
                                hatch="//////////",
                                zorder=5
                            )
                            ax.add_patch(rect)

            # n=0 は大きい空セルだけ表示
            x_total = xedges[-1] - xedges[0]
            y_total = yedges[-1] - yedges[0]
            min_label_w = 0.18 * x_total
            min_label_h = 0.18 * y_total

            for i in range(len(xedges) - 1):
                for j in range(len(yedges) - 1):
                    count = int(H[i, j])
                    if count != 0:
                        continue

                    xmin, xmax = xedges[i], xedges[i + 1]
                    ymin, ymax = yedges[j], yedges[j + 1]
                    w = xmax - xmin
                    h = ymax - ymin

                    if (w >= min_label_w) and (h >= min_label_h):
                        xcenter = (xmin + xmax) / 2
                        ycenter = (ymin + ymax) / 2
                        ax.text(
                            xcenter,
                            ycenter,
                            "n=0",
                            color="black",
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            zorder=6
                        )

            if SCATTER_OVERLAY_2D:
                ax.scatter(
                    X_ale[f2],
                    X_ale[f1],
                    color="black",
                    alpha=0.55,
                    s=30,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=10
                )

            ax.set_xlim(xedges[0], xedges[-1])
            ax.set_ylim(yedges[0], yedges[-1])
            ax.set_xlabel(f2)
            ax.set_ylabel(f1)
            plt.tight_layout()

            outpath = ale_2d_dir / f"Fig8_ALE_2D_{f1}_{f2}_with_density.pdf"
            plt.savefig(outpath)
            plt.close(fig)
            print("Saved:", outpath, "exists:", outpath.exists())

            H_df = pd.DataFrame(H.T, index=ale_res.index, columns=ale_res.columns)

            ale_flat = ale_res.stack().reset_index()
            ale_flat.columns = [f1, f2, "ALE"]

            count_flat = H_df.stack().reset_index()
            count_flat.columns = [f1, f2, "n_samples"]

            merged_df = pd.merge(ale_flat, count_flat, on=[f1, f2])
            src_path = ale_2d_dir / f"SourceData_ALE_2D_{f1}_{f2}.csv"
            merged_df.to_csv(src_path, index=False)

            _, df_filt, filt_summary = summarize_2d_ale_density_filtered(
                src_path, f1, f2, min_samples=min_samples_filter
            )
            df_filt.to_csv(
                ale_2d_dir / f"SourceData_ALE_2D_{f1}_{f2}_n_ge_{min_samples_filter}.csv",
                index=False
            )

            filtered_records.append(filt_summary)

            stats = compute_interaction_strength_statistics(ale_res)
            if stats:
                interaction_records.append({"Feature_1": f1, "Feature_2": f2, **stats})

        except Exception as e:
            print(f"Error in plotting {f1}-{f2}: {type(e).__name__}: {e}")
            if fig is not None:
                plt.close(fig)
            continue

    if do_bootstrap and interaction_records:
        boot_results = bootstrap_2d_ale_interaction_strength(
            best_pipe=best_pipe,
            X=X_ale,
            y=y,
            feature_pairs=feature_pairs,
            n_iterations=bootstrap_n,
            ale_grid_size=ale_grid_2d,
            seed=42,
            min_samples_filter=min_samples_filter
        )

        ci_df = compute_confidence_intervals(boot_results, confidence=0.95)
        ci_df.to_csv(
            results_dir / f"SourceData_Fig7_ALE_2D_bootstrap_CI_n_ge_{min_samples_filter}.csv",
            index=False
        )
        if len(ci_df) > 0:
            plot_interaction_with_ci(ci_df, results_dir, top_k=len(feature_pairs))

    if interaction_records:
        df_all = pd.DataFrame(interaction_records)
        df_all.to_csv(results_dir / "SourceData_ALE_2D_all_raw.csv", index=False)

    if filtered_records:
        df_filtered_summary = pd.DataFrame(filtered_records)
        if "mean_abs_filtered" in df_filtered_summary.columns:
            df_filtered_summary = df_filtered_summary.sort_values("mean_abs_filtered", ascending=False)

        df_filtered_summary.to_csv(
            results_dir / f"SourceData_ALE_2D_interaction_strength_summary_n_ge_{min_samples_filter}.csv",
            index=False
        )
        return df_filtered_summary

    return None


if __name__ == "__main__":
    print("=" * 70)
    print("ALE ANALYSIS WITH OVERLAY (FULL FIXED VERSION)")
    print("=" * 70)

    X, y = load_data(TRAIN_CSV)
    print(f"Data: X={X.shape}, features={list(X.columns)}")

    best_pipe = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH.name}")

    run_1d_ale_with_bootstrap(
        best_pipe=best_pipe,
        X=X, y=y,
        results_dir=RESULTS_DIR,
        target_features=TARGET_FEATURES_1D,
        ale_grid=ALE_GRID_1D,
        bootstrap_n=BOOTSTRAP_ITERATIONS,
        seed=42,
        individual_figs=INDIVIDUAL_1D_FIGS
    )

    if DO_2D:
        res_df = run_comprehensive_interpretation(
            best_pipe=best_pipe,
            X=X, y=y,
            results_dir=RESULTS_DIR,
            target_features=TARGET_FEATURES_1D,
            ale_grid_2d=ALE_GRID_2D,
            do_bootstrap=True,
            bootstrap_n=BOOTSTRAP_ITERATIONS,
            min_samples_filter=LOW_DENSITY_THRESHOLD
        )

        if res_df is not None:
            print("\n[Summary of Filtered 2D ALE Interactions]")
            print(res_df[["Feature_1", "Feature_2", "mean_abs_filtered", "retained_cells"]].head())

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Results saved to: {RESULTS_DIR.resolve()}")
    print("=" * 70)
