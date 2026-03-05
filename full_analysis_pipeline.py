import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import shap
from PyALE import ale
from joblib import dump

from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    DotProduct,
    ConstantKernel as C,
    WhiteKernel,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FIG_WIDTH, FIG_HEIGHT = 3.46, 3.46
mpl.rcParams.update(
    {
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
    }
)

RESULTS_DIR = Path("./final_analysis_results")
RESULTS_DIR.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

TRAIN_CSV = "A5.csv"
ALE_GRID_1D = 10
ALE_GRID_2D = 5
TOP_N_2D = 4
DO_2D = True
BOOTSTRAP_ITERATIONS = 1000


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _safe_predict(pipe, X):
    return pipe.predict(X)


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
            "model": GaussianProcessRegressor(
                random_state=42, n_restarts_optimizer=5
            ),
            "params": {
                "kernel": [kernel1, kernel2, kernel3],
                "alpha": [1e-10, 1e-6],
            },
            "use_scaler": True,
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                random_state=42, n_estimators=500
            ),
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
            "params": {
                "C": [1, 10, 100],
                "epsilon": [0.05, 0.1],
                "gamma": ["scale", 0.1],
            },
            "use_scaler": True,
        },
        "Elastic Net": {
            "model": ElasticNet(random_state=42),
            "params": {
                "alpha": [1e-4, 1e-2, 1.0],
                "l1_ratio": [0.1, 0.5, 0.9, 1.0],
            },
            "use_scaler": True,
        },
    }


def make_pipeline(model_info):
    steps = []
    if model_info.get("use_scaler", True):
        steps.append(("scaler", RobustScaler()))
    steps.append(("model", model_info["model"]))
    return Pipeline(steps)


def save_yy_plot(y_true, y_pred, out_pdf: Path):
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    l_min = min(y_true.min(), y_pred.min())
    l_max = max(y_true.max(), y_pred.max())
    plt.plot([l_min, l_max], [l_min, l_max], "b--", lw=0.8, zorder=1)
    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        edgecolors="k",
        s=18,
        color="#e31a1c",
        linewidths=0.4,
        zorder=2,
    )
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    stats_text = f"$R^2$: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}"
    ax = plt.gca()
    ax.text(
        0.05,
        0.92,
        stats_text,
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.7,
            edgecolor="none",
        ),
    )
    plt.xlabel("Measured value")
    plt.ylabel("Predicted value")
    plt.savefig(out_pdf)
    plt.close()


# ----------------------------------------------------------------------
# Nested CV
# ----------------------------------------------------------------------
def nested_cv_oof(X, y, model_info, outer_splits=5, outer_repeats=10, seed=42):
    outer_cv = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=seed,
    )
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2_list, rmse_list, mae_list = [], [], []
    fold_records = []
    pred_sum = pd.Series(0.0, index=X.index)
    pred_cnt = pd.Series(0, index=X.index, dtype=int)

    for fold_id, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
        pipe = make_pipeline(model_info)
        p_grid = {
            f"model__{k}": v for k, v in model_info["params"].items()
        }
        grid = GridSearchCV(
            pipe,
            p_grid,
            cv=inner_cv,
            scoring="r2",
            n_jobs=-1,
        )
        grid.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        best_est = grid.best_estimator_

        y_pred = best_est.predict(X.iloc[te_idx])
        idx = X.iloc[te_idx].index
        pred_sum.loc[idx] += y_pred
        pred_cnt.loc[idx] += 1

        yt = y.iloc[te_idx]
        r2 = r2_score(yt, y_pred)
        rmse = np.sqrt(mean_squared_error(yt, y_pred))
        mae = mean_absolute_error(yt, y_pred)
        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)
        fold_records.append(
            {
                "fold": fold_id,
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
            }
        )

    oof_pred = pred_sum / pred_cnt
    fold_df = pd.DataFrame(fold_records)
    metrics = {
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list, ddof=1)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list, ddof=1)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list, ddof=1)),
    }
    return metrics, oof_pred, fold_df


# ----------------------------------------------------------------------
# ALE + interaction
# ----------------------------------------------------------------------
def compute_interaction_strength_statistics(ale_res):
    num_cols = ale_res.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return None
    vals = ale_res[num_cols[0]].values
    abs_vals = np.abs(vals)
    return {
        "mean_abs": float(np.mean(abs_vals)),
        "range": float(np.max(vals) - np.min(vals)),
        "std": float(np.std(vals)),
        "max_abs": float(np.max(abs_vals)),
    }


def _compute_shap_any_model(best_pipe, X, results_dir: Path):
    results_dir = Path(results_dir)

    try:
        print("\n[SHAP Analysis on Final Model]")
        print(f"   Total samples: {len(X)} (all samples)")

        if hasattr(best_pipe, "named_steps"):
            model = best_pipe.named_steps["model"]
            print(f"   Model type: {type(model).__name__}")

            if "scaler" in best_pipe.named_steps:
                scaler = best_pipe.named_steps["scaler"]
                X_transformed = pd.DataFrame(
                    scaler.transform(X),
                    columns=X.columns,
                    index=X.index,
                )
                print("   Preprocessing: RobustScaler applied")
            else:
                X_transformed = X.copy()
                print("   Preprocessing: None")
        else:
            model = best_pipe
            X_transformed = X.copy()

        print("   Using shap.Explainer (auto-selection)...")
        explainer = shap.Explainer(model, X_transformed)

        explainer_type = type(explainer).__name__
        print(f"   → Auto-selected: {explainer_type}")
        if "Tree" in explainer_type:
            print("   → TreeExplainer computes exact Shapley values")

        print(f"   Computing SHAP values for all {len(X)} samples...")
        shap_values = explainer(X_transformed)
        feat_names = X.columns.tolist()
        shap_vals = shap_values.values
        print("   ✓ SHAP computation completed")

        shap_mean_abs = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": feat_names, "importance": shap_mean_abs}
        )

        # Bar plot
        print("   Creating Fig. 5(a): Bar plot...")
        importance_df_sorted = importance_df.sort_values(
            "importance", ascending=True
        )
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        y_pos = np.arange(len(importance_df_sorted))
        plt.barh(
            y_pos,
            importance_df_sorted["importance"].values,
            color="#1f78b4",
            edgecolor="black",
            linewidth=0.8,
        )
        plt.yticks(
            y_pos,
            importance_df_sorted["feature"].values,
            fontsize=8,
        )
        plt.xlabel("Mean |SHAP value|", fontsize=9)
        plt.ylabel("")
        plt.title("SHAP Feature Importance", fontsize=10, fontweight="bold")
        plt.grid(axis="x", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(results_dir / "05a_SHAP_importance_bar.pdf", dpi=300)
        plt.close()
        print("   ✓ Saved: 05a_SHAP_importance_bar.pdf")

        # Beeswarm
        print("   Creating Fig. 5(b): Beeswarm plot...")
        plt.figure(figsize=(8, 10))
        shap.summary_plot(
            shap_values,
            features=X_transformed,
            feature_names=feat_names,
            max_display=10,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(results_dir / "05b_SHAP_beeswarm.pdf", dpi=300)
        plt.close()
        print("   ✓ Saved: 05b_SHAP_beeswarm.pdf")

        importance_df.sort_values(
            "importance", ascending=False
        ).to_csv(
            results_dir / "02_SHAP_importance_ranking.csv", index=False
        )
        print("   ✓ Saved: 02_SHAP_importance_ranking.csv")

        print("\n   Feature Importance Ranking (mean |SHAP|):")
        for _, row in importance_df.sort_values(
            "importance", ascending=False
        ).iterrows():
            print(f"      {row['feature']:8s}  {row['importance']:.6f}")

    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        import traceback

        traceback.print_exc()


def bootstrap_2d_ale_interaction_strength(
    X,
    y,
    model_info,
    feature_pairs,
    n_iterations=1000,
    ale_grid_size=5,
    seed=42,
):
    np.random.seed(seed)
    bootstrap_results = {pair: [] for pair in feature_pairs}
    print(
        f"\n[Bootstrap] Running {n_iterations} iterations "
        f"for {len(feature_pairs)} pairs..."
    )

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
                ale_res = ale(
                    X=X_boot,
                    model=pipe,
                    feature=[f1, f2],
                    grid_size=ale_grid_size,
                    plot=False,
                )
                stats_dict = compute_interaction_strength_statistics(ale_res)
                if stats_dict is not None:
                    bootstrap_results[(f1, f2)].append(
                        stats_dict["mean_abs"]
                    )
            except Exception:
                continue

    return bootstrap_results


def compute_confidence_intervals(bootstrap_results, confidence=0.95):
    alpha = 1 - confidence
    ci_records = []
    for (f1, f2), values in bootstrap_results.items():
        if len(values) == 0:
            continue
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        ci_lower = np.percentile(values, 100 * alpha / 2)
        ci_upper = np.percentile(values, 100 * (1 - alpha / 2))
        ci_records.append(
            {
                "Feature_1": f1,
                "Feature_2": f2,
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_samples": len(values),
            }
        )
    return pd.DataFrame(ci_records)


def plot_interaction_with_ci(ci_df, results_dir, top_k=6):
    ci_df_sorted = (
        ci_df.sort_values("mean", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    ci_df_sorted["pair"] = (
        ci_df_sorted["Feature_1"] + "-" + ci_df_sorted["Feature_2"]
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.2, FIG_HEIGHT))
    y_pos = np.arange(len(ci_df_sorted))[::-1]
    means = ci_df_sorted["mean"].values[::-1]
    ci_lower = ci_df_sorted["ci_lower"].values[::-1]
    ci_upper = ci_df_sorted["ci_upper"].values[::-1]
    labels = ci_df_sorted["pair"].values[::-1]

    ax.barh(
        y_pos,
        means,
        color="#1f78b4",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.7,
    )
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
    ax.set_xlabel(
        "2D-ALE interaction strength (mean |effect|)", fontsize=9
    )
    ax.set_title(
        "Top Interaction Effects with 95% CI", fontsize=10, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(results_dir / "07_ALE_2D_interaction_with_CI.pdf", dpi=300)
    plt.close()
    print("   Saved: 07_ALE_2D_interaction_with_CI.pdf")


def run_comprehensive_interpretation(
    best_pipe,
    X,
    y,
    model_info,
    results_dir: Path,
    ale_grid_1d=10,
    ale_grid_2d=3,
    do_2d=True,
    top_n_2d=4,
    do_bootstrap=True,
    bootstrap_n=1000,
):
    results_dir = Path(results_dir)
    ale_1d_dir = results_dir / "1D_ALE"
    ale_2d_dir = results_dir / "2D_ALE"
    ale_1d_dir.mkdir(exist_ok=True)
    ale_2d_dir.mkdir(exist_ok=True)
    X_ale = X.reset_index(drop=True).copy()

    # 1D ALE
    print("\n[Step 1] Computing 1D-ALE plots...")
    for col in X_ale.columns:
        try:
            ale(
                X=X_ale,
                model=best_pipe,
                feature=[col],
                grid_size=ale_grid_1d,
                plot=True,
            )
            plt.savefig(ale_1d_dir / f"ALE_1D_{col}.pdf")
            plt.close()
        except Exception:
            plt.close()

    if not do_2d:
        return None

    # 2D ALE
    print("\n[Step 2] Computing 2D-ALE plots...")
    top_feats = list(X_ale.columns)[:top_n_2d]
    print(f"   Selected features: {top_feats}")

    interaction_records = []
    feature_pairs = list(itertools.combinations(top_feats, 2))

    for idx, (f1, f2) in enumerate(feature_pairs, start=1):
        print(f"   [{idx}/{len(feature_pairs)}] {f1} x {f2}...", end="", flush=True)
        try:
            ale_res = ale(
                X=X_ale,
                model=best_pipe,
                feature=[f1, f2],
                grid_size=ale_grid_2d,
                plot=True,
            )
            plt.savefig(ale_2d_dir / f"ALE_2D_{f1}_{f2}.pdf")
            plt.close()

            stats = compute_interaction_strength_statistics(ale_res)
            if stats:
                interaction_records.append(
                    {"Feature_1": f1, "Feature_2": f2, **stats}
                )
                print(" OK")
            else:
                print(" SKIP")
        except Exception:
            plt.close()
            print(" FAIL")

    if do_bootstrap and interaction_records:
        print("\n[Step 3] Bootstrap confidence interval estimation...")
        boot_results = bootstrap_2d_ale_interaction_strength(
            X=X_ale,
            y=y,
            model_info=model_info,
            feature_pairs=feature_pairs,
            n_iterations=bootstrap_n,
            ale_grid_size=ale_grid_2d,
            seed=42,
        )
        ci_df = compute_confidence_intervals(boot_results, confidence=0.95)
        ci_df.to_csv(
            results_dir / "08_ALE_2D_bootstrap_CI.csv", index=False
        )
        print("   Saved: 08_ALE_2D_bootstrap_CI.csv")

        plot_interaction_with_ci(ci_df, results_dir, top_k=len(feature_pairs))

    if interaction_records:
        inter_df = pd.DataFrame(interaction_records)
        inter_df_sorted = inter_df.sort_values("mean_abs", ascending=False)
        inter_df_sorted.to_csv(
            results_dir / "03_ALE_2D_interaction_strength_full.csv",
            index=False,
        )
        return inter_df

    return None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("FULL ANALYSIS PIPELINE")
    print("=" * 70)

    print(f"\nLoading data from '{TRAIN_CSV}'...")
    X, y = load_data(TRAIN_CSV)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Features: {list(X.columns)}")

    m_dict = get_models()

    print("\n" + "=" * 70)
    print("NESTED CV (5-fold, 10 repeats)")
    print("=" * 70)

    performance_records = []
    oof_store = {}
    fold_all = []

    for name, info in m_dict.items():
        print(f"[{name}]", end=" ", flush=True)
        metrics, oof_pred, fold_df = nested_cv_oof(
            X, y, info, outer_splits=5, outer_repeats=10, seed=42
        )
        performance_records.append({"Model": name, **metrics})
        oof_store[name] = oof_pred
        fold_df["Model"] = name
        fold_all.append(fold_df)
        print(f"R2={metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}")

    perf_df = pd.DataFrame(performance_records)
    perf_df.to_csv(RESULTS_DIR / "performance_summary.csv", index=False)
    pd.concat(fold_all, ignore_index=True).to_csv(
        RESULTS_DIR / "nestedcv_fold_metrics.csv", index=False
    )
    print("\n" + perf_df.to_string(index=False))

    # Learning curve
    print("\n" + "=" * 70)
    print("LEARNING CURVE ANALYSIS")
    print("=" * 70)

    best_name_lc = perf_df.loc[perf_df["r2_mean"].idxmax(), "Model"]
    best_info_lc = m_dict[best_name_lc]
    best_pipe_lc = make_pipeline(best_info_lc)

    train_sizes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator=best_pipe_lc,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1, ddof=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1, ddof=1)

    lc_df = pd.DataFrame(
        {
            "train_size": train_sizes_abs,
            "train_score_mean": train_scores_mean,
            "train_score_std": train_scores_std,
            "cv_score_mean": test_scores_mean,
            "cv_score_std": test_scores_std,
        }
    )
    lc_df.to_csv(
        RESULTS_DIR / f"learning_curve_{best_name_lc}.csv", index=False
    )

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(
        train_sizes_abs,
        train_scores_mean,
        "o-",
        color="#1f78b4",
        label="Training score",
    )
    plt.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="#1f78b4",
    )
    plt.plot(
        train_sizes_abs,
        test_scores_mean,
        "s-",
        color="#e31a1c",
        label="CV score",
    )
    plt.fill_between(
        train_sizes_abs,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="#e31a1c",
    )
    plt.xlabel("Number of training samples")
    plt.ylabel(r"$R^2$")
    plt.xlim(0, 90)
    plt.legend(loc="best", frameon=True)
    plt.grid(alpha=0.3, linestyle="--")
    plt.title(f"Learning Curve ({best_name_lc})")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"06_learning_curve_{best_name_lc}.pdf"
    )
    plt.close()

    # Best model
    best_name = perf_df.loc[perf_df["r2_mean"].idxmax(), "Model"]
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}")
    print("=" * 70)

    save_yy_plot(
        y_true=y,
        y_pred=oof_store[best_name],
        out_pdf=RESULTS_DIR
        / f"01_YY_{best_name}_nested_OOF.pdf",
    )

    # Retrain on all data
    print("\nRetraining best model on all data...")
    best_info = m_dict[best_name]
    final_pipe = make_pipeline(best_info)
    final_grid = GridSearchCV(
        final_pipe,
        {f"model__{k}": v for k, v in best_info["params"].items()},
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
        n_jobs=-1,
    ).fit(X, y)
    final_best_pipe = final_grid.best_estimator_

    model_save_dir = RESULTS_DIR / "model_artifacts"
    model_save_dir.mkdir(exist_ok=True)
    model_path = model_save_dir / f"final_best_pipe_{best_name}.joblib"
    dump(final_best_pipe, model_path)
    print(f"   Model saved: {model_path}")

    print("\n" + "=" * 70)
    print("INTERPRETATION ANALYSIS (ALE + BOOTSTRAP)")
    print("=" * 70)

    _ = run_comprehensive_interpretation(
        final_best_pipe,
        X,
        y,
        best_info,
        RESULTS_DIR,
        ale_grid_1d=ALE_GRID_1D,
        ale_grid_2d=ALE_GRID_2D,
        do_2d=DO_2D,
        top_n_2D=TOP_N_2D,
        do_bootstrap=True,
        bootstrap_n=BOOTSTRAP_ITERATIONS,
    )

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS ON FINAL MODEL")
    print("=" * 70)
    _compute_shap_any_model(final_best_pipe, X, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Output Files:")
    print(
        f"   01_YY_{best_name}_nested_OOF.pdf              - OOF predictions"
    )
    print(
        "   02_SHAP_importance_ranking.csv                - SHAP importance"
    )
    print("   05a_SHAP_importance_bar.pdf                    - SHAP bar plot")
    print("   05b_SHAP_beeswarm.pdf                          - SHAP beeswarm")
    print(
        f"   06_learning_curve_{best_name}.pdf             - Learning curve"
    )
    print(
        "   07_ALE_2D_interaction_with_CI.pdf             - Interaction with CI"
    )
    print(
        "   08_ALE_2D_bootstrap_CI.csv                    - Bootstrap CIs"
    )
    print("   1D_ALE/                                        - 1D ALE plots")
    print("   2D_ALE/                                        - 2D ALE plots")
    print("\n" + "=" * 70 + "\n")
