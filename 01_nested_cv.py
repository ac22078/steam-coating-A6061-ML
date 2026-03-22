"""
01_nested_cv.py
Repeated Nested Cross-Validation (5-fold, 10 repeats) for model selection.
Saves the best model (Random Forest) trained on all data.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, DotProduct, ConstantKernel as C, WhiteKernel,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_CSV   = "A5.csv"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "model_artifacts").mkdir(exist_ok=True)

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
            "params": {"max_depth": [3, 4, 5], "min_samples_leaf": [3, 5], "max_features": ["sqrt", None]},
            "use_scaler": False,
        },
        "GBR": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {"loss": ["huber"], "alpha": [0.9], "learning_rate": [0.03, 0.1],
                       "n_estimators": [200, 500, 1000], "max_depth": [1, 2]},
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


def nested_cv_oof(X, y, model_info, outer_splits=5, outer_repeats=10, seed=42):
    outer_cv = RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2_list, rmse_list, mae_list, fold_records = [], [], [], []
    pred_sum = pd.Series(0.0, index=X.index)
    pred_cnt = pd.Series(0,   index=X.index, dtype=int)

    for fold_id, (tr, te) in enumerate(outer_cv.split(X), 1):
        pipe = make_pipeline(model_info)
        grid = GridSearchCV(pipe, {f"model__{k}": v for k, v in model_info["params"].items()},
                            cv=inner_cv, scoring="r2", n_jobs=-1)
        grid.fit(X.iloc[tr], y.iloc[tr])
        yp = grid.best_estimator_.predict(X.iloc[te])
        pred_sum.loc[X.iloc[te].index] += yp
        pred_cnt.loc[X.iloc[te].index] += 1
        yt = y.iloc[te]
        r2_list.append(r2_score(yt, yp))
        rmse_list.append(np.sqrt(mean_squared_error(yt, yp)))
        mae_list.append(mean_absolute_error(yt, yp))
        fold_records.append({"fold": fold_id, "n_train": len(tr), "n_test": len(te),
                              "r2": r2_list[-1], "rmse": rmse_list[-1], "mae": mae_list[-1]})

    return (
        {"r2_mean":   np.mean(r2_list),   "r2_std":   np.std(r2_list,   ddof=1),
         "rmse_mean": np.mean(rmse_list), "rmse_std": np.std(rmse_list, ddof=1),
         "mae_mean":  np.mean(mae_list),  "mae_std":  np.std(mae_list,  ddof=1)},
        pred_sum / pred_cnt,
        pd.DataFrame(fold_records),
    )


def save_parity_plot(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "b--", lw=0.8, zorder=1)
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", s=18,
               color="#e31a1c", linewidths=0.4, zorder=2)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.92, f"$R^2$: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_xlabel("Measured $E_\\mathrm{pit}$ / V")
    ax.set_ylabel("Predicted $E_\\mathrm{pit}$ / V")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("01 NESTED CV")
    print("="*60)

    X, y = load_data(TRAIN_CSV)
    print(f"Data: X={X.shape}, features={list(X.columns)}")

    m_dict = get_models()
    performance_records, oof_store, fold_all = [], {}, []

    for name, info in m_dict.items():
        print(f"  [{name}] ", end="", flush=True)
        metrics, oof_pred, fold_df = nested_cv_oof(X, y, info)
        performance_records.append({"Model": name, **metrics})
        oof_store[name] = oof_pred
        fold_df["Model"] = name
        fold_all.append(fold_df)
        print(f"R2={metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}")

    perf_df = pd.DataFrame(performance_records)
    perf_df.to_csv(RESULTS_DIR / "performance_summary.csv", index=False)
    pd.concat(fold_all, ignore_index=True).to_csv(
        RESULTS_DIR / "nestedcv_fold_metrics.csv", index=False)
    print("\n" + perf_df.to_string(index=False))

    best_name = perf_df.loc[perf_df["r2_mean"].idxmax(), "Model"]
    print(f"\nBest model: {best_name}")

    # Parity plot (OOF)
    oof_df = pd.DataFrame({"Sample_ID": y.index,
                            "Measured": y.values,
                            "Predicted_OOF": oof_store[best_name].values})
    oof_df.to_csv(RESULTS_DIR / "SourceData_Fig4_parity.csv", index=False)
    save_parity_plot(y, oof_store[best_name],
                     RESULTS_DIR / f"Fig4_parity_OOF_{best_name}.pdf")
    print("  ✓ Fig4_parity_OOF.pdf")

    # Retrain on all data
    print("\nRetraining on all data...")
    best_info = m_dict[best_name]
    final_grid = GridSearchCV(
        make_pipeline(best_info),
        {f"model__{k}": v for k, v in best_info["params"].items()},
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2", n_jobs=-1,
    ).fit(X, y)
    dump(final_grid.best_estimator_,
         RESULTS_DIR / "model_artifacts" / f"final_best_pipe_{best_name}.joblib")
    print(f"  ✓ Model saved: final_best_pipe_{best_name}.joblib")
    print(f"  Best params: {final_grid.best_params_}")
