import argparse
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression

from hrt import HRTRegressor

# Optional baselines (extras)
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

M5PY = None
try:
    import m5py as _m5
    for _name in ("M5Prime", "M5P", "M5Rules", "M5", "M5Regressor", "M5PY"):
        if hasattr(_m5, _name):
            M5PY = getattr(_m5, _name)
            break
except Exception:
    M5PY = None

try:
    from lineartree import LinearTreeRegressor
except Exception:
    LinearTreeRegressor = None


# --- m5 wrapper (keeps your behavior) ---
if M5PY is not None:
    from sklearn.base import BaseEstimator, RegressorMixin

    class M5PYWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, M=4.0):
            self.M = M
            self.model = M5PY()
            self.n_leaves_ = None

        def fit(self, X, y):
            m_val = int(self.M) if self.M is not None else 4
            try:
                self.model.set_params(min_samples_leaf=m_val)
            except Exception:
                try:
                    self.model.set_params(M=m_val)
                except Exception:
                    pass
            self.model.fit(X, y)

            # best-effort to parse leaves/rules count
            try:
                if hasattr(self.model, "n_leaves_"):
                    self.n_leaves_ = int(self.model.n_leaves_)
                elif hasattr(self.model, "model"):
                    match = re.search(r"Number of Rules\s*:\s*(\d+)", self.model.model)
                    self.n_leaves_ = int(match.group(1)) if match else None
            except Exception:
                self.n_leaves_ = None
            return self

        def predict(self, X):
            return self.model.predict(X)

        def get_n_leaves(self):
            return self.n_leaves_


def export_summary_to_pdf(df, filename="results.pdf", title=None):
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    display_df = df.copy()

    column_order = [
        "Model",
        "RMSE (Mean ± Std)",
        "MAE (Mean ± Std)",
        "R² (Mean ± Std)",
        "Segments (K) (Mean ± Std)",
        "Fit Time (s)",
    ]
    for col in column_order:
        if col not in display_df.columns:
            display_df[col] = "N/A"
    display_df = display_df[column_order]

    def wrap_cell(x, width=40):
        return "\n".join(textwrap.wrap(str(x), width=width))

    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda v: wrap_cell(v, 40))

    n_rows = len(display_df)
    fig_height = 1.0 + 0.5 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PDF to: {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="kin8nm.data")
    ap.add_argument("--test-size", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pdf", type=str, default="results_kin.pdf")
    args = ap.parse_args()

    # load
    df = pd.read_csv(args.data, sep=",", header=None, skipinitialspace=True)
    y = df.iloc[:, -1].values
    X_raw = df.iloc[:, :-1]
    numerical_features = X_raw.columns.tolist()
    categorical_features = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # ---------- paper-mode: tune once ----------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_raw, y, test_size=args.test_size, random_state=args.seed
    )

    models_to_tune = {
        "HRT": {
            "estimator": HRTRegressor(),
            "params": {"threshold": [0], "max_depth": [6], "step_size": ["auto"], "ridge_alpha": [1]},
        },
        "Ridge": {
            "estimator": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        },
        "CART": {
            "estimator": DecisionTreeRegressor(random_state=args.seed),
            "params": {"max_depth": [3, 5, 7, 9, 11], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
        },
    }

    if XGBRegressor is not None:
        models_to_tune["XGBoost"] = {
            "estimator": XGBRegressor(random_state=args.seed, objective="reg:squarederror"),
            "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7], "subsample": [0.7, 1.0]},
        }

    if M5PY is not None:
        models_to_tune["M5P"] = {
            "estimator": M5PYWrapper(),
            "params": {"M": [4.0, 10.0, 20.0, 40.0]},
        }

    if LinearTreeRegressor is not None:
        models_to_tune["LinearTree"] = {
            "estimator": LinearTreeRegressor(base_estimator=LinearRegression()),
            "params": {"max_depth": [3, 5, 7, 9, 11], "min_samples_split": [10, 20, 40], "min_samples_leaf": [10, 20, 40]},
        }

    best_params = {}
    print("\n--- Paper-mode hyperparameter tuning (one split, cv=5) ---")
    for name, info in models_to_tune.items():
        print(f"Tuning {name}...")
        t0 = time.time()
        pipe = Pipeline([("preprocessor", preprocessor), ("regressor", info["estimator"])])
        grid_params = {f"regressor__{k}": v for k, v in info["params"].items()}
        gs = GridSearchCV(pipe, grid_params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, error_score="raise")
        gs.fit(X_train_full, y_train_full)
        best_params[name] = {k.replace("regressor__", ""): v for k, v in gs.best_params_.items()}
        print(f"  done in {time.time()-t0:.2f}s, best score={gs.best_score_:.4f}, params={best_params[name]}")

    # ---------- repetitions with fixed best params ----------
    metrics = {name: {"RMSE": [], "MAE": [], "R2": [], "Segments": [], "FitTime": []} for name in models_to_tune.keys()}
    print(f"\n--- Repetitions: {args.reps} (fixed tuned params) ---")

    for i in range(args.reps):
        rs = args.seed + i
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_raw, y, test_size=args.test_size, random_state=rs)

        for name, info in models_to_tune.items():
            params = dict(best_params[name])
            base_cls = info["estimator"].__class__

            # propagate random_state if supported
            try:
                if "random_state" in base_cls().get_params():
                    params["random_state"] = rs
            except Exception:
                pass
            if name == "HRT":
                params["random_state"] = rs

            if name == "LinearTree":
                reg = LinearTreeRegressor(base_estimator=LinearRegression(), **params)
            else:
                reg = base_cls(**params)

            pipe = Pipeline([("preprocessor", preprocessor), ("regressor", reg)])

            t0 = time.time()
            pipe.fit(X_train_i, y_train_i)
            fit_time = time.time() - t0

            pred = pipe.predict(X_test_i)
            metrics[name]["FitTime"].append(fit_time)
            metrics[name]["RMSE"].append(float(np.sqrt(mean_squared_error(y_test_i, pred))))
            metrics[name]["MAE"].append(float(mean_absolute_error(y_test_i, pred)))
            metrics[name]["R2"].append(float(r2_score(y_test_i, pred)))

            reg_instance = pipe.named_steps["regressor"]
            if name == "HRT":
                try:
                    metrics[name]["Segments"].append(float(reg_instance.root_node.count_leaves()) if reg_instance.root_node else 0.0)
                except Exception:
                    metrics[name]["Segments"].append(np.nan)
            elif hasattr(reg_instance, "get_n_leaves"):
                try:
                    k = reg_instance.get_n_leaves()
                    metrics[name]["Segments"].append(float(k) if k is not None else np.nan)
                except Exception:
                    metrics[name]["Segments"].append(np.nan)
            else:
                metrics[name]["Segments"].append(np.nan)

    # ---------- summary ----------
    rows = []
    for name in models_to_tune.keys():
        m = metrics[name]
        seg = np.array(m["Segments"], dtype=float)
        rows.append({
            "Model": name,
            "RMSE (Mean ± Std)": f"{np.mean(m['RMSE']):.3f} ± {np.std(m['RMSE']):.3f}",
            "MAE (Mean ± Std)": f"{np.mean(m['MAE']):.3f} ± {np.std(m['MAE']):.3f}",
            "R² (Mean ± Std)": f"{np.mean(m['R2']):.3f} ± {np.std(m['R2']):.3f}",
            "Segments (K) (Mean ± Std)": (f"{np.nanmean(seg):.1f} ± {np.nanstd(seg):.1f}" if np.isfinite(seg).any() else "N/A"),
            "Fit Time (s)": f"{np.mean(m['FitTime']):.3f}",
        })
    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))

    export_summary_to_pdf(
        out,
        filename=args.pdf,
        title=f"Algorithm Performance Comparison (Mean from {args.reps} Repetitions)",
    )


if __name__ == "__main__":
    main()
