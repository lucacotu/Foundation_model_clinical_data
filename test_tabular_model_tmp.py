import sys
import os
import argparse
import json
from collections import defaultdict
from joblib import dump, load
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
import optuna
import shap
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import KFold
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sksurv.ensemble import RandomSurvivalForest

from src.data_loader import load_data
from src.preprocessing import clean_and_impute, prepare_cox_data_cv, prepare_cox_data_hurrah_cv
from src.tabpfn import get_tabpfn_embeddings
from src.tabdpt import get_tabdpt_embeddings
from src.tabicl import get_tabicl_embeddings


DATASETS = [
    ("OrmoniTirodei", "Total mortality", "Follow Up Data"),
    ("HURRAH",        "STATO_AL_FU",     "FU"),
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def get_ckpt_dir(dataset_name: str, preprocess_type: str, seed: int, fold: int, model: str) -> Path:
    p = Path("checkpoints") / model / dataset_name / preprocess_type / f"seed{seed}" / f"fold{fold}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_params(ckpt_dir: Path, model_name: str, params: dict):
    path = ckpt_dir / f"{model_name}_params.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  [ckpt] Params saved → {path}")


def load_params(ckpt_dir: Path, model_name: str) -> dict | None:
    path = ckpt_dir / f"{model_name}_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def ckpt_exists(ckpt_dir: Path, model_name: str) -> bool:
    return (ckpt_dir / f"{model_name}.pt").exists() or (ckpt_dir / f"{model_name}.pkl").exists()


def save_net_state(net: nn.Module, path: Path, *, baseline_hazards=None, baseline_cumulative_hazards=None, params: dict | None = None):
    torch.save({
        "net_state": net.state_dict(),
        "baseline_hazards": baseline_hazards,
        "baseline_cumulative_hazards": baseline_cumulative_hazards,
        "params": params or {},
    }, str(path))


def load_net_state(net: nn.Module, path: Path, device=None, model=None):
    checkpoint = torch.load(str(path), map_location=device or "cpu", weights_only=False)
    net.load_state_dict(checkpoint["net_state"])
    if model is not None:
        if checkpoint.get("baseline_hazards") is not None:
            model.baseline_hazards_ = checkpoint["baseline_hazards"]
        if checkpoint.get("baseline_cumulative_hazards") is not None:
            model.baseline_cumulative_hazards_ = checkpoint["baseline_cumulative_hazards"]
    if device is not None:
        net.to(device)
    return checkpoint.get("params", {})


# ── Survival analysis helpers ─────────────────────────────────────────────────

def make_structured_y(events, times):
    return np.array(
        [(bool(e), float(t)) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )


def rsf_surv_to_df(surv_fns) -> pd.DataFrame:
    time_points = surv_fns[0].x
    matrix = np.vstack([fn(time_points) for fn in surv_fns]).T
    return pd.DataFrame(matrix, index=time_points)


def concordance_td(surv_df: pd.DataFrame, t, y) -> float:
    return EvalSurv(surv_df, t, np.asarray(y), censor_surv="km").concordance_td()


def make_early_stopping(ckpt_path: Path, patience: int = 20, min_delta: float = 1e-4):
    tmp = ckpt_path.with_name(ckpt_path.stem + "_tmp" + ckpt_path.suffix)
    return tt.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        checkpoint_model=True,
        file_path=str(tmp),
        load_best=True,
    )


def load_or_fit_deepsurv(
    model, ckpt_path: Path,
    train_data, t_train, y_train,
    eval_data,  t_eval,  y_eval,
    lr: float = 0.01, batch_size: int = 256, epochs: int = 100,
    label: str = "",
):
    if ckpt_path.exists():
        print(f"  [ckpt] Loading {label} from {ckpt_path}")
        load_net_state(model.net, ckpt_path, device=model.device, model=model)
        return
    cb = make_early_stopping(ckpt_path)
    model.fit(
        train_data, (t_train, np.asarray(y_train)),
        batch_size, epochs, [cb],
        val_data=(eval_data, (t_eval, np.asarray(y_eval))),
        verbose=True,
    )
    model.compute_baseline_hazards()
    save_net_state(
        model.net, ckpt_path,
        baseline_hazards=model.baseline_hazards_,
        baseline_cumulative_hazards=model.baseline_cumulative_hazards_,
    )


def load_or_fit_rsf(ckpt_path: Path, rsf, train_data, train_structured, label: str = "RSF"):
    if ckpt_path.exists():
        print(f"  [ckpt] Loading {label} from {ckpt_path}")
        return load(ckpt_path)
    rsf.fit(train_data, train_structured)
    dump(rsf, ckpt_path)
    print(f"  [ckpt] {label} saved → {ckpt_path}")
    return rsf


def load_or_fit_cox(ckpt_path: Path, df_fit: pd.DataFrame, label: str = "Cox") -> CoxPHFitter:
    if ckpt_path.exists():
        print(f"  [ckpt] Loading {label} from {ckpt_path}")
        return load(ckpt_path)
    print(f"  Fitting {label}...")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_fit, duration_col="__t__", event_col="__e__")
    dump(cph, ckpt_path)
    return cph


# ── Embedding generation ──────────────────────────────────────────────────────

def get_embeddings(model_name: str, X_train, y_train, X_query, y_query, seed: int, device: str):
    if model_name == "tabpfn":
        return get_tabpfn_embeddings(X_train, y_train, X_query, y_query, seed)
    if model_name == "tabicl":
        y_tr = y_train.values if hasattr(y_train, "values") else y_train
        return get_tabicl_embeddings(X_train, y_tr, X_query, device=device, random_state=seed)
    if model_name == "tabdpt":
        return get_tabdpt_embeddings(X_train, y_train, X_query, device=device)
    raise ValueError(f"Unknown model: {model_name}")


# ── Fold management ───────────────────────────────────────────────────────────

def get_or_create_folds(X, dataset_name="dataset", seed=42, n_splits=5, base_path="tmp/splits/"):
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{dataset_name}_seed{seed}.pkl")

    if os.path.exists(file_path):
        print(f"Loading existing folds from {file_path}")
        return load(file_path)

    print(f"Creating new folds and saving to {file_path}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(kf.split(X))
    data = {"seed": seed, "n_splits": n_splits, "folds": folds}
    dump(data, file_path)
    return data


# ── Printing helpers ──────────────────────────────────────────────────────────

def print_stats(label, values):
    arr = np.array(values)
    print(f"  {label:10s} → mean: {arr.mean():.6f} | std: {arr.std():.6f} | min: {arr.min():.6f} | max: {arr.max():.6f}")


class Tee:
    """Writes simultaneously to console and file."""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, "w")

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.console
        self.file.close()

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


# ── KM curve plotting ─────────────────────────────────────────────────────────

def plot_km_curves(fold_data, dataset_name, model_name, preprocess_type, seed,
                   age_threshold=60, pdf_pages=None):
    t_all   = np.concatenate([d["t_test"] for d in fold_data])
    y_all   = np.concatenate([d["y_test"] for d in fold_data])
    has_age = fold_data[0]["age_test"] is not None
    age_all = np.concatenate([d["age_test"] for d in fold_data]) if has_age else None

    t_max        = float(t_all.max())
    common_times = np.linspace(0, t_max, 300)

    # Interpolate each patient's predicted survival onto common time grid
    all_surv = []
    for d in fold_data:
        surv_df = d["surv_df"]
        idx     = surv_df.index.values.astype(float)
        for col in surv_df.columns:
            vals = np.interp(common_times, idx, surv_df[col].values.astype(float))
            vals = np.minimum.accumulate(vals)
            all_surv.append(vals)
    all_surv = np.array(all_surv)  # (n_patients_total, 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Kaplan-Meier — {dataset_name} | {model_name.upper()} DeepSurv Tuned"
        f" | preprocess: {preprocess_type} | seed: {seed}",
        fontsize=11,
    )

    # ── Subplot 1: Overall ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Overall")
    kmf = KaplanMeierFitter()
    kmf.fit(t_all, event_observed=y_all, label="KM observed")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="steelblue")
    mean_pred = all_surv.mean(axis=0)
    ax.step(common_times, mean_pred, where="post",
            label="DeepSurv Tuned (mean predicted)", color="darkorange", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Subplot 2: Age-stratified ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title(f"Age-stratified (< {age_threshold} vs ≥ {age_threshold})")
    if has_age and age_all is not None:
        groups = [
            (age_all < age_threshold,  "steelblue", f"age < {age_threshold}"),
            (age_all >= age_threshold, "firebrick",  f"age ≥ {age_threshold}"),
        ]
        for mask, color, lbl in groups:
            if mask.sum() == 0:
                continue
            kmf_g = KaplanMeierFitter()
            kmf_g.fit(t_all[mask], event_observed=y_all[mask], label=f"KM {lbl}")
            kmf_g.plot_survival_function(ax=ax2, ci_show=True, color=color)
            mean_g = all_surv[mask].mean(axis=0)
            ax2.step(common_times, mean_g, where="post",
                     label=f"Predicted {lbl}", color=color, linewidth=2, linestyle="--")
    else:
        ax2.text(0.5, 0.5, "Age data not available",
                 ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Survival probability")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if pdf_pages is not None:
        print(f"[PDF] Saving page: {dataset_name} | {model_name} | preprocess: {preprocess_type} | seed: {seed}")
        pdf_pages.savefig(fig)
        print(f"[PDF] Page saved successfully")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dataset_name, feature_event, feature_time, seed, model, tuning=False, compute_shap=False, pdf_pages=None):
    match dataset_name:
        case "OrmoniTirodei":
            data = load_data(dataset_name, "Dataset Sirbu")
            df = clean_and_impute(dataset_name, data)
            df_train, df_eval = prepare_cox_data_cv(df)
        case "HURRAH":
            data = load_data(dataset_name, "Dataset Sirbu")
            df = clean_and_impute(dataset_name, data)
            df_train, df_eval = prepare_cox_data_hurrah_cv(df)
        case _:
            print("Name_dataset not defined")
            return []

    X_eval = df_eval.drop(columns=[feature_time, feature_event])
    y_eval = df_eval[feature_event]
    t_eval = df_eval[feature_time].values.astype(np.float32)

    X = df_train.drop(columns=[feature_time, feature_event])
    y = df_train[feature_event]
    t = df_train[feature_time].values.astype(np.float32)

    df_tmp      = X.copy()
    df_tmp["__event__"]      = y.values
    df_tmp["__duration__"]   = t
    df_tmp_eval = X_eval.copy()
    df_tmp_eval["__event__"]    = y_eval.values
    df_tmp_eval["__duration__"] = t_eval

    df_train_orig = df_tmp.copy()
    df_eval_orig  = df_tmp_eval.copy()

    results = []

    for preprocess_type in ["-1", "NaN"]:
        df_tmp      = df_train_orig.copy()
        df_tmp_eval = df_eval_orig.copy()

        if preprocess_type == "-1":
            print("\n\nPreprocess: Replace NaN with -1")
            df_tmp      = df_tmp.fillna(-1)
            df_tmp_eval = df_tmp_eval.fillna(-1)
        else:
            print("\n\nPreprocess: Keep NaN")
            df_tmp      = df_tmp.fillna(np.nan)
            df_tmp_eval = df_tmp_eval.fillna(np.nan)

        df_tmp      = df_tmp.reset_index(drop=True)
        df_tmp_eval = df_tmp_eval.reset_index(drop=True)

        X      = df_tmp.drop(columns=["__event__", "__duration__"])
        y      = df_tmp["__event__"]
        t      = df_tmp["__duration__"].values.astype(np.float32)
        X_eval = df_tmp_eval.drop(columns=["__event__", "__duration__"])
        y_eval = df_tmp_eval["__event__"]
        t_eval = df_tmp_eval["__duration__"].values.astype(np.float32)

        feature_names_shap = list(X.columns)

        # Extract age column before potential numpy conversion (for KM stratification)
        _km_age_col = "Age" if dataset_name == "OrmoniTirodei" else "ETA"
        _km_ages = X[_km_age_col].values.copy() if _km_age_col in X.columns else None

        if model == "tabdpt":
            X      = X.to_numpy()
            y      = y.to_numpy()
            X_eval = X_eval.to_numpy()
            y_eval = y_eval.to_numpy()

        folds  = get_or_create_folds(X, dataset_name=dataset_name, seed=seed, n_splits=5)
        scores = defaultdict(list)
        km_fold_data = []

        if compute_shap:
            shap_fold_values = {k: [] for k in ["deepsurv_simple", "deepsurv_vanilla", "rsf", "cox"]}
            if tuning:
                shap_fold_values["deepsurv_tuned"] = []
            if preprocess_type != "NaN":
                shap_fold_values.update({
                    "deepsurv_simple_baseline": [],
                    "deepsurv_vanilla_baseline": [],
                    "rsf_baseline": [],
                    "cox_baseline": [],
                })

        for fold, (train_idx, test_idx) in enumerate(folds["folds"]):
            fold_n = fold + 1

            if model == "tabdpt":
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            else:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            t_train, t_test = t[train_idx], t[test_idx]

            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Generating {model.upper()} Embeddings...")
            train_emb, test_emb = get_embeddings(model, X_train, y_train, X_test, y_test, seed, device)
            _,         eval_emb = get_embeddings(model, X_train, y_train, X_eval, y_eval, seed, device)
            print(f"Successfully generated embeddings with shape: {train_emb.shape}")

            y_train_structured = make_structured_y(y_train, t_train)
            y_eval_structured  = make_structured_y(y_eval,  t_eval)
            y_test_structured  = make_structured_y(y_test,  t_test)

            ckpt_dir = get_ckpt_dir(dataset_name, preprocess_type, seed, fold_n, model)

            # ── Tuned models ──────────────────────────────────────────────────
            if tuning:
                deepsurv_params = load_params(ckpt_dir, f"{model}_deepsurv_tuned")
                if deepsurv_params is not None and ckpt_exists(ckpt_dir, f"{model}_deepsurv_tuned"):
                    print(f"  [ckpt] Loading {model.upper()} DeepSurv Tuned from {ckpt_dir}")
                    net = tt.practical.MLPVanilla(
                        in_features=deepsurv_params["embedding_dim"],
                        num_nodes=deepsurv_params["num_nodes"],
                        out_features=1,
                        batch_norm=deepsurv_params["batch_norm"],
                        dropout=deepsurv_params["dropout"],
                    )
                    deepsurv_tuned = CoxPH(net, tt.optim.Adam)
                    deepsurv_tuned.optimizer.set_lr(deepsurv_params["learning_rate"])
                    load_net_state(deepsurv_tuned.net, ckpt_dir / f"{model}_deepsurv_tuned.pt", device=deepsurv_tuned.device, model=deepsurv_tuned)
                else:
                    study = optuna.create_study(
                        study_name=f"deepsurv_tuning_{model}_{fold_n}_{preprocess_type}_seed{seed}",
                        direction="maximize",
                    )

                    def objective_deepsurv(trial):
                        num_nodes  = trial.suggest_categorical("num_nodes", [[32], [64], [128], [64, 32], [128, 64]])
                        dropout    = trial.suggest_float("dropout", 0.0, 0.3)
                        lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
                        _net = tt.practical.MLPVanilla(train_emb.shape[1], num_nodes, 1, batch_norm=batch_norm, dropout=dropout)
                        _m   = CoxPH(_net, tt.optim.Adam)
                        _m.optimizer.set_lr(lr)
                        _m.fit(train_emb, (t_train, np.asarray(y_train)), epochs=100, batch_size=256)
                        _m.compute_baseline_hazards()
                        surv = _m.predict_surv_df(eval_emb)
                        return EvalSurv(surv, t_eval, np.asarray(y_eval), censor_surv="km").concordance_td()

                    study.optimize(objective_deepsurv, n_trials=100, n_jobs=-1)
                    best_parameters = {"embedding_dim": train_emb.shape[1], **study.best_params}
                    print(f"Best parameters: {study.best_params}")
                    save_params(ckpt_dir, f"{model}_deepsurv_tuned", best_parameters)

                    net = tt.practical.MLPVanilla(
                        best_parameters["embedding_dim"], best_parameters["num_nodes"], 1,
                        batch_norm=best_parameters["batch_norm"], dropout=best_parameters["dropout"],
                    )
                    deepsurv_tuned = CoxPH(net, tt.optim.Adam)
                    deepsurv_tuned.optimizer.set_lr(best_parameters["learning_rate"])
                    tuned_ckpt = ckpt_dir / f"{model}_deepsurv_tuned.pt"
                    deepsurv_tuned.fit(
                        train_emb, (t_train, np.asarray(y_train)),
                        val_data=(eval_emb, (t_eval, np.asarray(y_eval))),
                        epochs=200, callbacks=[make_early_stopping(tuned_ckpt)], batch_size=128,
                    )
                    deepsurv_tuned.compute_baseline_hazards()
                    save_net_state(
                        deepsurv_tuned.net, tuned_ckpt,
                        baseline_hazards=deepsurv_tuned.baseline_hazards_,
                        baseline_cumulative_hazards=deepsurv_tuned.baseline_cumulative_hazards_,
                        params=best_parameters,
                    )
                    print(f"  [ckpt] DeepSurv Tuned saved → {tuned_ckpt}")

                _surv_df_tuned_test = deepsurv_tuned.predict_surv_df(test_emb)
                c_train = concordance_td(deepsurv_tuned.predict_surv_df(train_emb), t_train, y_train)
                c_test  = concordance_td(_surv_df_tuned_test, t_test, y_test)
                scores["train_tab_tuned_deepsurv"].append(c_train)
                scores["test_tab_tuned_deepsurv"].append(c_test)
                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")

                # Tuned RSF
                rsf_tuned_path   = ckpt_dir / f"{model}_rsf_tuned.pkl"
                rsf_tuned_params = load_params(ckpt_dir, f"{model}_rsf_tuned")

                if rsf_tuned_params is not None and rsf_tuned_path.exists():
                    print(f"  [ckpt] Loading {model.upper()} RSF Tuned from {ckpt_dir}")
                    rsf_tuned = load(rsf_tuned_path)
                else:
                    study_rsf = optuna.create_study(
                        study_name=f"rsf_tuning_{model}_{fold_n}_{preprocess_type}_seed{seed}",
                        direction="maximize",
                    )

                    def objective_rsf(trial):
                        print(f"[Trial {trial.number}] START")
                        params = dict(
                            n_estimators      = trial.suggest_int("n_estimators", 50, 300, step=50),
                            max_depth         = trial.suggest_int("max_depth", 3, 10),
                            min_samples_split = trial.suggest_int("min_samples_split", 5, 20),
                            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 3, 15),
                            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                        )
                        _rsf = RandomSurvivalForest(**params, n_jobs=-1, random_state=seed)
                        print(f"[Trial {trial.number}] fitting...")
                        _rsf.fit(train_emb, y_train_structured)
                        print(f"[Trial {trial.number}] done")
                        return _rsf.score(eval_emb, y_eval_structured)

                    study_rsf.optimize(objective_rsf, n_trials=100, n_jobs=-1)
                    best_rsf_params = {"n_jobs": -1, "random_state": seed, **study_rsf.best_params}
                    save_params(ckpt_dir, f"{model}_rsf_tuned", best_rsf_params)
                    rsf_tuned = load_or_fit_rsf(
                        rsf_tuned_path, RandomSurvivalForest(**best_rsf_params),
                        train_emb, y_train_structured,
                        label=f"{model.upper()} RSF Tuned",
                    )

                _surv_df_rsf_tuned_test = rsf_surv_to_df(rsf_tuned.predict_survival_function(test_emb))
                c_train = concordance_td(rsf_surv_to_df(rsf_tuned.predict_survival_function(train_emb)), t_train, y_train)
                c_test  = concordance_td(_surv_df_rsf_tuned_test,  t_test,  y_test)
                scores["train_tab_tuned_rsf"].append(c_train)
                scores["test_tab_tuned_rsf"].append(c_test)
                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")

            # ── DeepSurv Simple (embedding) ───────────────────────────────────
            simple_path     = ckpt_dir / f"{model}_deepsurv_simple.pt"
            deepsurv_simple = CoxPH(nn.Linear(train_emb.shape[1], 1), tt.optim.Adam)
            deepsurv_simple.optimizer.set_lr(0.01)
            load_or_fit_deepsurv(deepsurv_simple, simple_path, train_emb, t_train, y_train, eval_emb, t_eval, y_eval, label="DeepSurv Simple")

            _surv_df_simple_test = deepsurv_simple.predict_surv_df(test_emb)
            c_train = concordance_td(deepsurv_simple.predict_surv_df(train_emb), t_train, y_train)
            c_test  = concordance_td(_surv_df_simple_test,  t_test,  y_test)
            scores["train_tab_deepsurv_simple"].append(c_train)
            scores["test_tab_deepsurv_simple"].append(c_test)
            print(f"C-index TRAIN: {c_train:.4f}")
            print(f"C-index TEST:  {c_test:.4f}")

            # ── DeepSurv Vanilla (embedding) ──────────────────────────────────
            vanilla_path         = ckpt_dir / f"{model}_deepsurv_vanilla.pt"
            deepsurv_vanilla_emb = CoxPH(
                tt.practical.MLPVanilla(train_emb.shape[1], [32, 32], 1, batch_norm=True, dropout=0.1),
                tt.optim.Adam,
            )
            deepsurv_vanilla_emb.optimizer.set_lr(0.01)
            load_or_fit_deepsurv(deepsurv_vanilla_emb, vanilla_path, train_emb, t_train, y_train, eval_emb, t_eval, y_eval, label="DeepSurv Vanilla")

            _surv_df_vanilla_test = deepsurv_vanilla_emb.predict_surv_df(test_emb)
            c_train = concordance_td(deepsurv_vanilla_emb.predict_surv_df(train_emb), t_train, y_train)
            c_test  = concordance_td(_surv_df_vanilla_test,  t_test,  y_test)
            scores["train_tab_deepsurv_vanilla"].append(c_train)
            scores["test_tab_deepsurv_vanilla"].append(c_test)
            print(f"C-index TRAIN: {c_train:.4f}")
            print(f"C-index TEST:  {c_test:.4f}")

            # ── RSF (embedding) ───────────────────────────────────────────────
            rsf_emb_path = ckpt_dir / f"{model}_rsf.pkl"
            rsf_emb = load_or_fit_rsf(
                rsf_emb_path,
                RandomSurvivalForest(n_estimators=300, max_depth=10, min_samples_split=15, min_samples_leaf=10, max_features="log2", n_jobs=-1, random_state=seed),
                train_emb, y_train_structured,
                label=f"{model.upper()} RSF",
            )

            _surv_df_rsf_test = rsf_surv_to_df(rsf_emb.predict_survival_function(test_emb))
            c_train = concordance_td(rsf_surv_to_df(rsf_emb.predict_survival_function(train_emb)), t_train, y_train)
            c_test  = concordance_td(_surv_df_rsf_test,  t_test,  y_test)
            scores["train_tab_rsf"].append(c_train)
            scores["test_tab_rsf"].append(c_test)
            print(f"C-index train: {c_train:.4f}")
            print(f"C-index test:  {c_test:.4f}")

            # ── Cox (embedding) ───────────────────────────────────────────────
            cox_emb_path = ckpt_dir / f"{model}_tab_cox.pkl"
            df_fit = pd.DataFrame(train_emb)
            df_fit["__t__"] = t_train
            df_fit["__e__"] = np.asarray(y_train)
            cph_emb = load_or_fit_cox(cox_emb_path, df_fit, label=f"{model.upper()} Cox (embedding)")

            _surv_df_cox_test = cph_emb.predict_survival_function(pd.DataFrame(test_emb))
            c_train = concordance_td(cph_emb.predict_survival_function(pd.DataFrame(train_emb)), t_train, y_train)
            c_test  = concordance_td(_surv_df_cox_test,  t_test,  y_test)
            scores["train_tab_cox"].append(c_train)
            scores["test_tab_cox"].append(c_test)
            print(f"C-index train: {c_train:.4f}")
            print(f"C-index test:  {c_test:.4f}")

            # ── Capture SHAP refs before baseline models ──────────────────────
            if compute_shap:
                _shap_simple_emb  = deepsurv_simple
                _shap_vanilla_emb = deepsurv_vanilla_emb
                _shap_rsf_emb     = rsf_emb
                _shap_cph_emb     = cph_emb
                if tuning:
                    _shap_tuned_emb = deepsurv_tuned

            # ── Baseline models (raw features, skipped for NaN preprocess) ────
            if preprocess_type != "NaN":
                X_train_f32 = np.asarray(X_train, dtype=np.float32)
                X_test_f32  = np.asarray(X_test,  dtype=np.float32)
                X_eval_f32  = np.asarray(X_eval,  dtype=np.float32)

                # DeepSurv Vanilla baseline
                vanilla_base_path     = ckpt_dir / "deepsurv_vanilla_baseline.pt"
                deepsurv_vanilla_base = CoxPH(
                    tt.practical.MLPVanilla(X_train.shape[1], [32, 32], 1, batch_norm=True, dropout=0.1),
                    tt.optim.Adam,
                )
                deepsurv_vanilla_base.optimizer.set_lr(0.01)
                load_or_fit_deepsurv(deepsurv_vanilla_base, vanilla_base_path, X_train_f32, t_train, y_train, X_eval_f32, t_eval, y_eval, label="DeepSurv Vanilla baseline")

                _surv_df_vanilla_base_test = deepsurv_vanilla_base.predict_surv_df(X_test_f32)
                c_train = concordance_td(deepsurv_vanilla_base.predict_surv_df(X_train_f32), t_train, y_train)
                c_test  = concordance_td(_surv_df_vanilla_base_test,  t_test,  y_test)
                scores["train_deepsurv_vanilla"].append(c_train)
                scores["test_deepsurv_vanilla"].append(c_test)
                print(f"C-index TRAIN: {c_train:.4f}")
                print(f"C-index TEST:  {c_test:.4f}")

                # DeepSurv Simple baseline
                simple_base_path     = ckpt_dir / "deepsurv_simple_baseline.pt"
                deepsurv_simple_base = CoxPH(nn.Linear(X_train.shape[1], 1), tt.optim.Adam)
                deepsurv_simple_base.optimizer.set_lr(0.01)
                load_or_fit_deepsurv(deepsurv_simple_base, simple_base_path, X_train_f32, t_train, y_train, X_eval_f32, t_eval, y_eval, label="DeepSurv Simple baseline")

                _surv_df_simple_base_test = deepsurv_simple_base.predict_surv_df(X_test_f32)
                c_train = concordance_td(deepsurv_simple_base.predict_surv_df(X_train_f32), t_train, y_train)
                c_test  = concordance_td(_surv_df_simple_base_test,  t_test,  y_test)
                scores["train_deepsurv_simple"].append(c_train)
                scores["test_deepsurv_simple"].append(c_test)
                print(f"C-index TRAIN: {c_train:.4f}")
                print(f"C-index TEST:  {c_test:.4f}")

                # RSF baseline
                rsf_base_path = ckpt_dir / "rsf_baseline.pkl"
                rsf_base = load_or_fit_rsf(
                    rsf_base_path,
                    RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=seed),
                    np.asarray(X_train), y_train_structured,
                    label="RSF baseline",
                )

                _surv_df_rsf_base_test = rsf_surv_to_df(rsf_base.predict_survival_function(X_test))
                c_train = concordance_td(rsf_surv_to_df(rsf_base.predict_survival_function(X_train)), t_train, y_train)
                c_test  = concordance_td(_surv_df_rsf_base_test,  t_test,  y_test)
                scores["train_rsf"].append(c_train)
                scores["test_rsf"].append(c_test)
                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")

                # Cox baseline
                cox_base_path = ckpt_dir / f"{model}_cox.pkl"
                if model == "tabdpt":
                    df_fit_base = pd.DataFrame(np.asarray(X_train))
                else:
                    df_fit_base = X_train.copy()
                df_fit_base["__t__"] = t_train
                df_fit_base["__e__"] = np.asarray(y_train)
                cph_base = load_or_fit_cox(cox_base_path, df_fit_base, label="Cox baseline")

                if model == "tabdpt":
                    surv_train_base = cph_base.predict_survival_function(pd.DataFrame(np.asarray(X_train)))
                    surv_test_base  = cph_base.predict_survival_function(pd.DataFrame(np.asarray(X_test)))
                else:
                    surv_train_base = cph_base.predict_survival_function(X_train)
                    surv_test_base  = cph_base.predict_survival_function(X_test)

                c_train = concordance_td(surv_train_base, t_train, y_train)
                c_test  = concordance_td(surv_test_base,  t_test,  y_test)
                scores["train_cox"].append(c_train)
                scores["test_cox"].append(c_test)
                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")

            # ── SHAP computation ──────────────────────────────────────────────
            if compute_shap:
                shap_cache_path = ckpt_dir / "shap_values.pkl"
                if shap_cache_path.exists():
                    print(f"  [SHAP] Fold {fold_n}: loading cached values from {shap_cache_path}")
                    cached = load(shap_cache_path)
                    for key, val in cached.items():
                        if key in shap_fold_values:
                            shap_fold_values[key].append(val)
                else:
                    print(f"\n[SHAP] Fold {fold_n}: computing SHAP values...")
                    if model == "tabdpt":
                        X_train_shap = pd.DataFrame(X_train, columns=feature_names_shap)
                        X_test_shap  = pd.DataFrame(X_test,  columns=feature_names_shap)
                    else:
                        X_train_shap = X_train.reset_index(drop=True)
                        X_test_shap  = X_test.reset_index(drop=True)

                    background  = shap.sample(X_train_shap, min(100, len(X_train_shap)))
                    X_shap_test = X_test_shap.iloc[:min(100, len(X_test_shap))]

                    def _embed_fn(X_np):
                        X_np    = np.atleast_2d(X_np)
                        X_df_in = pd.DataFrame(X_np, columns=feature_names_shap)
                        if model == "tabpfn":
                            _, emb = get_tabpfn_embeddings(X_train_shap, y_train, X_df_in, np.zeros(len(X_df_in)), seed)
                        elif model == "tabicl":
                            y_tr = y_train.values if hasattr(y_train, "values") else y_train
                            _, emb = get_tabicl_embeddings(X_train_shap, y_tr, X_df_in, device=device, random_state=seed)
                        elif model == "tabdpt":
                            y_tr = y_train if isinstance(y_train, np.ndarray) else y_train.to_numpy()
                            _, emb = get_tabdpt_embeddings(X_train_shap.to_numpy(), y_tr, X_np, device=device)
                        return np.atleast_2d(emb)

                    def _emb_net_risk(net, X_np):
                        emb = _embed_fn(X_np)
                        net.eval()
                        with torch.no_grad():
                            dev    = next(net.parameters()).device
                            tensor = torch.FloatTensor(emb.astype(np.float32)).to(dev)
                            if tensor.dim() == 1:
                                tensor = tensor.unsqueeze(0)
                            return net(tensor).cpu().numpy().flatten()

                    def _raw_net_risk(net, X_np):
                        net.eval()
                        with torch.no_grad():
                            dev    = next(net.parameters()).device
                            tensor = torch.FloatTensor(np.asarray(X_np, dtype=np.float32)).to(dev)
                            if tensor.dim() == 1:
                                tensor = tensor.unsqueeze(0)
                            return net(tensor).cpu().numpy().flatten()

                    for key, net_ref in [("deepsurv_simple", _shap_simple_emb.net), ("deepsurv_vanilla", _shap_vanilla_emb.net)]:
                        def _fn(X_np, _n=net_ref):
                            return _emb_net_risk(_n, X_np)
                        print(f"  [SHAP] {key}...")
                        sv = shap.KernelExplainer(_fn, background).shap_values(X_shap_test, nsamples=100)
                        shap_fold_values[key].append(np.abs(sv).mean(axis=0))

                    if tuning:
                        def _fn_tuned(X_np, _n=_shap_tuned_emb.net):
                            return _emb_net_risk(_n, X_np)
                        print("  [SHAP] deepsurv_tuned...")
                        sv = shap.KernelExplainer(_fn_tuned, background).shap_values(X_shap_test, nsamples=100)
                        shap_fold_values["deepsurv_tuned"].append(np.abs(sv).mean(axis=0))

                    print("  [SHAP] rsf (embedding)...")
                    sv = shap.KernelExplainer(
                        lambda X, _r=_shap_rsf_emb: _r.predict(_embed_fn(X)), background
                    ).shap_values(X_shap_test, nsamples=100)
                    shap_fold_values["rsf"].append(np.abs(sv).mean(axis=0))

                    print("  [SHAP] cox (embedding)...")
                    sv = shap.KernelExplainer(
                        lambda X, _c=_shap_cph_emb: _c.predict_partial_hazard(pd.DataFrame(_embed_fn(X))).values,
                        background,
                    ).shap_values(X_shap_test, nsamples=100)
                    shap_fold_values["cox"].append(np.abs(sv).mean(axis=0))

                    if preprocess_type != "NaN":
                        for key, net_ref in [("deepsurv_vanilla_baseline", deepsurv_vanilla_base.net),
                                             ("deepsurv_simple_baseline",  deepsurv_simple_base.net)]:
                            def _fn_base(X_np, _n=net_ref):
                                return _raw_net_risk(_n, X_np)
                            print(f"  [SHAP] {key}...")
                            sv = shap.KernelExplainer(_fn_base, background).shap_values(X_shap_test, nsamples=100)
                            shap_fold_values[key].append(np.abs(sv).mean(axis=0))

                        print("  [SHAP] rsf_baseline...")
                        sv = shap.KernelExplainer(
                            lambda X, _r=rsf_base: _r.predict(np.atleast_2d(np.asarray(X))), background
                        ).shap_values(X_shap_test, nsamples=100)
                        shap_fold_values["rsf_baseline"].append(np.abs(sv).mean(axis=0))

                        if model == "tabdpt":
                            def _cph_base_fn(X_np, _c=cph_base):
                                return _c.predict_partial_hazard(pd.DataFrame(np.atleast_2d(np.asarray(X_np)))).values
                        else:
                            def _cph_base_fn(X_np, _c=cph_base):
                                return _c.predict_partial_hazard(pd.DataFrame(np.atleast_2d(np.asarray(X_np, dtype=float)), columns=feature_names_shap)).values
                        print("  [SHAP] cox_baseline...")
                        sv = shap.KernelExplainer(_cph_base_fn, background).shap_values(X_shap_test, nsamples=100)
                        shap_fold_values["cox_baseline"].append(np.abs(sv).mean(axis=0))

                    fold_shap = {key: shap_fold_values[key][-1] for key in shap_fold_values if shap_fold_values[key]}
                    dump(fold_shap, shap_cache_path)
                    print(f"  [SHAP] Fold {fold_n}: values cached → {shap_cache_path}")

            # ── Collect survival curves for KM plotting ───────────────────────
            _fold_models = {
                "deepsurv_simple":  _surv_df_simple_test,
                "deepsurv_vanilla": _surv_df_vanilla_test,
                "rsf":              _surv_df_rsf_test,
                "cox":              _surv_df_cox_test,
            }
            if tuning:
                _fold_models["deepsurv_tuned"] = _surv_df_tuned_test
                _fold_models["rsf_tuned"]      = _surv_df_rsf_tuned_test
            if preprocess_type != "NaN":
                _fold_models.update({
                    "deepsurv_simple_baseline":  _surv_df_simple_base_test,
                    "deepsurv_vanilla_baseline": _surv_df_vanilla_base_test,
                    "rsf_baseline":              _surv_df_rsf_base_test,
                    "cox_baseline":              surv_test_base,
                })
            km_fold_data.append({
                "t_test":   t_test.copy(),
                "y_test":   np.asarray(y_test).copy(),
                "age_test": _km_ages[test_idx] if _km_ages is not None else None,
                "models":   _fold_models,
            })

            '''
            # 5. Visualize embeddings via t-SNE
            # Requires: from sklearn.manifold import TSNE; import matplotlib.pyplot as plt
            # Requires: from src.tabpfn import setup_figure, create_savefig_partial
            if train_emb.ndim == 3:
                train_emb = train_emb[0]

            print(f"Running t-SNE on {train_emb.shape} points...")
            tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
            X_2d = tsne.fit_transform(train_emb)

            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.sans-serif'] = ['Arial']

            fig, ax = setup_figure(figsize=(8, 8), style='seaborn-v0_8-paper')

            palette = ['#1f77b4', '#d62728']
            labels  = {0: 'Negative (Alive)', 1: 'Positive (Deceased)'}

            unique_classes = np.unique(y_test)
            for i, cls in enumerate(unique_classes):
                mask = (y_train.values == cls)
                ax.scatter(
                    X_2d[mask, 0],
                    X_2d[mask, 1],
                    label=labels.get(cls, str(cls)),
                    s=120,
                    color=palette[i % len(palette)],
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.1,
                )

            ax.set_xlabel("t-SNE Component 1", fontsize=24, fontweight='bold')
            ax.set_ylabel("t-SNE Component 2", fontsize=24, fontweight='bold')
            ax.legend(
                bbox_to_anchor=(1.0, 1.0),
                loc='upper right',
                title="Mortality Status",
                frameon=True,
                shadow=True,
                fontsize=10,
            )
            ax.grid(True, alpha=0.3, linestyle='--')

            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            save_name_base = f"tabpfn_mortality_tsne_dataset{dataset_name}_preprocess{preprocess_type}_seed{seed}_fold{fold_n}"
            savefig = create_savefig_partial(fig_dir=out_dir, fig_fmt='pdf', fig_size=(12, 12), save=True, dpi=300)
            savefig(fig, save_name_base)
            plt.close(fig)
            print(f"Visualization saved to {os.path.join(out_dir, save_name_base)}.pdf")
            '''

        # ── SHAP results summary (after fold loop) ────────────────────────────
        if compute_shap and shap_fold_values:
            shap_dir = Path("results") / dataset_name / model
            shap_dir.mkdir(parents=True, exist_ok=True)
            shap_path = shap_dir / f"shap_{dataset_name}_{model}_{preprocess_type}_seed{seed}.txt"
            with open(shap_path, "w") as sf:
                sf.write(f"SHAP — {dataset_name} | {model} | preprocess: {preprocess_type} | seed: {seed}\n")
                sf.write("=" * 70 + "\n")
                for mname, fv in shap_fold_values.items():
                    if not fv:
                        continue
                    arr    = np.stack(fv, axis=0)
                    mean_s = arr.mean(axis=0)
                    std_s  = arr.std(axis=0)
                    sf.write(f"\n--- {mname} ---\n")
                    sf.write(f"{'Feature':<35} {'Mean|SHAP|':>14} {'Std|SHAP|':>14}\n")
                    sf.write("-" * 65 + "\n")
                    for i in np.argsort(mean_s)[::-1]:
                        sf.write(f"{feature_names_shap[i]:<35} {mean_s[i]:>14.6f} {std_s[i]:>14.6f}\n")
            print(f"[SHAP] Results saved → {shap_path}")

        if km_fold_data and pdf_pages is not None:
            model_names = list(km_fold_data[0]["models"].keys())
            print(f"[PDF] Plotting KM curves for {len(model_names)} models: {model_names}")
            for m_name in model_names:
                per_model_data = [
                    {"t_test": d["t_test"], "y_test": d["y_test"], "age_test": d["age_test"],
                     "surv_df": d["models"][m_name]}
                    for d in km_fold_data
                ]
                plot_km_curves(
                    per_model_data,
                    dataset_name=dataset_name,
                    model_name=f"{model}/{m_name}",
                    preprocess_type=preprocess_type,
                    seed=seed,
                    age_threshold=60,
                    pdf_pages=pdf_pages,
                )

        results.append({"preprocess_type": preprocess_type, "scores": dict(scores)})

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--model",  type=str, default="tabpfn", choices=["tabpfn", "tabicl", "tabdpt"])
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--shap",   action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"STARTED  model={args.model}  seed={args.seed}  tuning={args.tuning}  shap={args.shap}")

    output_pdf_path = Path("results") / f"plot_curves_{args.model}_{args.seed}.pdf"
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[PDF] Opening PdfPages → {output_pdf_path}")
    km_pdf = PdfPages(output_pdf_path)

    try:
        for dataset_name, feature_event, feature_time in DATASETS:
            results = main(dataset_name, feature_event, feature_time, args.seed, args.model, args.tuning, args.shap, pdf_pages=km_pdf)

            output_dir = Path("results") / dataset_name / args.model
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"results_cv_{dataset_name}_{args.model}_seed{args.seed}.txt"

            with Tee(filepath):
                for result in results:
                    pt = result["preprocess_type"]
                    sc = result["scores"]
                    m  = args.model.upper()
                    print(f"\n  Preprocess: {pt} | Seed: {args.seed}")
                    if sc.get("train_tab_tuned_deepsurv"):
                        print_stats(f"{m} Train Tuned DeepSurv", sc["train_tab_tuned_deepsurv"])
                        print_stats(f"{m} Test Tuned DeepSurv",  sc["test_tab_tuned_deepsurv"])
                    if sc.get("train_tab_tuned_rsf"):
                        print_stats(f"{m} Train Tuned RSF",      sc["train_tab_tuned_rsf"])
                        print_stats(f"{m} Test Tuned RSF",       sc["test_tab_tuned_rsf"])
                    print_stats(f"{m} Train Vanilla",            sc["train_tab_deepsurv_vanilla"])
                    print_stats(f"{m} Test Vanilla",             sc["test_tab_deepsurv_vanilla"])
                    print_stats(f"{m} Train Simple",             sc["train_tab_deepsurv_simple"])
                    print_stats(f"{m} Test Simple",              sc["test_tab_deepsurv_simple"])
                    print_stats(f"{m} Train RSF",                sc["train_tab_rsf"])
                    print_stats(f"{m} Test RSF",                 sc["test_tab_rsf"])
                    print_stats(f"{m} Train Cox",                sc["train_tab_cox"])
                    print_stats(f"{m} Test Cox",                 sc["test_tab_cox"])
                    if sc.get("train_deepsurv_vanilla"):
                        print_stats("Train DeepSurv Vanilla",    sc["train_deepsurv_vanilla"])
                        print_stats("Test DeepSurv Vanilla",     sc["test_deepsurv_vanilla"])
                    if sc.get("train_deepsurv_simple"):
                        print_stats("Train DeepSurv Simple",     sc["train_deepsurv_simple"])
                        print_stats("Test DeepSurv Simple",      sc["test_deepsurv_simple"])
                    if sc.get("train_rsf"):
                        print_stats("Train RSF",                 sc["train_rsf"])
                        print_stats("Test RSF",                  sc["test_rsf"])
                    if sc.get("train_cox"):
                        print_stats("Train Cox",                 sc["train_cox"])
                        print_stats("Test Cox",                  sc["test_cox"])

            print(f"Result saved in '{filepath}'")

    finally:
        print(f"[PDF] Closing PdfPages → {output_pdf_path}")
        km_pdf.close()
        print(f"KM curves saved → {output_pdf_path}")
