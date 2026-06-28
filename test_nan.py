import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import random
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torch.nn as nn
from sksurv.ensemble import RandomSurvivalForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from lifelines import CoxPHFitter
import joblib
from joblib import dump, load
from sklearn.model_selection import KFold

from src.data_loader import load_data
from src.preprocessing import clean_and_impute, prepare_cox_data_cv, prepare_cox_data_hurrah_cv
from src.tabpfn import get_tabpfn_embeddings
from src.tabdpt import get_tabdpt_embeddings
from src.tabicl import get_tabicl_embeddings


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(dataset_name, feature_event, feature_time, seed, percentages, f_model):
    # FIX: reset seed at the start of each main call so results are independent
    # of how many times or in which order main() is called externally.
    set_seed(seed)

    prepare_fn = {
        "OrmoniTirodei": prepare_cox_data_cv,
        "HURRAH":        prepare_cox_data_hurrah_cv,
    }
    if dataset_name not in prepare_fn:
        print("Name_dataset not defined")
        return

    data = load_data(dataset_name, "Dataset Sirbu")
    df   = clean_and_impute(dataset_name, data)
    df_mortality_train, df_mortality_eval = prepare_fn[dataset_name](df)

    df_mortality_train = df_mortality_train.reset_index(drop=True)
    df_mortality_eval  = df_mortality_eval.reset_index(drop=True)

    X      = df_mortality_train.drop(columns=[feature_time, feature_event])
    y      = df_mortality_train[feature_event]
    t      = df_mortality_train[feature_time].values.astype(np.float32)
    X_eval = df_mortality_eval.drop(columns=[feature_time, feature_event])
    y_eval = df_mortality_eval[feature_event]
    t_eval = df_mortality_eval[feature_time].values.astype(np.float32)

    if f_model == "tabdpt":
        X      = X.to_numpy()
        y      = y.to_numpy()
        X_eval = X_eval.to_numpy()
        y_eval = y_eval.to_numpy()

    # Fill any pre-existing NaN with -1 before injecting synthetic NaN
    if isinstance(X, pd.DataFrame):
        X      = X.fillna(-1)
        X_eval = X_eval.fillna(-1)
    else:
        X[np.isnan(X)]           = -1
        X_eval[np.isnan(X_eval)] = -1

    folds = get_or_create_folds(X, dataset_name=dataset_name, seed=seed, n_splits=5,
                                 base_path="tmp/splits/")

    imputation_methods = ["embeddings", "mean", "median", "constant", "knn", "imputer_bayesian"]

    def empty_scores():
        return {m: [] for m in imputation_methods}

    # Results keyed by percentage → compatible with process_results after flattening
    all_results = {pct: {
        "train_rsf":              empty_scores(),
        "test_rsf":               empty_scores(),
        "train_deepsurv_simple":  empty_scores(),
        "test_deepsurv_simple":   empty_scores(),
        "train_deepsurv_vanilla": empty_scores(),
        "test_deepsurv_vanilla":  empty_scores(),
        "train_cox":              empty_scores(),
        "test_cox":               empty_scores(),
    } for pct in percentages}

    X_eval_orig = X_eval.copy()

    # Eval ordering is fold-independent (same validation set for every fold).
    # fold=0 is a sentinel that identifies the eval split in the filename.
    eval_ordering = get_or_create_nan_ordering(
        X_eval_orig, dataset_name, f_model, seed, fold=0, split_name="eval"
    )

    for fold, (train_idx, test_idx) in enumerate(folds["folds"]):
        if f_model == "tabdpt":
            X_train_clean = X[train_idx].copy()
            X_test_clean  = X[test_idx].copy()
            y_train       = y[train_idx]
            y_test        = y[test_idx]
        else:
            X_train_clean = X.iloc[train_idx].copy()
            X_test_clean  = X.iloc[test_idx].copy()
            y_train       = y.iloc[train_idx]
            y_test        = y.iloc[test_idx]
        t_train, t_test = t[train_idx], t[test_idx]

        # Generate NaN orderings ONCE per fold. Each split has its own
        # independent permutation so train/test/eval masks don't correlate.
        train_ordering = get_or_create_nan_ordering(
            X_train_clean, dataset_name, f_model, seed, fold + 1, "train"
        )
        test_ordering = get_or_create_nan_ordering(
            X_test_clean, dataset_name, f_model, seed, fold + 1, "test"
        )

        for percentage_nan in percentages:
            # CUMULATIVE MASKS: the 20% mask is a strict superset of the 10% mask
            # because both derive from the same ordering (first N indices).
            X_train = apply_mask_from_ordering(X_train_clean, train_ordering, percentage_nan)
            X_test  = apply_mask_from_ordering(X_test_clean,  test_ordering,  percentage_nan)
            X_eval  = apply_mask_from_ordering(X_eval_orig,   eval_ordering,  percentage_nan)

            actual_pct = (
                np.isnan(X_train) if isinstance(X_train, np.ndarray) else X_train.isna().values
            ).sum() / X_train.size * 100
            print(f"[fold={fold+1}, target={percentage_nan:.0%}] actual NaN%: {actual_pct:.1f}%")
            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            path_dir = get_ckpt_dir(dataset_name, seed, percentage_nan, fold + 1, f_model)
            device   = "cuda" if torch.cuda.is_available() else "cpu"

            set_seed(seed + fold)
            if f_model == "tabpfn":
                print("Generating TabPFN Embeddings...")
                train_embeddings, test_embeddings = get_tabpfn_embeddings(
                    X_train, y_train, X_test, y_test, seed)
                _, eval_embeddings = get_tabpfn_embeddings(
                    X_train, y_train, X_eval, y_eval, seed)
            elif f_model == "tabicl":
                print("Generating Tabicl Embeddings...")
                train_embeddings, test_embeddings = get_tabicl_embeddings(
                    X_train, y_train.values, X_test, device=device, random_state=seed)
                _, eval_embeddings = get_tabicl_embeddings(
                    X_train, y_train.values, X_eval, device=device, random_state=seed)
            elif f_model == "tabdpt":
                print("Generating TabDPT Embeddings...")
                train_embeddings, test_embeddings = get_tabdpt_embeddings(
                    X_train, y_train, X_test, device=device)
                _, eval_embeddings = get_tabdpt_embeddings(
                    X_train, y_train, X_eval, device=device)
            else:
                raise ValueError("Model not defined")
            print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")

            y_train_structured = make_structured_array(y_train, t_train)

            sets = [(train_embeddings, test_embeddings, eval_embeddings)]
            sets.append(load_or_fit_imputer(path_dir, "imputer_mean.pkl",
                SimpleImputer(strategy="mean"),
                X_train, X_test, X_eval, label="SimpleImputer (mean)"))
            sets.append(load_or_fit_imputer(path_dir, "imputer_median.pkl",
                SimpleImputer(strategy="median"),
                X_train, X_test, X_eval, label="SimpleImputer (median)"))
            sets.append(load_or_fit_imputer(path_dir, "imputer_constant.pkl",
                SimpleImputer(strategy="constant", fill_value=-1),
                X_train, X_test, X_eval, label="SimpleImputer (constant, fill=-1)"))
            sets.append(load_or_fit_imputer(path_dir, "imputer_knn.pkl",
                KNNImputer(n_neighbors=5, weights="uniform"),
                X_train, X_test, X_eval, label="KNNImputer"))
            sets.append(load_or_fit_imputer(path_dir, "imputer_mice.pkl",
                IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                                 random_state=seed, sample_posterior=False),
                X_train, X_test, X_eval, label="IterativeImputer (BayesianRidge, MICE-like)"))

            X_train_mean, _, X_eval_mean = sets[1]

            if ckpt_exists(path_dir, "fixed_epochs_deepsurv_simple.json"):
                print("Loading epochs value...")
                fixed_epochs_simple = load_fixed_epochs(path_dir, "deepsurv_simple")
            else:
                try:
                    set_seed(seed + fold + 1)
                    print("Fitting DeepSurv Simple pilot model...")
                    pilot    = create_deepsurv_simple(X_train_mean.shape[1])
                    callback = create_callbacks("pilot_simple")
                    pilot.fit(X_train_mean, (t_train, np.asarray(y_train)), 256, 100, callback,
                              val_data=(X_eval_mean, (t_eval, np.asarray(y_eval))), verbose=True)
                    log = pilot.log.to_pandas()
                    fixed_epochs_simple = len(log) - callback[0]._iter_since_best
                    save_fixed_epochs(path_dir, "deepsurv_simple", fixed_epochs_simple)
                except Exception as e:
                    print(f"    DeepSurv-simple pilot failed: {e}. Using default 50 epochs.")
                    fixed_epochs_simple = 50

            if ckpt_exists(path_dir, "fixed_epochs_deepsurv_vanilla.json"):
                print("Loading epochs value...")
                fixed_epochs_vanilla = load_fixed_epochs(path_dir, "deepsurv_vanilla")
            else:
                try:
                    set_seed(seed + fold + 2)
                    print("Fitting DeepSurv Vanilla pilot model...")
                    pilot    = create_deepsurv_vanilla(X_train_mean.shape[1])
                    callback = create_callbacks("pilot_vanilla")
                    pilot.fit(X_train_mean, (t_train, np.asarray(y_train)), 256, 100, callback,
                              val_data=(X_eval_mean, (t_eval, np.asarray(y_eval))), verbose=True)
                    log = pilot.log.to_pandas()
                    fixed_epochs_vanilla = len(log) - callback[0]._iter_since_best
                    save_fixed_epochs(path_dir, "deepsurv_vanilla", fixed_epochs_vanilla)
                except Exception as e:
                    print(f"    DeepSurv-vanilla pilot failed: {e}. Using default 50 epochs.")
                    fixed_epochs_vanilla = 50

            for method, (X_train_m, X_test_m, X_eval_m) in zip(imputation_methods, sets):
                method_dir = get_ckpt_dir(dataset_name, seed, percentage_nan, fold + 1,
                                          f_model, method)
                print(f"Evaluating method: {method}")

                # RSF
                set_seed(seed + fold + 3)
                try:
                    model_file = "rsf.pt"
                    if ckpt_exists(method_dir, model_file):
                        print("Loading Random Survival Forest...")
                        rsf = load(os.path.join(method_dir, model_file))
                    else:
                        print("Fitting Random Survival Forest model...")
                        rsf = create_rsf_model(seed)
                        rsf.fit(X_train_m, y_train_structured)
                        dump(rsf, os.path.join(method_dir, model_file))

                    surv_train = rsf_surv_to_df(rsf.predict_survival_function(X_train_m))
                    surv_test  = rsf_surv_to_df(rsf.predict_survival_function(X_test_m))
                    ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
                    ev_test  = EvalSurv(surv_test,  t_test,  np.asarray(y_test),  censor_surv='km')
                    c_train, c_test = ev_train.concordance_td(), ev_test.concordance_td()
                    print("C-index TRAIN:", c_train)
                    print("C-index TEST :", c_test)
                except Exception as e:
                    print(f"    RSF fit failed: {e}")
                    c_train, c_test = np.nan, np.nan
                all_results[percentage_nan]["train_rsf"][method].append(c_train)
                all_results[percentage_nan]["test_rsf"][method].append(c_test)

                # DeepSurv Simple
                set_seed(seed + fold + 4)
                try:
                    deepsurv_simple = create_deepsurv_simple(X_train_m.shape[1])
                    fit_or_load_deepsurv(deepsurv_simple, method_dir, "deepsurv_simple.pt",
                                         X_train_m, t_train, y_train, fixed_epochs_simple)
                    c_train, c_test = eval_surv_concordance(
                        deepsurv_simple, X_train_m, X_test_m, t_train, t_test, y_train, y_test)
                except Exception as e:
                    print(f"    DeepSurv-simple fit failed: {e}")
                    c_train, c_test = np.nan, np.nan
                all_results[percentage_nan]["train_deepsurv_simple"][method].append(c_train)
                all_results[percentage_nan]["test_deepsurv_simple"][method].append(c_test)

                # DeepSurv Vanilla
                set_seed(seed + fold + 5)
                try:
                    deepsurv_vanilla = create_deepsurv_vanilla(X_train_m.shape[1])
                    fit_or_load_deepsurv(deepsurv_vanilla, method_dir, "deepsurv_vanilla.pt",
                                         X_train_m, t_train, y_train, fixed_epochs_vanilla)
                    c_train, c_test = eval_surv_concordance(
                        deepsurv_vanilla, X_train_m, X_test_m, t_train, t_test, y_train, y_test)
                except Exception as e:
                    print(f"    DeepSurv-vanilla fit failed: {e}")
                    c_train, c_test = np.nan, np.nan
                all_results[percentage_nan]["train_deepsurv_vanilla"][method].append(c_train)
                all_results[percentage_nan]["test_deepsurv_vanilla"][method].append(c_test)

                # Cox
                set_seed(seed + fold + 6)
                try:
                    model_file = "cox.pt"
                    if ckpt_exists(method_dir, model_file):
                        print("Loading Cox model...")
                        cph = load(os.path.join(method_dir, model_file))
                    else:
                        print("Fitting Cox model...")
                        cph    = create_cox()
                        df_fit = pd.DataFrame(X_train_m)
                        df_fit['__t__'] = t_train
                        df_fit['__e__'] = np.asarray(y_train)
                        cph.fit(df_fit, duration_col='__t__', event_col='__e__')
                        dump(cph, os.path.join(method_dir, model_file))

                    surv_test  = cph.predict_survival_function(X_test_m)
                    surv_train = cph.predict_survival_function(X_train_m)
                    ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
                    ev_test  = EvalSurv(surv_test,  t_test,  np.asarray(y_test),  censor_surv='km')
                    c_train, c_test = ev_train.concordance_td(), ev_test.concordance_td()
                    print("C-index TRAIN:", c_train)
                    print("C-index TEST :", c_test)
                except Exception as e:
                    print(f"    Cox fit failed: {e}")
                    c_train, c_test = np.nan, np.nan
                all_results[percentage_nan]["train_cox"][method].append(c_train)
                all_results[percentage_nan]["test_cox"][method].append(c_test)

    # Return in the same format as process_results expects after flattening
    return {
        pct: (
            (all_results[pct]["train_rsf"],              all_results[pct]["test_rsf"]),
            (all_results[pct]["train_deepsurv_simple"],  all_results[pct]["test_deepsurv_simple"]),
            (all_results[pct]["train_deepsurv_vanilla"], all_results[pct]["test_deepsurv_vanilla"]),
            (all_results[pct]["train_cox"],              all_results[pct]["test_cox"]),
        )
        for pct in percentages
    }


# ── NaN mask (cumulative approach) ───────────────────────────────────────────

def get_or_create_nan_ordering(data, dataset_name, f_model, seed, fold, split_name,
                                base_path="tmp/masks/"):
    """
    Returns a 1-D array of flat indices: a full random permutation of all
    positions in `data`. The first int(size * p) entries define the cumulative
    NaN mask at percentage p, so masks at lower percentages are always strict
    subsets of masks at higher percentages.

    Uses np.random.default_rng (independent of the global numpy random state)
    so mask generation does not affect model-training reproducibility.
    The mask is deterministic given (seed, fold, split_name) and is saved to
    disk so it can be reloaded without recomputation.
    """
    os.makedirs(base_path, exist_ok=True)
    fname = f"{dataset_name}_{f_model}_seed{seed}_fold{fold}_{split_name}.npy"
    path  = os.path.join(base_path, fname)

    if os.path.exists(path):
        print(f"Loading NaN ordering from {path}")
        return np.load(path)

    print(f"Generating NaN ordering → {path}")
    split_id = {"train": 0, "test": 1, "eval": 2}[split_name]
    # Independent RNG: list seed makes each (seed, fold, split) combination unique
    rng      = np.random.default_rng([seed, fold, split_id])
    ordering = rng.permutation(data.size)
    np.save(path, ordering)
    return ordering


def apply_mask_from_ordering(data, ordering, target_percentage):
    """
    Apply a NaN mask at `target_percentage` using a pre-computed ordering.
    The first int(size * target_percentage) flat indices become NaN.
    Works for both pandas DataFrames and numpy arrays.
    """
    is_df     = isinstance(data, pd.DataFrame)
    data_copy = data.copy()

    n_nan = int(data_copy.size * target_percentage)
    if n_nan == 0:
        return data_copy

    flat_idx   = ordering[:n_nan]
    rows, cols = np.unravel_index(flat_idx, data_copy.shape)

    if is_df:
        # Convert to numpy for vectorised assignment, then rebuild DataFrame
        arr            = data_copy.to_numpy(dtype=float)
        arr[rows, cols] = np.nan
        data_copy       = pd.DataFrame(arr, index=data_copy.index, columns=data_copy.columns)
    else:
        data_copy[rows, cols] = np.nan

    return data_copy


# ── helpers ──────────────────────────────────────────────────────────────────

def load_or_fit_imputer(path_dir, name, imputer, X_train, X_test, X_eval, label=None):
    path = os.path.join(path_dir, name)
    desc = label or type(imputer).__name__
    if ckpt_exists(path_dir, name):
        print(f"Loading {desc}...")
        imputer     = joblib.load(path)
        X_train_out = imputer.transform(X_train).astype(np.float32)
    else:
        print(f"Fitting {desc}...")
        X_train_out = imputer.fit_transform(X_train).astype(np.float32)
        joblib.dump(imputer, path)
    return (
        X_train_out,
        imputer.transform(X_test).astype(np.float32),
        imputer.transform(X_eval).astype(np.float32),
    )


def make_structured_array(y, t):
    return np.array(
        [(bool(e), float(ti)) for e, ti in zip(y, t)],
        dtype=[('event', bool), ('time', float)]
    )


def rsf_surv_to_df(surv_fns):
    time_points = surv_fns[0].x
    surv_matrix = np.vstack([fn(time_points) for fn in surv_fns]).T
    return pd.DataFrame(surv_matrix, index=time_points)


def fit_or_load_deepsurv(model, path_dir, model_name, X_train, t_train, y_train, fixed_epochs):
    path = os.path.join(path_dir, model_name)
    if ckpt_exists(path_dir, model_name):
        print(f"Loading {model_name}...")
        load_net_state(model.net, path, device=model.device, model=model)
    else:
        print(f"Fitting {model_name}...")
        model.fit(X_train, (t_train, np.asarray(y_train)), 256, fixed_epochs, verbose=True)
        model.compute_baseline_hazards()
        save_net_state(model.net, path,
                       baseline_hazards=model.baseline_hazards_,
                       baseline_cumulative_hazards=model.baseline_cumulative_hazards_)


def eval_surv_concordance(model, X_train_m, X_test_m, t_train, t_test, y_train, y_test):
    surv_train = model.predict_surv_df(X_train_m)
    surv_test  = model.predict_surv_df(X_test_m)
    ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
    ev_test  = EvalSurv(surv_test,  t_test,  np.asarray(y_test),  censor_surv='km')
    c_train, c_test = ev_train.concordance_td(), ev_test.concordance_td()
    print("C-index TRAIN:", c_train)
    print("C-index TEST :", c_test)
    return c_train, c_test


# ── model / checkpoint utilities ─────────────────────────────────────────────

def get_ckpt_dir(dataset_name: str, seed: int, nan_percentage: float, fold: int,
                 model: str, method="") -> Path:
    parts = ["nan", model, dataset_name, f"nan_percentage{nan_percentage}",
             f"seed{seed}", f"fold{fold}"]
    if method:
        parts.append(method)
    p = Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ckpt_exists(ckpt_dir: Path, model_name: str) -> bool:
    return (ckpt_dir / model_name).exists()


def save_net_state(net: nn.Module, path: Path, baseline_hazards=None,
                   baseline_cumulative_hazards=None, params: dict | None = None):
    checkpoint = {
        "net_state":                   net.state_dict(),
        "baseline_hazards":            baseline_hazards,
        "baseline_cumulative_hazards": baseline_cumulative_hazards,
        "params":                      params or {},
    }
    torch.save(checkpoint, str(path))


def load_net_state(net: nn.Module, path: Path, device=None, model=None):
    map_dev    = device if device is not None else "cpu"
    checkpoint = torch.load(str(path), map_location=map_dev, weights_only=False)
    net.load_state_dict(checkpoint["net_state"])
    if model is not None:
        if checkpoint.get("baseline_hazards") is not None:
            model.baseline_hazards_ = checkpoint["baseline_hazards"]
        if checkpoint.get("baseline_cumulative_hazards") is not None:
            model.baseline_cumulative_hazards_ = checkpoint["baseline_cumulative_hazards"]
    if device is not None:
        net.to(device)
    return checkpoint.get("params", {})


def save_fixed_epochs(path_dir, deepsurv_type, best_epoch):
    config   = {"fixed_epochs": best_epoch}
    filepath = os.path.join(path_dir, f"fixed_epochs_{deepsurv_type}.json")
    with open(filepath, "w") as f:
        json.dump(config, f)
    print(f"Saved fixed_epochs={best_epoch} → {filepath}")


def load_fixed_epochs(path_dir, deepsurv_type):
    filepath = os.path.join(path_dir, f"fixed_epochs_{deepsurv_type}.json")
    with open(filepath, "r") as f:
        config = json.load(f)
    return config["fixed_epochs"]


# ── model factories ──────────────────────────────────────────────────────────

def create_rsf_model(seed):
    return RandomSurvivalForest(
                n_estimators=300, 
                max_depth=10, 
                min_samples_split=15, 
                min_samples_leaf=10, 
                max_features="log2", 
                n_jobs=-1, 
                random_state=seed
                )


def create_deepsurv_vanilla(in_features):
    net   = tt.practical.MLPVanilla(in_features, [64, 64], 1, batch_norm=True, dropout=0.1)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    return model


def create_deepsurv_simple(in_features):
    net   = nn.Linear(in_features, 1)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    return model


def create_cox():
    return CoxPHFitter(penalizer=0.1)


def create_callbacks(name):
    os.makedirs("models", exist_ok=True)
    return [tt.callbacks.EarlyStopping(
        patience=20, min_delta=1e-4,
        checkpoint_model=True,
        file_path=f"models/{name}.pt",
        load_best=True,
    )]


# ── fold management ──────────────────────────────────────────────────────────

def get_or_create_folds(X, y=None, dataset_name="dataset", seed=42,
                        n_splits=5, base_path="tmp/splits/"):
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{dataset_name}_seed{seed}.pkl")

    if os.path.exists(file_path):
        print(f"Loading existing folds from {file_path}")
        return load(file_path)

    print(f"Creating new folds and saving to {file_path}")
    kf    = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]
    data  = {"seed": seed, "n_splits": n_splits, "folds": folds}
    dump(data, file_path)
    return data


# ── results processing ────────────────────────────────────────────────────────

def process_results(results):
    rows = []
    for (seed, nan_ratio, model_results) in results:
        rsf, deepsurv_simple, deepsurv_vanilla, cox = model_results
        models = {
            "RSF":         (rsf[0],             rsf[1]),
            "Cox Simple":  (deepsurv_simple[0],  deepsurv_simple[1]),
            "Cox Vanilla": (deepsurv_vanilla[0], deepsurv_vanilla[1]),
            "Cox":         (cox[0],              cox[1]),
        }
        for model_name, (train_dict, test_dict) in models.items():
            for method in train_dict:
                train_vals = np.array(train_dict[method], dtype=float)
                test_vals  = np.array(test_dict[method], dtype=float)
                n_train = np.sum(~np.isnan(train_vals))
                n_test  = np.sum(~np.isnan(test_vals))
                if n_train == 0 or n_test == 0:
                    print(f"  WARNING: all folds are NaN for {model_name}/{method}")
                elif n_train < len(train_vals):
                    print(f"  WARNING: {len(train_vals)-n_train} NaN fold(s) excluded from train mean for {model_name}/{method}")
                if n_test > 0 and n_test < len(test_vals):
                    print(f"  WARNING: {len(test_vals)-n_test} NaN fold(s) excluded from test mean for {model_name}/{method}")
                rows.append({
                    "seed":       seed,
                    "nan_ratio":  nan_ratio,
                    "model":      model_name,
                    "method":     method,
                    "train_mean": np.nanmean(train_vals) if n_train > 0 else np.nan,
                    "train_std":  np.nanstd(train_vals)  if n_train > 0 else np.nan,
                    "test_mean":  np.nanmean(test_vals)  if n_test  > 0 else np.nan,
                    "test_std":   np.nanstd(test_vals)   if n_test  > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def print_stats(label, values):
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print(f"  {label:10s} → nan")
    else:
        print(f"  {label:10s} → mean: {valid.mean():.4f} | std: {valid.std():.4f} | "
              f"min: {valid.min():.4f} | max: {valid.max():.4f}")


# ── output tee ────────────────────────────────────────────────────────────────

class Tee:
    """Writes simultaneously to console and file."""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file    = open(filepath, 'w')

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


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tabpfn", choices=["tabpfn", "tabicl", "tabdpt"])
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    seed  = args.seed
    model = args.model

    DATASETS = [
        ("OrmoniTirodei", "Total mortality", "Follow Up Data"),
        ("HURRAH",        "STATO_AL_FU",     "FU"),
    ]

    percentages = [0.10, 0.30, 0.50, 0.70, 0.90]  # extend freely: masks are cumulative

    print("STARTED")
    res_path = "results_nan"
    os.makedirs(res_path, exist_ok=True)

    for dataset_name, feature_event, feature_time in DATASETS:
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")

        # main() now handles all percentages in a single call so that:
        #   1. set_seed is reset exactly once per dataset
        #   2. NaN masks are generated once per fold and reused across percentages
        results_by_pct = main(dataset_name, feature_event, feature_time,
                              seed, percentages, model)

        # Flatten to the format expected by process_results
        flat_results = [
            (seed, pct, model_results)
            for pct, model_results in sorted(results_by_pct.items())
        ]

        output_file = f"{res_path}/results_cv_{dataset_name}_{model}_nan_seed{seed}.txt"

        with Tee(output_file):
            df = process_results(flat_results)

            df_agg = df.groupby(["nan_ratio", "model", "method"]).agg(
                test_mean= ("test_mean",  "mean"),
                test_std=  ("test_std",   "mean"),
                train_mean=("train_mean", "mean"),
                train_std= ("train_std",  "mean"),
            ).reset_index()

            df_agg["test_summary"]  = df_agg.apply(
                lambda x: f"{x['test_mean']:.3f} ± {x['test_std']:.3f}", axis=1)
            df_agg["train_summary"] = df_agg.apply(
                lambda x: f"{x['train_mean']:.3f} ± {x['train_std']:.3f}", axis=1)

            pivot_test = df_agg.pivot_table(
                index=["nan_ratio", "model"], columns="method",
                values="test_summary", aggfunc="first")
            print(pivot_test.to_markdown())

            pivot_train = df_agg.pivot_table(
                index=["nan_ratio", "model"], columns="method",
                values="train_summary", aggfunc="first")
            print(pivot_train.to_markdown())

            for nan_ratio in df_agg["nan_ratio"].unique():
                print(f"\n=== NaN Ratio: {nan_ratio} ===")
                subset = df_agg[df_agg["nan_ratio"] == nan_ratio]
                for mdl in subset["model"].unique():
                    print(f"\n-- {mdl} --")
                    for _, row in subset[subset["model"] == mdl].iterrows():
                        print(f"{row['method']:>20} | "
                              f"Train: {row['train_mean']:.3f} ± {row['train_std']:.3f} | "
                              f"Test: {row['test_mean']:.3f} ± {row['test_std']:.3f}")

        print(f"Result saved in '{output_file}'")
