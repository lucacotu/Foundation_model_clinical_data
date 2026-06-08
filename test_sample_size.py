import sys
import os
import argparse
import json
from joblib import dump, load
from pathlib import Path
import numpy as np
from lifelines import CoxPHFitter
import pandas as pd
from sklearn.model_selection import KFold
import torch
import random
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torch.nn as nn
from sksurv.ensemble import RandomSurvivalForest

from src.data_loader import load_data
from src.preprocessing import clean_and_impute, prepare_cox_data_cv, prepare_cox_data_hurrah_cv
from src.tabpfn import get_tabpfn_embeddings
from src.tabdpt import get_tabdpt_embeddings
from src.tabicl import get_tabicl_embeddings

DEFAULT_SAMPLE_SIZES = [100, 500, 1000, 2500, 5000, 10000, 15000, 20000, 25000]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(dataset_name, feature_event, feature_time, seed, model, sample_sizes=None):
    if sample_sizes is None:
        sample_sizes = DEFAULT_SAMPLE_SIZES

    match dataset_name:
        case "OrmoniTirodei":
            data = load_data(dataset_name, "Dataset Sirbu")
            df = clean_and_impute(dataset_name, data)
            df_train_raw, df_eval_raw = prepare_cox_data_cv(df)
        case "HURRAH":
            data = load_data(dataset_name, "Dataset Sirbu")
            df = clean_and_impute(dataset_name, data)
            df_train_raw, df_eval_raw = prepare_cox_data_hurrah_cv(df)
        case _:
            print("Name_dataset not defined")
            return None

    X_raw = df_train_raw.drop(columns=[feature_time, feature_event])
    y_raw = df_train_raw[feature_event]
    t_raw = df_train_raw[feature_time].values.astype(np.float32)

    X_eval_raw = df_eval_raw.drop(columns=[feature_time, feature_event])
    y_eval_raw = df_eval_raw[feature_event]
    t_eval_raw = df_eval_raw[feature_time].values.astype(np.float32)

    # results[preprocess_type][sample_size] = {metric: [fold_score, ...], actual_sizes: [...]}
    results = {}

    for preprocess_type in ["-1", "NaN"]:
        print(f"\n{'='*60}")
        print(f"Preprocess: {preprocess_type}")
        print(f"{'='*60}")

        if preprocess_type == "-1":
            X = X_raw.copy().fillna(-1).reset_index(drop=True)
            X_eval = X_eval_raw.copy().fillna(-1).reset_index(drop=True)
        else:
            X = X_raw.copy().fillna(np.nan).reset_index(drop=True)
            X_eval = X_eval_raw.copy().fillna(np.nan).reset_index(drop=True)

        y = y_raw.reset_index(drop=True)
        t = t_raw.copy()
        y_eval = y_eval_raw.reset_index(drop=True)
        t_eval = t_eval_raw.copy()

        if model == "tabdpt":
            X_arr = X.to_numpy()
            y_arr = y.to_numpy()
            X_eval_arr = X_eval.to_numpy()
            y_eval_arr = y_eval.to_numpy()
        else:
            X_arr = X
            y_arr = y
            X_eval_arr = X_eval
            y_eval_arr = y_eval

        folds = get_or_create_folds(X_arr, dataset_name=dataset_name, seed=seed, n_splits=5)

        results[preprocess_type] = {}

        fold_train_sizes = [len(tr) for tr, _ in folds["folds"]]
        processed_actual_n_combos = set()

        for sample_size in sample_sizes:
            actual_ns = tuple(min(sample_size, s) for s in fold_train_sizes)
            if actual_ns in processed_actual_n_combos:
                print(f"\n--- Sample size: {sample_size} skipped (actual train size {actual_ns[0]} already processed) ---")
                continue
            processed_actual_n_combos.add(actual_ns)
            print(f"\n--- Sample size: {sample_size} ---")

            scores = {
                'train_tab_deepsurv_simple':   [],
                'test_tab_deepsurv_simple':    [],
                'train_tab_deepsurv_vanilla':  [],
                'test_tab_deepsurv_vanilla':   [],
                'train_tab_rsf':               [],
                'test_tab_rsf':                [],
                'train_tab_cox':               [],
                'test_tab_cox':                [],
                'train_deepsurv_simple':       [],
                'test_deepsurv_simple':        [],
                'train_deepsurv_vanilla':      [],
                'test_deepsurv_vanilla':       [],
                'train_rsf':                   [],
                'test_rsf':                    [],
                'train_cox':                   [],
                'test_cox':                    [],
                'actual_train_sizes':          [],
            }

            for fold, (train_idx, test_idx) in enumerate(folds["folds"]):
                if model == "tabdpt":
                    X_train_full = X_arr[train_idx]
                    X_test       = X_arr[test_idx]
                    y_train_full = y_arr[train_idx]
                    y_test       = y_arr[test_idx]
                else:
                    X_train_full = X_arr.iloc[train_idx]
                    X_test       = X_arr.iloc[test_idx]
                    y_train_full = y_arr.iloc[train_idx]
                    y_test       = y_arr.iloc[test_idx]

                t_train_full = t[train_idx]
                t_test       = t[test_idx]

                # Nested subsampling: same permutation per fold across all sizes
                rng = np.random.default_rng(seed + fold * 10000)
                perm = rng.permutation(len(X_train_full))
                n = min(sample_size, len(X_train_full))
                sub_idx = np.sort(perm[:n])

                if model == "tabdpt":
                    X_train = X_train_full[sub_idx]
                    y_train = y_train_full[sub_idx]
                else:
                    X_train = X_train_full.iloc[sub_idx].reset_index(drop=True)
                    y_train = y_train_full.iloc[sub_idx].reset_index(drop=True)

                t_train   = t_train_full[sub_idx]
                actual_n  = len(X_train)
                scores['actual_train_sizes'].append(actual_n)
                print(f"  Fold {fold+1}: train={actual_n}, test={len(X_test)}")

                device = "cuda" if torch.cuda.is_available() else "cpu"

                if model == "tabpfn":
                    train_emb, test_emb = get_tabpfn_embeddings(X_train, y_train, X_test, y_test, seed)
                    _, eval_emb = get_tabpfn_embeddings(X_train, y_train, X_eval_arr, y_eval_arr, seed)
                elif model == "tabicl":
                    y_tr = y_train.values if hasattr(y_train, 'values') else y_train
                    train_emb, test_emb = get_tabicl_embeddings(X_train, y_tr, X_test, device=device, random_state=seed)
                    _, eval_emb = get_tabicl_embeddings(X_train, y_tr, X_eval_arr, device=device, random_state=seed)
                elif model == "tabdpt":
                    train_emb, test_emb = get_tabdpt_embeddings(X_train, y_train, X_test, device=device)
                    _, eval_emb = get_tabdpt_embeddings(X_train, y_train, X_eval_arr, device=device)
                else:
                    raise ValueError(f"Unknown model: {model}")

                print(f"  Embedding shape: {train_emb.shape}")

                y_train_struct = np.array(
                    [(bool(e), t_) for e, t_ in zip(y_train, t_train)],
                    dtype=[('event', bool), ('time', float)],
                )
                y_test_struct = np.array(
                    [(bool(e), t_) for e, t_ in zip(y_test, t_test)],
                    dtype=[('event', bool), ('time', float)],
                )

                ckpt_dir = get_ckpt_dir(dataset_name, preprocess_type, seed, fold + 1, model, actual_n)

                # ── Foundation model + DeepSurv Simple ───────────────────
                c_tr, c_te = _fit_deepsurv_simple(
                    train_emb, t_train, y_train, test_emb, t_test, y_test,
                    eval_emb, t_eval, y_eval_arr,
                    ckpt_dir / f"{model}_deepsurv_simple.pt",
                )
                scores['train_tab_deepsurv_simple'].append(c_tr)
                scores['test_tab_deepsurv_simple'].append(c_te)
                print(f"  [{model}+DS-simple]  train={c_tr:.4f}  test={c_te:.4f}")

                # ── Foundation model + DeepSurv Vanilla ──────────────────
                c_tr, c_te = _fit_deepsurv_vanilla(
                    train_emb, t_train, y_train, test_emb, t_test, y_test,
                    eval_emb, t_eval, y_eval_arr,
                    ckpt_dir / f"{model}_deepsurv_vanilla.pt",
                )
                scores['train_tab_deepsurv_vanilla'].append(c_tr)
                scores['test_tab_deepsurv_vanilla'].append(c_te)
                print(f"  [{model}+DS-vanilla] train={c_tr:.4f}  test={c_te:.4f}")

                # ── Foundation model + RSF ────────────────────────────────
                c_tr, c_te = _fit_rsf(
                    train_emb, y_train_struct, t_train, y_train,
                    test_emb,  y_test_struct,  t_test,  y_test,
                    ckpt_dir / f"{model}_rsf.pkl", seed,
                )
                scores['train_tab_rsf'].append(c_tr)
                scores['test_tab_rsf'].append(c_te)
                print(f"  [{model}+RSF]        train={c_tr:.4f}  test={c_te:.4f}")

                # ── Foundation model + Cox ────────────────────────────────
                c_tr, c_te = _fit_cox(
                    train_emb, t_train, y_train, test_emb, t_test, y_test,
                    ckpt_dir / f"{model}_cox.pkl",
                )
                scores['train_tab_cox'].append(c_tr)
                scores['test_tab_cox'].append(c_te)
                print(f"  [{model}+Cox]        train={c_tr:.4f}  test={c_te:.4f}")

                # ── Baselines on raw features (only for -1, models need no NaN) ──
                if preprocess_type != "NaN":
                    if model == "tabdpt":
                        X_tr_raw = X_train.astype(np.float32)
                        X_te_raw = X_test.astype(np.float32)
                        X_ev_raw = X_eval_arr.astype(np.float32)
                    else:
                        X_tr_raw = np.asarray(X_train, dtype=np.float32)
                        X_te_raw = np.asarray(X_test,  dtype=np.float32)
                        X_ev_raw = np.asarray(X_eval_arr, dtype=np.float32)

                    c_tr, c_te = _fit_deepsurv_simple(
                        X_tr_raw, t_train, y_train, X_te_raw, t_test, y_test,
                        X_ev_raw, t_eval, y_eval_arr,
                        ckpt_dir / "deepsurv_simple_baseline.pt",
                    )
                    scores['train_deepsurv_simple'].append(c_tr)
                    scores['test_deepsurv_simple'].append(c_te)
                    print(f"  [DS-simple]         train={c_tr:.4f}  test={c_te:.4f}")

                    c_tr, c_te = _fit_deepsurv_vanilla(
                        X_tr_raw, t_train, y_train, X_te_raw, t_test, y_test,
                        X_ev_raw, t_eval, y_eval_arr,
                        ckpt_dir / "deepsurv_vanilla_baseline.pt",
                    )
                    scores['train_deepsurv_vanilla'].append(c_tr)
                    scores['test_deepsurv_vanilla'].append(c_te)
                    print(f"  [DS-vanilla]        train={c_tr:.4f}  test={c_te:.4f}")

                    c_tr, c_te = _fit_rsf(
                        X_tr_raw, y_train_struct, t_train, y_train,
                        X_te_raw, y_test_struct,  t_test,  y_test,
                        ckpt_dir / "rsf_baseline.pkl", seed,
                    )
                    scores['train_rsf'].append(c_tr)
                    scores['test_rsf'].append(c_te)
                    print(f"  [RSF]               train={c_tr:.4f}  test={c_te:.4f}")

                    if model == "tabdpt":
                        X_cox_tr = pd.DataFrame(X_tr_raw)
                        X_cox_te = pd.DataFrame(X_te_raw)
                    else:
                        X_cox_tr = X_train
                        X_cox_te = X_test

                    c_tr, c_te = _fit_cox(
                        X_cox_tr, t_train, y_train, X_cox_te, t_test, y_test,
                        ckpt_dir / "cox_baseline.pkl",
                    )
                    scores['train_cox'].append(c_tr)
                    scores['test_cox'].append(c_te)
                    print(f"  [Cox]               train={c_tr:.4f}  test={c_te:.4f}")

            results[preprocess_type][sample_size] = scores

    return results


# ─── Model helpers ──────────────────────────────────────────────────────────

def _fit_deepsurv_simple(X_tr, t_tr, y_tr, X_te, t_te, y_te, X_ev, t_ev, y_ev, ckpt_path: Path):
    in_f = X_tr.shape[1]
    net  = nn.Linear(in_f, 1)
    m    = CoxPH(net, tt.optim.Adam)
    m.optimizer.set_lr(0.01)

    if ckpt_path.exists():
        load_net_state(m.net, ckpt_path, device=m.device, model=m)
    else:
        cb = [tt.callbacks.EarlyStopping(
            patience=20, min_delta=1e-4, checkpoint_model=True,
            file_path=str(ckpt_path.with_suffix('.tmp.pt')), load_best=True,
        )]
        m.fit(X_tr, (t_tr, np.asarray(y_tr)), 256, 100, cb,
              val_data=(X_ev, (t_ev, np.asarray(y_ev))), verbose=False)
        m.compute_baseline_hazards()
        save_net_state(m.net, ckpt_path,
                       baseline_hazards=m.baseline_hazards_,
                       baseline_cumulative_hazards=m.baseline_cumulative_hazards_)
    return _eval_deepsurv(m, X_tr, t_tr, y_tr, X_te, t_te, y_te)


def _fit_deepsurv_vanilla(X_tr, t_tr, y_tr, X_te, t_te, y_te, X_ev, t_ev, y_ev, ckpt_path: Path):
    in_f = X_tr.shape[1]
    net  = tt.practical.MLPVanilla(in_f, [32, 32], 1, batch_norm=True, dropout=0.1)
    m    = CoxPH(net, tt.optim.Adam)
    m.optimizer.set_lr(0.01)

    if ckpt_path.exists():
        load_net_state(m.net, ckpt_path, device=m.device, model=m)
    else:
        cb = [tt.callbacks.EarlyStopping(
            patience=20, min_delta=1e-4, checkpoint_model=True,
            file_path=str(ckpt_path.with_suffix('.tmp.pt')), load_best=True,
        )]
        m.fit(X_tr, (t_tr, np.asarray(y_tr)), 256, 100, cb,
              val_data=(X_ev, (t_ev, np.asarray(y_ev))), verbose=False)
        m.compute_baseline_hazards()
        save_net_state(m.net, ckpt_path,
                       baseline_hazards=m.baseline_hazards_,
                       baseline_cumulative_hazards=m.baseline_cumulative_hazards_)
    return _eval_deepsurv(m, X_tr, t_tr, y_tr, X_te, t_te, y_te)


def _eval_deepsurv(m, X_tr, t_tr, y_tr, X_te, t_te, y_te):
    s_te = m.predict_surv_df(X_te)
    s_tr = m.predict_surv_df(X_tr)
    ev_te = EvalSurv(s_te, t_te, np.asarray(y_te), censor_surv='km')
    ev_tr = EvalSurv(s_tr, t_tr, np.asarray(y_tr), censor_surv='km')
    return ev_tr.concordance_td(), ev_te.concordance_td()


def _fit_rsf(X_tr, y_tr_struct, t_tr, y_tr, X_te, y_te_struct, t_te, y_te, ckpt_path: Path, seed: int):
    if ckpt_path.exists():
        rsf = load(ckpt_path)
    else:
        rsf = RandomSurvivalForest(
            n_estimators=100, min_samples_split=10, min_samples_leaf=15,
            n_jobs=-1, random_state=seed,
        )
        rsf.fit(X_tr, y_tr_struct)
        dump(rsf, ckpt_path)

    def _surv_df(rsf, X):
        fns = rsf.predict_survival_function(X)
        tp  = fns[0].x
        mat = np.vstack([f(tp) for f in fns]).T
        return pd.DataFrame(mat, index=tp)

    s_te = _surv_df(rsf, X_te)
    s_tr = _surv_df(rsf, X_tr)
    ev_te = EvalSurv(s_te, t_te, np.asarray(y_te), censor_surv='km')
    ev_tr = EvalSurv(s_tr, t_tr, np.asarray(y_tr), censor_surv='km')
    return ev_tr.concordance_td(), ev_te.concordance_td()


def _fit_cox(X_tr, t_tr, y_tr, X_te, t_te, y_te, ckpt_path: Path):
    cph = None
    if ckpt_path.exists():
        cph = load(ckpt_path)
    else:
        df_fit = pd.DataFrame(X_tr) if not isinstance(X_tr, pd.DataFrame) else X_tr.copy()
        df_fit['__t__'] = t_tr
        df_fit['__e__'] = np.asarray(y_tr)
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(df_fit, duration_col='__t__', event_col='__e__')
            dump(cph, ckpt_path)
        except Exception as e:
            print(f"    Cox fit failed: {e}")
            cph = None

    if cph is None:
        return np.nan, np.nan

    X_te_df = pd.DataFrame(X_te) if not isinstance(X_te, pd.DataFrame) else X_te
    X_tr_df = pd.DataFrame(X_tr) if not isinstance(X_tr, pd.DataFrame) else X_tr
    s_te = cph.predict_survival_function(X_te_df)
    s_tr = cph.predict_survival_function(X_tr_df)
    ev_te = EvalSurv(s_te, t_te, np.asarray(y_te), censor_surv='km')
    ev_tr = EvalSurv(s_tr, t_tr, np.asarray(y_tr), censor_surv='km')
    return ev_tr.concordance_td(), ev_te.concordance_td()


# ─── Utilities ──────────────────────────────────────────────────────────────

def get_or_create_folds(X, y=None, dataset_name="dataset", seed=42, n_splits=5, base_path="tmp/splits/"):
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{dataset_name}_seed{seed}.pkl")

    if os.path.exists(file_path):
        print(f"Loading existing folds from {file_path}")
        return load(file_path)

    print(f"Creating new folds and saving to {file_path}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in kf.split(X)]
    data = {"seed": seed, "n_splits": n_splits, "folds": folds}
    dump(data, file_path)
    return data


def get_ckpt_dir(dataset_name, preprocess_type, seed, fold, model, sample_size) -> Path:
    p = (Path("checkpoints_samplesize") / model / dataset_name
         / preprocess_type / f"seed{seed}" / f"fold{fold}" / f"size{sample_size}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_net_state(net: nn.Module, path: Path, baseline_hazards=None, baseline_cumulative_hazards=None):
    torch.save({
        "net_state": net.state_dict(),
        "baseline_hazards": baseline_hazards,
        "baseline_cumulative_hazards": baseline_cumulative_hazards,
    }, str(path))


def load_net_state(net: nn.Module, path: Path, device=None, model=None):
    map_dev = device if device is not None else "cpu"
    ckpt = torch.load(str(path), map_location=map_dev, weights_only=False)
    net.load_state_dict(ckpt["net_state"])
    if model is not None:
        if ckpt.get("baseline_hazards") is not None:
            model.baseline_hazards_ = ckpt["baseline_hazards"]
        if ckpt.get("baseline_cumulative_hazards") is not None:
            model.baseline_cumulative_hazards_ = ckpt["baseline_cumulative_hazards"]
    if device is not None:
        net.to(device)


def process_results(results, model_name, seed, dataset_name):
    metric_pairs = [
        ("tab_deepsurv_simple",  "train_tab_deepsurv_simple",  "test_tab_deepsurv_simple"),
        ("tab_deepsurv_vanilla", "train_tab_deepsurv_vanilla", "test_tab_deepsurv_vanilla"),
        ("tab_rsf",              "train_tab_rsf",              "test_tab_rsf"),
        ("tab_cox",              "train_tab_cox",              "test_tab_cox"),
        ("deepsurv_simple",      "train_deepsurv_simple",      "test_deepsurv_simple"),
        ("deepsurv_vanilla",     "train_deepsurv_vanilla",     "test_deepsurv_vanilla"),
        ("rsf",                  "train_rsf",                  "test_rsf"),
        ("cox",                  "train_cox",                  "test_cox"),
    ]
    rows = []
    for preprocess_type, size_results in results.items():
        for sample_size, scores in size_results.items():
            actual_sizes = scores.get('actual_train_sizes', [])
            actual_mean  = int(np.mean(actual_sizes)) if actual_sizes else sample_size
            for metric, tr_key, te_key in metric_pairs:
                tr_vals = np.array([v for v in scores[tr_key] if not np.isnan(v)]) if scores[tr_key] else np.array([np.nan])
                te_vals = np.array([v for v in scores[te_key] if not np.isnan(v)]) if scores[te_key] else np.array([np.nan])
                if len(tr_vals) == 0:
                    tr_vals = np.array([np.nan])
                if len(te_vals) == 0:
                    te_vals = np.array([np.nan])
                rows.append({
                    "dataset":       dataset_name,
                    "foundation_model": model_name,
                    "seed":          seed,
                    "preprocess":    preprocess_type,
                    "sample_size":   sample_size,
                    "actual_size":   actual_mean,
                    "metric":        metric,
                    "train_mean":    np.nanmean(tr_vals),
                    "train_std":     np.nanstd(tr_vals),
                    "test_mean":     np.nanmean(te_vals),
                    "test_std":      np.nanstd(te_vals),
                })
    return pd.DataFrame(rows)


def print_stats(label, values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        print(f"  {label:35s} → no valid scores")
    else:
        print(f"  {label:35s} → mean: {arr.mean():.4f} | std: {arr.std():.4f} | min: {arr.min():.4f} | max: {arr.max():.4f}")


class Tee:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file    = open(filepath, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse model performance vs training set size.")
    parser.add_argument("--model", type=str, default="tabpfn", choices=["tabpfn", "tabicl", "tabdpt"])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES,
                        help="List of training-set sizes to evaluate (e.g. 100 500 1000 2500 5000)")

    args = parser.parse_args()
    seed         = args.seed
    model        = args.model
    sample_sizes = args.sample_sizes

    print(f"Model:        {model}")
    print(f"Seed:         {seed}")
    print(f"Sample sizes: {sample_sizes}")
    set_seed(seed)

    datasets = [
        ("OrmoniTirodei", "Total mortality", "Follow Up Data"),
        ("HURRAH",        "STATO_AL_FU",     "FU"),
    ]

    all_dfs = []

    for dataset_name, event_col, time_col in datasets:
        print(f"\n{'#'*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'#'*70}")

        res = main(dataset_name, event_col, time_col, seed, model, sample_sizes)
        if res is None:
            continue

        df_res = process_results(res, model, seed, dataset_name)
        all_dfs.append(df_res)

        output_dir = f"results/{dataset_name}/{model}"
        os.makedirs(output_dir, exist_ok=True)
        data_path = os.path.join(output_dir, f"results_samplesize_{dataset_name}_{model}_seed{seed}_data.txt")
        df_res.to_csv(data_path, index=False, sep='\t')
        print(f"\nData saved → {data_path}")

        txt_path = os.path.join(output_dir, f"results_samplesize_{dataset_name}_{model}_seed{seed}.txt")
        tee = Tee(txt_path)
        sys.stdout = tee

        print(f"\nDataset: {dataset_name} | Model: {model} | Seed: {seed}")
        print("=" * 70)

        for preprocess_type in ["-1", "NaN"]:
            if preprocess_type not in res:
                continue
            print(f"\nPreprocess: {preprocess_type}")
            print("-" * 60)

            for sample_size in sample_sizes:
                if sample_size not in res[preprocess_type]:
                    continue
                scores = res[preprocess_type][sample_size]
                actual_sizes = scores.get('actual_train_sizes', [])
                actual_mean  = int(np.mean(actual_sizes)) if actual_sizes else sample_size
                print(f"\n  Sample size (requested): {sample_size:6d}  |  actual mean: {actual_mean}")

                for label, tr_key, te_key in [
                    (f"{model}+DS-simple",  "train_tab_deepsurv_simple",  "test_tab_deepsurv_simple"),
                    (f"{model}+DS-vanilla", "train_tab_deepsurv_vanilla", "test_tab_deepsurv_vanilla"),
                    (f"{model}+RSF",        "train_tab_rsf",              "test_tab_rsf"),
                    (f"{model}+Cox",        "train_tab_cox",              "test_tab_cox"),
                    ("DS-simple (base)",    "train_deepsurv_simple",      "test_deepsurv_simple"),
                    ("DS-vanilla (base)",   "train_deepsurv_vanilla",     "test_deepsurv_vanilla"),
                    ("RSF (base)",          "train_rsf",                  "test_rsf"),
                    ("Cox (base)",          "train_cox",                  "test_cox"),
                ]:
                    if scores[te_key]:
                        print_stats(f"  TRAIN {label}", scores[tr_key])
                        print_stats(f"  TEST  {label}", scores[te_key])

        sys.stdout = tee.console
        tee.close()
        print(f"Text summary saved → {txt_path}")

