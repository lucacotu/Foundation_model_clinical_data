import sys
import os
import argparse
import joblib
import json
from unittest import case
from pathlib import Path
from joblib import dump, load
import numpy as np
from optuna import trial
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
import torch
import random
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import torch.nn as nn
from sksurv.ensemble import RandomSurvivalForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

from src.data_loader import load_data
from src.preprocessing import clean_and_impute, prepare_cox_data_cv, prepare_cox_data_hurrah_cv
from src.tabpfn import (
	get_tabpfn_embeddings,
	setup_figure, create_savefig_partial
)
from src.embedding_cox import EmbeddingCoxPH

def set_seed(seed: int):
    random.seed(seed)                        
    np.random.seed(seed)                     
    torch.manual_seed(seed)                  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(dataset_name, feature_event, feature_time, seed, percentage_nan):    
    match dataset_name:
        case "OrmoniTirodei":
            # 1. Load and clean the data
            data = load_data(dataset_name, "Dataset Sirbu")
            df = clean_and_impute(dataset_name, data)

            # 2. Extract specific features and targets (e.g. Mortality data)
            # And split into Train, Eval and Test sets
            df_mortality_train, df_mortality_eval = prepare_cox_data_cv(df)
        case "HURRAH":
            data = load_data(dataset_name, "Dataset Sirbu")
            df = clean_and_impute(dataset_name, data)
    
            # 2. Extract specific features and targets (e.g. Mortality data)
            # And split into Train, Eval and Test sets
            df_mortality_train, df_mortality_eval = prepare_cox_data_hurrah_cv(df)
        case _:
            print("Name_dataset not defined")
            return

    nan_percentage = df_mortality_train.isna().sum().sum() / df_mortality_train.size * 100

    print(df_mortality_train.size)
    print(df_mortality_train.shape)
    print("% PRIMA: ",nan_percentage)

    X_eval = df_mortality_eval.drop(columns=[feature_time,feature_event])
    y_eval = df_mortality_eval[feature_event]
    t_eval = df_mortality_eval[feature_time].values.astype(np.float32)

    X = df_mortality_train.drop(columns=[feature_time,feature_event])
    y = df_mortality_train[feature_event]
    t = df_mortality_train[feature_time].values.astype(np.float32)
    
    df_tmp = X.copy()
    df_tmp['__event__']    = y.values
    df_tmp['__duration__'] = t

    df_tmp_eval = X_eval.copy()
    df_tmp_eval['__event__']    = y_eval.values
    df_tmp_eval['__duration__'] = t_eval

    results = []

    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp_eval = df_tmp_eval.reset_index(drop=True)

    X = df_tmp.drop(columns=['__event__', '__duration__'])
    y = df_tmp['__event__']
    t = df_tmp['__duration__'].values.astype(np.float32)

    X_eval = df_tmp_eval.drop(columns=['__event__', '__duration__'])
    y_eval = df_tmp_eval['__event__']
    t_eval = df_tmp_eval['__duration__'].values.astype(np.float32)

    folds = get_or_create_folds(X, dataset_name=dataset_name, seed=seed, n_splits=5, base_path="nan/tabpfn")

    imputation_methods = ["embeddings","mean","median","constant","knn","imputer_bayesian", "imputer_rsf"]

    train_scores_rsf = {method: [] for method in imputation_methods}
    test_scores_rsf  = {method: [] for method in imputation_methods}
    train_scores_cox_vanilla = {method: [] for method in imputation_methods}
    test_scores_cox_vanilla  = {method: [] for method in imputation_methods}
    train_scores_cox_simple = {method: [] for method in imputation_methods}
    test_scores_cox_simple = {method: [] for method in imputation_methods}

    for fold, (train_idx, test_idx) in enumerate(folds["folds"]): 
        path_dir = get_ckpt_dir(dataset_name, seed, percentage_nan, fold+1)

        X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test   = y.iloc[train_idx], y.iloc[test_idx]
        t_train, t_test   = t[train_idx], t[test_idx]

        X_train = add_nan_to_target(X_train, target_percentage=percentage_nan)
        X_test = add_nan_to_target(X_test, target_percentage=percentage_nan)
        X_eval = add_nan_to_target(X_eval, target_percentage=percentage_nan)
    
        nan_percentage = X_train.isna().sum().sum() / X_train.size * 100
    
        print("% DOPO DOPO: ",nan_percentage)

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
        # 4. Generate the embeddings!
        print("Generating TabPFN Embeddings...")
        train_embeddings, test_embeddings = get_tabpfn_embeddings(X_train, y_train, X_test, y_test, seed)
        _, eval_embeddings = get_tabpfn_embeddings(X_train, y_train, X_eval, y_eval, seed)
        print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")

        print("DOPO GET TABPFN EMBEDDINGS")

        y_train_structured = np.array(
                    [(bool(e), t) for e, t in zip(y_train, t_train)],
                    dtype=[('event', bool), ('time', float)]
                )

        y_test_structured = np.array(
                [(bool(e), t) for e, t in zip(y_test, t_test)],
                dtype=[('event', bool), ('time', float)]
        )

        sets = [(train_embeddings,test_embeddings,eval_embeddings)]
        # ─────────────────────────────────────────────────────────────
        # 1. MEAN 
        # ─────────────────────────────────────────────────────────────
        name = "imputer_mean.pkl"
        if ckpt_exists(path_dir, name):
            print("Loading SimpleImputer with mean strategy...")
            imputer = joblib.load(os.path.join(path_dir, name))
            X_train_mean = imputer.transform(X_train).astype(np.float32)
            X_test_mean  = imputer.transform(X_test).astype(np.float32)
            X_eval_mean  = imputer.transform(X_eval).astype(np.float32)
        else:
            print("Fitting SimpleImputer with mean strategy...")
            imputer = SimpleImputer(strategy="mean")
            X_train_mean = imputer.fit_transform(X_train).astype(np.float32)
            X_test_mean  = imputer.transform(X_test).astype(np.float32)
            X_eval_mean  = imputer.transform(X_eval).astype(np.float32)
            joblib.dump(imputer, os.path.join(path_dir, name))

        sets.append((X_train_mean, X_test_mean, X_eval_mean))

        # ─────────────────────────────────────────────────────────────
        # 2. MEDIAN 
        # ─────────────────────────────────────────────────────────────
        name = "imputer_median.pkl"
        if ckpt_exists(path_dir, name):
            print("Loading SimpleImputer with median strategy...")
            imputer = joblib.load(os.path.join(path_dir, name))
            X_train_median = imputer.transform(X_train).astype(np.float32)
            X_test_median  = imputer.transform(X_test).astype(np.float32)
            X_eval_median  = imputer.transform(X_eval).astype(np.float32)
        else:
            print("Fitting SimpleImputer with median strategy...")
            imputer = SimpleImputer(strategy="median")
            X_train_median = imputer.fit_transform(X_train).astype(np.float32)
            X_test_median  = imputer.transform(X_test).astype(np.float32)
            X_eval_median  = imputer.transform(X_eval).astype(np.float32)
            joblib.dump(imputer, os.path.join(path_dir, name))
        
        sets.append((X_train_median, X_test_median, X_eval_median))

        # ─────────────────────────────────────────────────────────────
        # 3. CONSTANT — sostituisce NaN con un valore fisso
        # ─────────────────────────────────────────────────────────────

        # Caso numerico: riempie con -1 (utile per tree-based models
        # che possono imparare a gestire il valore sentinella)
        name = "imputer_constant.pkl"
        if ckpt_exists(path_dir, name):
            print("Loading SimpleImputer with constant strategy...")
            imputer = joblib.load(os.path.join(path_dir, name))
            X_train_const_num = imputer.transform(X_train).astype(np.float32)
            X_test_const_num  = imputer.transform(X_test).astype(np.float32)
            X_eval_const_num  = imputer.transform(X_eval).astype(np.float32)
        else:
            print("Fitting SimpleImputer with constant strategy (fill_value=-1)...")
            imputer = SimpleImputer(strategy="constant", fill_value=-1)
            X_train_const_num = imputer.fit_transform(X_train).astype(np.float32)
            X_test_const_num  = imputer.transform(X_test).astype(np.float32)
            X_eval_const_num = imputer.transform(X_eval).astype(np.float32)
            joblib.dump(imputer, os.path.join(path_dir, name))

        sets.append((X_train_const_num, X_test_const_num, X_eval_const_num))

        # ─────────────────────────────────────────────────────────────
        # 4. KNN IMPUTER — imputa basandosi sui k vicini più simili
        # ─────────────────────────────────────────────────────────────
        # n_neighbors: quanti campioni usare per calcolare il valore imputato
        # weights="uniform" → media semplice, "distance" → pesata per distanza
        name = "imputer_knn.pkl"
        if ckpt_exists(path_dir, name):
            print("Loading KNNImputer")
            imputer = joblib.load(os.path.join(path_dir, name))
            X_train_knn = imputer.transform(X_train).astype(np.float32)
            X_test_knn  = imputer.transform(X_test).astype(np.float32)
            X_eval_knn  = imputer.transform(X_eval).astype(np.float32)
        else:
            print("Fitting KNNImputer...")
            imputer = KNNImputer(n_neighbors=5, weights="uniform")
            X_train_knn = imputer.fit_transform(X_train).astype(np.float32)
            X_test_knn  = imputer.transform(X_test).astype(np.float32)
            X_eval_knn  = imputer.transform(X_eval).astype(np.float32)
            joblib.dump(imputer, os.path.join(path_dir, name))

        sets.append((X_train_knn, X_test_knn, X_eval_knn))

        # ─────────────────────────────────────────────────────────────
        # 5. ITERATIVE IMPUTER + BayesianRidge (equivalente MICE)
        # ─────────────────────────────────────────────────────────────
        # Ogni feature con NaN viene predetta dalle altre, iterativamente.
        # BayesianRidge è il modello di default ed è il più usato in letteratura.
        # max_iter: numero massimo di cicli di imputazione
        # sample_posterior=True → campiona dalla distribuzione posteriore
        #   (più fedele allo spirito MICE ma più lento)

        name = "imputer_mice.pkl"
        if ckpt_exists(path_dir, name):
            print("Loading IterativaImputer with BayesianRidge (MICE-like)...")
            imputer = joblib.load(os.path.join(path_dir, name))
            X_train_bayesian = imputer.transform(X_train).astype(np.float32)
            X_test_bayesian  = imputer.transform(X_test).astype(np.float32)
            X_eval_bayesian  = imputer.transform(X_eval).astype(np.float32)
        else:
            print("Fitting IterativeImputer with BayesianRidge (MICE-like)...")
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,
                random_state=seed,
                sample_posterior=False
            )
            X_train_bayesian = imputer.fit_transform(X_train).astype(np.float32)
            X_test_bayesian  = imputer.transform(X_test).astype(np.float32)
            X_eval_bayesian = imputer.transform(X_eval).astype(np.float32)
            joblib.dump(imputer, os.path.join(path_dir, name))
        
        sets.append((X_train_bayesian, X_test_bayesian, X_eval_bayesian))

        # ─────────────────────────────────────────────────────────────
        # 6. ITERATIVE IMPUTER + RandomForest (MissForest-like)
        # ─────────────────────────────────────────────────────────────
        # Versione più potente: usa RF invece di un modello lineare.
        # Cattura relazioni non lineari tra le variabili.
        # Più lento ma spesso più accurato su dati clinici.
        name = "imputer_rf.pkl"
        if ckpt_exists(path_dir, name):
            print("Loading IterativeImputer with RandomForestRegressor (MissForest-like)...")
            imputer = joblib.load(os.path.join(path_dir, name))
            X_train_rf = imputer.transform(X_train).astype(np.float32)
            X_test_rf  = imputer.transform(X_test).astype(np.float32)
            X_eval_rf  = imputer.transform(X_eval).astype(np.float32)
        else:
            print("Fitting IterativeImputer with RandomForestRegressor (MissForest-like)...")
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=50,
                    random_state=seed,
                    n_jobs=-1
                ),
                max_iter=10,
                random_state=seed
            )
            X_train_rf = imputer.fit_transform(X_train).astype(np.float32)
            X_test_rf  = imputer.transform(X_test).astype(np.float32)
            X_eval_rf  = imputer.transform(X_eval).astype(np.float32)
            joblib.dump(imputer, os.path.join(path_dir, name))

        sets.append((X_train_rf, X_test_rf, X_eval_rf))

        name_epochs_simple  = "fixed_epochs_cox_simple.json"
        name_epochs_vanilla = "fixed_epochs_cox_vanilla.json"

        if ckpt_exists(path_dir, name_epochs_simple):
            print("Loading epochs value...")
            fixed_epochs_simple = load_fixed_epochs(path_dir, "cox_simple")
        else:
            print("Fitting Cox Simple pilot model...")
            cox_simple_pilot = create_cox_simple(X_train_mean.shape[1])
            callback = create_callbacks("pilot_simple")
            cox_simple_pilot.fit(
                X_train_mean, (t_train , y_train.values),
                256, 100,
                callback,
                val_data=(X_eval_mean, (t_eval, y_eval.values)),
                verbose=True
                )
            log = cox_simple_pilot.log.to_pandas()
            total_epochs = len(log)
            fixed_epochs_simple = total_epochs - callback[0]._iter_since_best

            save_fixed_epochs(path_dir, "cox_simple", fixed_epochs_simple)


        if ckpt_exists(path_dir, name_epochs_vanilla):
            print("Loading epochs value...")
            fixed_epochs_vanilla = load_fixed_epochs(path_dir, "cox_vanilla")
        else:
            print("Fitting Cox Vanilla pilot model...")
            cox_vanilla_pilot = create_cox_vanilla(X_train_mean.shape[1])
            callback = create_callbacks("pilot_vanilla")
            cox_vanilla_pilot.fit(
                X_train_mean, (t_train , y_train.values),
                256, 100,
                callback,
                val_data=(X_eval_mean, (t_eval, y_eval.values)),
                verbose=True
                )
            log = cox_vanilla_pilot.log.to_pandas()
            total_epochs = len(log)
            fixed_epochs_vanilla = total_epochs - callback[0]._iter_since_best
            save_fixed_epochs(path_dir, "cox_vanilla", fixed_epochs_vanilla)

        for method, (X_train_m, X_test_m, X_eval_m ) in zip(imputation_methods, sets):

            path_dir = get_ckpt_dir(dataset_name, seed, percentage_nan, fold+1, method)

            print(f"Evaluating method: {method}")
            model = "rsf.pt"
            if ckpt_exists(path_dir, model):
                print("Loading Random Survival Forest...")
                rsf = load(os.path.join(path_dir, model))
            else:
                print("Fitting Random Survival Forest model...")
                rsf = create_rsf_model(seed)
                rsf.fit(X_train_m, y_train_structured)
                dump(rsf, os.path.join(path_dir, model))
    
            c_train = rsf.score(X_train_m, y_train_structured)
            c_test  = rsf.score(X_test_m, y_test_structured)
    
            train_scores_rsf[method].append(c_train)
            test_scores_rsf[method].append(c_test)

            model = "cox_simple.pt"
            cox_simple = create_cox_simple(X_train_m.shape[1])
            if ckpt_exists(path_dir, model):
                print("Loading Cox Simple model...")
                load_net_state(cox_simple.net, os.path.join(path_dir, model), device=cox_simple.device, model=cox_simple)
            else:
                print("Fitting Cox Simple model...")
                cox_simple.fit(
                    X_train_m, (t_train , y_train.values),
                    256, fixed_epochs_simple,
                    verbose=True
                    )
                cox_simple.compute_baseline_hazards()
                save_net_state(
                        cox_simple.net,
                        os.path.join(path_dir, model),
                        baseline_hazards=cox_simple.baseline_hazards_,
                        baseline_cumulative_hazards=cox_simple.baseline_cumulative_hazards_
                    )


            surv_test = cox_simple.predict_surv_df(X_test_m)
            surv_train = cox_simple.predict_surv_df(X_train_m)

            ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

            ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

            train_scores_cox_simple[method].append(ev_train.concordance_td())
            test_scores_cox_simple[method].append(ev_test.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())

            
            model = "cox_vanilla.pt"
            cox_vanilla = create_cox_vanilla(X_train_m.shape[1])
            if ckpt_exists(path_dir, model):
                print("Loading Cox Vanilla model...")
                load_net_state(cox_vanilla.net, os.path.join(path_dir, model), device=cox_vanilla.device, model=cox_vanilla)
            else:
                print("Loading Cox Vanilla model...")
                cox_vanilla.fit(
                    X_train_m, (t_train , y_train.values),
                    256, fixed_epochs_vanilla,
                    verbose=True
                    )
                cox_vanilla.compute_baseline_hazards()
                save_net_state(
                        cox_vanilla.net,
                        os.path.join(path_dir, model),
                        baseline_hazards=cox_vanilla.baseline_hazards_,
                        baseline_cumulative_hazards=cox_vanilla.baseline_cumulative_hazards_
                    )

            surv_test = cox_vanilla.predict_surv_df(X_test_m)
            surv_train = cox_vanilla.predict_surv_df(X_train_m)

            ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

            ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

            train_scores_cox_vanilla[method].append(ev_train.concordance_td())
            test_scores_cox_vanilla[method].append(ev_test.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td()) 
            
    return ((train_scores_rsf, test_scores_rsf),(train_scores_cox_simple,test_scores_cox_simple), (train_scores_cox_vanilla, test_scores_cox_vanilla) )


def get_ckpt_dir(dataset_name: str, seed: int, nan_percentage: float, fold: int, method="") -> Path:
    """Restituisce (e crea) la directory di imputer per questa combinazione."""
    if not method:
        p = Path("nan/tabpfn") / dataset_name /f"nan_percentage{nan_percentage}" / f"seed{seed}" / f"fold{fold}"
    else:
        p = Path("nan/tabpfn") / dataset_name /f"nan_percentage{nan_percentage}" / f"seed{seed}" / f"fold{fold}" / f"{method}"

    p.mkdir(parents=True, exist_ok=True)
    return p

def ckpt_exists(ckpt_dir: Path, model_name: str) -> bool:
    """True se esiste almeno il file del modello."""
    return (ckpt_dir / model_name).exists()

def save_net_state(net: nn.Module, path: Path, baseline_hazards=None, baseline_cumulative_hazards=None, params: dict | None = None):
    checkpoint = {
        "net_state": net.state_dict(),
        "baseline_hazards": baseline_hazards,
        "baseline_cumulative_hazards": baseline_cumulative_hazards,
        "params": params or {},
    }
    torch.save(checkpoint, str(path))


def load_net_state(net: nn.Module, path: Path, device=None, model=None):
    map_dev = device if device is not None else "cpu"
    checkpoint = torch.load(str(path), map_location=map_dev, weights_only=False)
    net.load_state_dict(checkpoint["net_state"])
    if model is not None:
        if checkpoint.get("baseline_hazards", None) is not None:
            model.baseline_hazards_ = checkpoint["baseline_hazards"]
        if checkpoint.get("baseline_cumulative_hazards", None) is not None:
            model.baseline_cumulative_hazards_ = checkpoint["baseline_cumulative_hazards"]
    if device is not None:
        net.to(device)
    return checkpoint.get("params", {})

def save_fixed_epochs(path_dir, cox_type, best_epoch):
    config = {"fixed_epochs": best_epoch}
    filepath = os.path.join(path_dir, f"fixed_epochs_{cox_type}.json")
    with open(filepath, "w") as f:
        json.dump(config, f)
    print(f"Saved fixed_epochs={best_epoch} → {filepath}")

def load_fixed_epochs(path_dir, cox_type):
    filepath = os.path.join(path_dir, f"fixed_epochs_{cox_type}.json")
    with open(filepath, "r") as f:
        config = json.load(f)
    return config["fixed_epochs"]

def create_rsf_model(seed):
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
        )
    return rsf

def create_cox_vanilla(in_features):
    net = tt.practical.MLPVanilla(
        in_features, [64, 64], 1,
        batch_norm=True, dropout=0.1
    )
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    return model

def create_cox_simple(in_features):
    net = nn.Linear(in_features, 1)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    return model

def create_callbacks(name):
    return [tt.callbacks.EarlyStopping(
        patience=20,
        min_delta=1e-4,
        checkpoint_model=True,
        file_path=f"models/{name}.pt",
        load_best=True
    )]

def get_or_create_folds(
    X,
    y=None,
    dataset_name="dataset",
    seed=42,
    n_splits=5,
    base_path="nan/tabpfn/"
):

    base_path = os.path.join(base_path,dataset_name)
    os.makedirs(base_path, exist_ok=True)
    # Nome file
    file_path = os.path.join(base_path, f"split_{dataset_name}_seed{seed}.pkl")

    # 🔁 Se esiste → carica
    if os.path.exists(file_path):
        print(f"Loading existing folds from {file_path}")
        data = load(file_path)
        return data

    # 🆕 Altrimenti crea
    print(f"Creating new folds and saving to {file_path}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = kf.split(X)

    folds = [(train_idx, test_idx) for train_idx, test_idx in splits]

    data = {
        "seed": seed,
        "n_splits": n_splits,
        "folds": folds
    }

    dump(data, file_path)

    return data

def process_results(results):
    rows = []

    for (seed, nan_ratio, model_results) in results:
        (   rsf,
            cox_simple,
            cox_vanilla
        ) = model_results

        models = {
            "RSF": (rsf[0], rsf[1]),
            "Cox Simple": (cox_simple[0], cox_simple[1]),
            "Cox Vanilla": (cox_vanilla[0], cox_vanilla[1])
        }

        for model_name, (train_dict, test_dict) in models.items():
            for method in train_dict.keys():

                train_vals = np.array(train_dict[method])
                test_vals = np.array(test_dict[method])

                rows.append({
                    "seed": seed,
                    "nan_ratio": nan_ratio,
                    "model": model_name,
                    "method": method,
                    "train_mean": train_vals.mean(),
                    "train_std": train_vals.std(),
                    "test_mean": test_vals.mean(),
                    "test_std": test_vals.std()
                })

    return pd.DataFrame(rows)



def print_stats(label, values):
        arr = np.array(values)
        print(f"  {label:10s} → mean: {arr.mean():.4f} | std: {arr.std():.4f} | min: {arr.min():.4f} | max: {arr.max():.4f}")

class Tee:
    """Scrive simultaneamente su console e su file."""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, 'w')
    
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
    
    def flush(self):
        self.console.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

def add_nan_to_target(df, target_percentage):
    df_copy = df.copy()
    
    total_values = df_copy.size
    current_nan = df_copy.isna().sum().sum()
    
    target_nan = int(total_values * target_percentage)
    to_add = target_nan - current_nan

    if to_add <= 0:
        return df_copy
    
    not_nan_indices = np.argwhere(~df_copy.isna().values)    
    to_add = min(to_add, len(not_nan_indices))
    selected_indices = not_nan_indices[
        np.random.choice(len(not_nan_indices), size=to_add, replace=False)
    ]
    for i, j in selected_indices:
        df_copy.iat[i, j] = np.nan
    
    return df_copy


def print_age_stats(label, series):
    print(f"  {label}:")
    print(f"    n      = {series.notna().sum()}")
    print(f"    mean   = {series.mean():.2f}")
    print(f"    std    = {series.std():.2f}")
    print(f"    min    = {series.min():.2f}")
    print(f"    25%    = {series.quantile(0.25):.2f}")
    print(f"    median = {series.median():.2f}")
    print(f"    75%    = {series.quantile(0.75):.2f}")
    print(f"    max    = {series.max():.2f}")
    print(f"    NaN    = {series.isna().sum()}")


if __name__ == "__main__":
    DATASETS = [
        ("OrmoniTirodei", "Total mortality", "Follow Up Data", "Age"),
        ("HURRAH",        "STATO_AL_FU",     "FU",            "ETA"),
    ]

    for dataset_name, feature_event, feature_time, age_col in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        data = load_data(dataset_name, "Dataset Sirbu")
        df = clean_and_impute(dataset_name, data)

        if dataset_name == "OrmoniTirodei":
            df_train, df_eval = prepare_cox_data_cv(df)
        else:
            df_train, df_eval = prepare_cox_data_hurrah_cv(df)

        print(f"Train shape: {df_train.shape} | Eval shape: {df_eval.shape}")

        for split_name, split_df in [("Train", df_train), ("Eval", df_eval)]:
            if age_col in split_df.columns:
                print_age_stats(f"{split_name} — {age_col}", split_df[age_col])
            else:
                print(f"  {split_name}: colonna '{age_col}' non trovata")