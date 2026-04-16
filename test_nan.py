import sys
import os
import argparse
from unittest import case
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

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = kf.split(X)

    imputation_methods = ["embeddings","mean","median","constant","knn","imputer_bayesian", "imputer_rsf"]

    train_scores_rsf = {method: [] for method in imputation_methods}
    test_scores_rsf  = {method: [] for method in imputation_methods}
    train_scores_cox_vanilla = {method: [] for method in imputation_methods}
    test_scores_cox_vanilla  = {method: [] for method in imputation_methods}
    train_scores_cox_simple = {method: [] for method in imputation_methods}
    test_scores_cox_simple = {method: [] for method in imputation_methods}

    for fold, (train_idx, test_idx) in enumerate(splits): 

        X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test   = y.iloc[train_idx], y.iloc[test_idx]
        t_train, t_test   = t[train_idx], t[test_idx]

        X_train = add_nan_to_target(X_train, target_percentage=percentage_nan)
    
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

        #y_eval_structured = np.array(
        #            [(bool(e), t) for e, t in zip(y_eval, t_eval)],
        #            dtype=[('event', bool), ('time', float)]
        #)

        y_test_structured = np.array(
                [(bool(e), t) for e, t in zip(y_test, t_test)],
                dtype=[('event', bool), ('time', float)]
        )

        sets = [(train_embeddings,test_embeddings,eval_embeddings)]
        # ─────────────────────────────────────────────────────────────
        # 1. MEAN 
        # ─────────────────────────────────────────────────────────────
        print("Fitting SimpleImputer with mean strategy...")
        imputer_mean = SimpleImputer(strategy="mean")
        X_train_mean = imputer_mean.fit_transform(X_train).astype(np.float32)
        X_test_mean  = imputer_mean.transform(X_test).astype(np.float32)
        X_eval_mean  = imputer_mean.transform(X_eval).astype(np.float32)

        sets.append((X_train_mean, X_test_mean, X_eval_mean))

        # ─────────────────────────────────────────────────────────────
        # 2. MEDIAN 
        # ─────────────────────────────────────────────────────────────
        print("Fitting SimpleImputer with median strategy...")
        imputer_median = SimpleImputer(strategy="median")
        X_train_median = imputer_median.fit_transform(X_train).astype(np.float32)
        X_test_median  = imputer_median.transform(X_test).astype(np.float32)
        X_eval_median  = imputer_median.transform(X_eval).astype(np.float32)
        
        sets.append((X_train_median, X_test_median, X_eval_median))

        # ─────────────────────────────────────────────────────────────
        # 3. CONSTANT — sostituisce NaN con un valore fisso
        # ─────────────────────────────────────────────────────────────

        # Caso numerico: riempie con -1 (utile per tree-based models
        # che possono imparare a gestire il valore sentinella)
        print("Fitting SimpleImputer with constant strategy (fill_value=-1)...")
        imputer_const_num = SimpleImputer(strategy="constant", fill_value=-1)
        X_train_const_num = imputer_const_num.fit_transform(X_train).astype(np.float32)
        X_test_const_num  = imputer_const_num.transform(X_test).astype(np.float32)
        X_eval_const_num = imputer_const_num.transform(X_eval).astype(np.float32)

        sets.append((X_train_const_num, X_test_const_num, X_eval_const_num))

        # ─────────────────────────────────────────────────────────────
        # 4. KNN IMPUTER — imputa basandosi sui k vicini più simili
        # ─────────────────────────────────────────────────────────────
        # n_neighbors: quanti campioni usare per calcolare il valore imputato
        # weights="uniform" → media semplice, "distance" → pesata per distanza
        print("Fitting KNNImputer...")
        imputer_knn = KNNImputer(n_neighbors=5, weights="uniform")
        X_train_knn = imputer_knn.fit_transform(X_train).astype(np.float32)
        X_test_knn  = imputer_knn.transform(X_test).astype(np.float32)
        X_eval_knn  = imputer_knn.transform(X_eval).astype(np.float32)

        sets.append((X_train_knn, X_test_knn, X_eval_knn))

        # ─────────────────────────────────────────────────────────────
        # 5. ITERATIVE IMPUTER + BayesianRidge (equivalente MICE)
        # ─────────────────────────────────────────────────────────────
        # Ogni feature con NaN viene predetta dalle altre, iterativamente.
        # BayesianRidge è il modello di default ed è il più usato in letteratura.
        # max_iter: numero massimo di cicli di imputazione
        # sample_posterior=True → campiona dalla distribuzione posteriore
        #   (più fedele allo spirito MICE ma più lento)
        print("Fitting IterativeImputer with BayesianRidge (MICE-like)...")
        imputer_mice = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=seed,
            sample_posterior=False   # True per MICE "puro"
        )
        X_train_bayesian = imputer_mice.fit_transform(X_train).astype(np.float32)
        X_test_bayesian  = imputer_mice.transform(X_test).astype(np.float32)
        X_eval_bayesian = imputer_mice.transform(X_eval).astype(np.float32)

        sets.append((X_train_bayesian, X_test_bayesian, X_eval_bayesian))

        # ─────────────────────────────────────────────────────────────
        # 6. ITERATIVE IMPUTER + RandomForest (MissForest-like)
        # ─────────────────────────────────────────────────────────────
        # Versione più potente: usa RF invece di un modello lineare.
        # Cattura relazioni non lineari tra le variabili.
        # Più lento ma spesso più accurato su dati clinici.
        print("Fitting IterativeImputer with RandomForestRegressor (MissForest-like)...")
        imputer_rf = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=50,
                random_state=seed,
                n_jobs=-1
            ),
            max_iter=10,
            random_state=seed
        )
        X_train_rf = imputer_rf.fit_transform(X_train).astype(np.float32)
        X_test_rf  = imputer_rf.transform(X_test).astype(np.float32)
        X_eval_rf  = imputer_rf.transform(X_eval).astype(np.float32)

        sets.append((X_train_rf, X_test_rf, X_eval_rf))


        for method, (X_train_m, X_test_m, X_eval_m ) in zip(imputation_methods, sets):
            print(f"Evaluating method: {method}")
    
            rsf = create_rsf_model(seed)
            rsf.fit(X_train_m, y_train_structured)
    
            c_train = rsf.score(X_train_m, y_train_structured)
            c_test  = rsf.score(X_test_m, y_test_structured)
    
            train_scores_rsf[method].append(c_train)
            test_scores_rsf[method].append(c_test)

            cox_simple = create_cox_simple(X_train_m.shape[1])
            callback = create_callbacks(method)

            batch_size = 256
            epochs     = 100
            cox_simple.fit(
                X_train_m, (t_train , y_train.values),
                batch_size, epochs,
                callback,
                val_data=(X_eval_m, (t_eval, y_eval.values)),
                verbose=True
                )
            _ = cox_simple.compute_baseline_hazards()

            surv_test = cox_simple.predict_surv_df(X_test_m)
            surv_train = cox_simple.predict_surv_df(X_train_m)

            ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

            ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

            train_scores_cox_simple[method].append(ev_train.concordance_td())
            test_scores_cox_simple[method].append(ev_test.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())

            cox_vanilla = create_cox_vanilla(X_train_m.shape[1])
            callback = create_callbacks(method+"_vanilla")

            batch_size = 256
            epochs     = 100
            cox_vanilla.fit(
                X_train_m, (t_train , y_train.values),
                batch_size, epochs,
                callback,
                val_data=(X_eval_m, (t_eval, y_eval.values)),
                verbose=True
                )
            _ = cox_vanilla.compute_baseline_hazards()

            surv_test = cox_vanilla.predict_surv_df(X_test_m)
            surv_train = cox_vanilla.predict_surv_df(X_train_m)

            ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

            ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

            train_scores_cox_vanilla[method].append(ev_train.concordance_td())
            test_scores_cox_vanilla[method].append(ev_test.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td()) 
            
    return ((train_scores_rsf, test_scores_rsf),(train_scores_cox_simple,test_scores_cox_simple), (train_scores_cox_vanilla, test_scores_cox_vanilla) )

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

def process_results(results):
    rows = []

    print(results)

    for (seed, nan_ratio, model_results) in results:

        (
            rsf,
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

    import numpy as np

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


if __name__ == "__main__":
    print("STARTED")
    seeds = [42, 123, 456, 789, 2024]
    percentage = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    res = [[]]#,[]]
    for index, seed in enumerate(seeds): 
        set_seed(seed)
        for percentage_nan in percentage:
            res[0].append((seed, percentage_nan, main("OrmoniTirodei", "Total mortality", "Follow Up Data", seed, percentage_nan)))

    df = process_results(res[0])

    df["test_summary"] = df.apply(
        lambda x: f"{x['test_mean']:.3f} ± {x['test_std']:.3f}", axis=1
    )

    df["train_summary"] = df.apply(
        lambda x: f"{x['train_mean']:.3f} ± {x['train_std']:.3f}", axis=1
    )
    tee = Tee("results_cv_tabpfn_nan.txt")
    sys.stdout = tee

    pivot = df.pivot_table(
        index=["nan_ratio", "model"],
        columns="method",
        values="test_summary",
        aggfunc="first"
    )

    print(pivot)

    pivot = df.pivot_table(
        index=["nan_ratio", "model"],
        columns="method",
        values="train_summary",
        aggfunc="first"
    )

    print(pivot)

    for nan_ratio in df["nan_ratio"].unique():
        print(f"\n=== NaN Ratio: {nan_ratio} ===")

        subset = df[df["nan_ratio"] == nan_ratio]

        for model in subset["model"].unique():
            print(f"\n-- {model} --")

            model_df = subset[subset["model"] == model]

            for _, row in model_df.iterrows():
                print(f"{row['method']:>20} | "
                    f"Train: {row['train_mean']:.3f} ± {row['train_std']:.3f} | "
                    f"Test: {row['test_mean']:.3f} ± {row['test_std']:.3f}")
                
    sys.stdout = tee.console
    tee.close()
    print("✅ Result saved in 'results_cv_tabpfn_nan.txt'")