import sys
import os
import argparse
import json
from joblib import dump, load
import pickle
from pathlib import Path
from unittest import case
import numpy as np
from collections import defaultdict
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from optuna import trial
import pandas as pd
import matplotlib.pyplot as plt
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
import shap



from src.data_loader import load_data
from src.preprocessing import clean_and_impute, prepare_cox_data_cv, prepare_cox_data_hurrah_cv
from src.tabpfn import (
	get_tabpfn_embeddings,
	setup_figure, create_savefig_partial
)
from src.tabdpt import get_tabdpt_embeddings
from src.tabicl import get_tabicl_embeddings
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

def main(dataset_name, feature_event, feature_time, seed, model, tuning=False, compute_shap=False):
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

    df_tmp_original = df_tmp.copy()
    df_tmp_eval_original = df_tmp_eval.copy()

    results = []

    preprocess = [ "-1", "NaN"]
    for preprocess_type in preprocess:
        df_tmp = df_tmp_original.copy()
        df_tmp_eval = df_tmp_eval_original.copy()
        match preprocess_type:
            case "-1": 
                print("\n\nPreprocess: Replace NaN with -1")
                df_tmp = df_tmp.fillna(-1)
                df_tmp_eval = df_tmp_eval.fillna(-1)
            case "NaN":
                print("\n\nPreprocess: Replace NaN with np.nan")
                df_tmp = df_tmp.fillna(np.nan)
                df_tmp_eval = df_tmp_eval.fillna(np.nan)
            case _:
                print("Preprocess type not defined")

        df_tmp = df_tmp.reset_index(drop=True)
        df_tmp_eval = df_tmp_eval.reset_index(drop=True)

        c_index_train_scores_tab_tuned_deepsurv = []
        c_index_test_scores_tab_tuned_deepsurv = []
        c_index_train_scores_tab_tuned_rsf = []
        c_index_test_scores_tab_tuned_rsf = []
        c_index_train_scores_tab_deepsurv_vanilla = []
        c_index_test_scores_tab_deepsurv_vanilla = []
        c_index_train_scores_tab_deepsurv_simple = []
        c_index_test_scores_tab_deepsurv_simple = []
        c_index_train_scores_tab_cox = []
        c_index_test_scores_tab_cox = []
        c_index_train_scores_tab_rsf = []
        c_index_test_scores_tab_rsf = []
        c_index_train_scores_deepsurv_vanilla = []
        c_index_test_scores_deepsurv_vanilla = []
        c_index_train_scores_deepsurv_simple = []
        c_index_test_scores_deepsurv_simple = []
        c_index_test_scores_rsf = []
        c_index_train_scores_rsf = []
        c_index_test_scores_cox = []
        c_index_train_scores_cox = []
        
        X = df_tmp.drop(columns=['__event__', '__duration__'])
        y = df_tmp['__event__']
        t = df_tmp['__duration__'].values.astype(np.float32)

        X_eval = df_tmp_eval.drop(columns=['__event__', '__duration__'])
        y_eval = df_tmp_eval['__event__']
        t_eval = df_tmp_eval['__duration__'].values.astype(np.float32)

        feature_names_shap = list(X.columns)

        if model == "tabdpt":
            X = X.to_numpy()
            y = y.to_numpy()

            X_eval = X_eval.to_numpy()
            y_eval = y_eval.to_numpy()

        folds = get_or_create_folds(X, dataset_name=dataset_name, seed=seed, n_splits=5, base_path="tmp/splits/")

        if compute_shap:
            shap_fold_values = {key: [] for key in [
                "deepsurv_simple", "deepsurv_vanilla", "rsf", "cox",
            ]}
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

            if model == "tabdpt":
                X_train, X_test   = X[train_idx], X[test_idx]
                y_train, y_test   = y[train_idx], y[test_idx]
            else:
                X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test   = y.iloc[train_idx], y.iloc[test_idx]
            t_train, t_test   = t[train_idx], t[test_idx]

            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 4. Generate the embeddings!
            if model == "tabpfn":
                print("Generating TabPFN Embeddings...")
                train_embeddings, test_embeddings = get_tabpfn_embeddings(X_train, y_train, X_test, y_test, seed)
                _, eval_embeddings = get_tabpfn_embeddings(X_train, y_train, X_eval, y_eval, seed)
                print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")
            elif model == "tabicl":
                print("Generating Tabicl Embeddings...")
                train_embeddings, test_embeddings = get_tabicl_embeddings(X_train, y_train.values, X_test, device=device, random_state=seed)
                _, eval_embeddings = get_tabicl_embeddings(X_train, y_train.values, X_eval, device=device, random_state=seed)
                print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")
            elif model == "tabdpt":
                print("Generating TabDPT Embeddings...")
                train_embeddings, test_embeddings = get_tabdpt_embeddings(X_train, y_train, X_test, device=device)
                _, eval_embeddings = get_tabdpt_embeddings(X_train, y_train, X_eval, device=device)
                print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")
            else: 
                raise ValueError("Model not defined")
            
            y_train_structured = np.array(
                        [(bool(e), t) for e, t in zip(y_train, t_train)],
                        dtype=[('event', bool), ('time', float)]
                    )

            y_eval_structured = np.array(
                        [(bool(e), t) for e, t in zip(y_eval, t_eval)],
                        dtype=[('event', bool), ('time', float)]
            )

            y_test_structured = np.array(
                    [(bool(e), t) for e, t in zip(y_test, t_test)],
                    dtype=[('event', bool), ('time', float)]
            )
            
            ckpt_dir = get_ckpt_dir(dataset_name, preprocess_type, seed, fold + 1, model)

            if tuning:

                deepsurv_params = load_params(ckpt_dir, f"{model}_deepsurv_tuned")

                if deepsurv_params is not None and ckpt_exists(ckpt_dir, f"{model}_deepsurv_tuned"):
                    print(f"  [ckpt] Loading {model.upper()} DeepSurv Tuned from {ckpt_dir}")
                    net = tt.practical.MLPVanilla(
                            in_features = deepsurv_params["embedding_dim"], 
                            num_nodes = deepsurv_params["num_nodes"], 
                            out_features = 1,
                            batch_norm=deepsurv_params["batch_norm"], 
                            dropout=deepsurv_params["dropout"],)
                    deepsurv = CoxPH(net, tt.optim.Adam)
                    deepsurv.optimizer.set_lr(deepsurv_params["learning_rate"])
                    load_net_state(deepsurv.net, ckpt_dir / f"{model}_deepsurv_tuned.pt", device=deepsurv.device, model=deepsurv)
                else:
                    study_name = "deepsurv_tuning_{model}_{fold}_{type}_seed{seed}".format(model=model, fold=fold+1, type=preprocess_type, seed=seed)
                    study = optuna.create_study(
                        study_name=study_name,
                        direction="maximize",
                    )
                    
                    def objective_deepsurv(trial):
                        num_nodes = trial.suggest_categorical('num_nodes', [[32], [64], [128], [64, 32], [128, 64]])
                        dropout = trial.suggest_float('dropout', 0.0, 0.3)
                        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                        batch_norm = trial.suggest_categorical('batch_norm', [True, False])

                        net = tt.practical.MLPVanilla(
                            in_features = train_embeddings.shape[1], 
                            num_nodes = num_nodes, 
                            out_features = 1,
                            batch_norm=batch_norm, 
                            dropout=dropout,
                            )
                        _model = CoxPH(net, tt.optim.Adam)
                        _model.optimizer.set_lr(learning_rate)

                        _model.fit(train_embeddings, (t_train, np.asarray(y_train)),
                                epochs=100,
                                batch_size=256)
                        _model.compute_baseline_hazards()

                        surv = _model.predict_surv_df(eval_embeddings)

                        ev = EvalSurv(surv, t_eval, np.asarray(y_eval), censor_surv='km')

                        return ev.concordance_td() #_model.concordance_index(eval_embeddings, t_eval, y_eval.values)

                    study.optimize(objective_deepsurv, n_trials=100,n_jobs=-1)
                    print(f"Best parameters: {study.best_params}")
                    best_parameters = {
                        "embedding_dim": train_embeddings.shape[1],
                        **study.best_params
                    }
                    save_params(ckpt_dir, f"{model}_deepsurv_tuned", best_parameters)
                    
                    net = tt.practical.MLPVanilla(
                            in_features = best_parameters["embedding_dim"], 
                            num_nodes = best_parameters["num_nodes"], 
                            out_features = 1,
                            batch_norm=best_parameters["batch_norm"], 
                            dropout=best_parameters["dropout"],)
                    deepsurv = CoxPH(net, tt.optim.Adam)
                    deepsurv.optimizer.set_lr(best_parameters["learning_rate"])

                    callbacks = [tt.callbacks.EarlyStopping(patience=20,          
                                                            min_delta=1e-4,        
                                                            checkpoint_model=True, 
                                                            file_path=str(ckpt_dir / f"{model}_deepsurv_tuned_tmp.pt"), 
                                                            load_best=True,
                                                            )]
                    deepsurv.fit(
                        train_embeddings, (t_train, np.asarray(y_train)),
                        val_data=(eval_embeddings, (t_eval, np.asarray(y_eval))),
                        epochs=200,
                        callbacks=callbacks,
                        batch_size=128,
                    )

                    deepsurv.compute_baseline_hazards()
                    save_net_state(
                        deepsurv.net,
                        ckpt_dir / f"{model}_deepsurv_tuned.pt",
                        baseline_hazards=deepsurv.baseline_hazards_,
                        baseline_cumulative_hazards=deepsurv.baseline_cumulative_hazards_,
                        params=best_parameters,
                    )
                    print(f"  [ckpt] DeepSurv Tuned saved → {ckpt_dir / f'{model}_deepsurv_tuned.pt'}")
                
                surv_test = deepsurv.predict_surv_df(test_embeddings)
                surv_train = deepsurv.predict_surv_df(train_embeddings)

                ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
                ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')

                c_train = ev_train.concordance_td()
                c_test = ev_test.concordance_td()

                c_index_test_scores_tab_tuned_deepsurv.append(c_test)
                c_index_train_scores_tab_tuned_deepsurv.append(c_train)

                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")

                rsf_params = load_params(ckpt_dir, f"{model}_rsf_tuned")
                rsf_path   = ckpt_dir / f"{model}_rsf_tuned.pkl"

                if rsf_params is not None and rsf_path.exists():
                    print(f"  [ckpt] Loading {model.upper()} RSF Tuned from {ckpt_dir}")
                    rsf_tuned = load(rsf_path)

                else:
                    study_name = "rsf_tuning_{model}_{fold}_{type}_seed{seed}".format(model=model, fold=fold+1, type=preprocess_type, seed=seed)
                    study_rsf = optuna.create_study(
                        study_name=study_name,
                        direction="maximize",
                    )
                    
                    def objective_rsf(trial):
                        print(f"[Trial {trial.number}] START")
                        
                        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
                        max_depth = trial.suggest_int("max_depth", 3, 10)                  
                        min_samples_split = trial.suggest_int("min_samples_split", 5, 20)
                        min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 15)
                        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

                        rsf = RandomSurvivalForest(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            n_jobs=-1,
                            random_state=seed
                        )
                        
                        print(f"[Trial {trial.number}] fitting...") 

                        rsf.fit(train_embeddings, y_train_structured)
                        print(f"[Trial {trial.number}] done")

                        return rsf.score(eval_embeddings, y_eval_structured)

                    study_rsf.optimize(objective_rsf, n_trials=100,n_jobs=-1)
                    best_rsf_params = {
                        "n_jobs": -1,
                        "random_state": seed,
                        **study_rsf.best_params
                    }
                    save_params(ckpt_dir, f"{model}_rsf_tuned", best_rsf_params)

                    rsf_tuned = RandomSurvivalForest(**best_rsf_params)
                    rsf_tuned.fit(train_embeddings, y_train_structured)
                    dump(rsf_tuned, rsf_path)
                    print(f"  [ckpt] RSF Tuned saved → {rsf_path}")

                surv_test = rsf_tuned.predict_survival_function(test_embeddings)
                surv_train = rsf_tuned.predict_survival_function(train_embeddings)

                # Costruisci un DataFrame compatibile con pycox
                # righe = time points, colonne = pazienti
                time_points_test = surv_test[0].x
                surv_matrix_test = np.vstack([fn(time_points_test) for fn in surv_test]).T
                surv_test = pd.DataFrame(surv_matrix_test, index=time_points_test)
                time_points_train = surv_train[0].x
                surv_matrix_train = np.vstack([fn(time_points_train) for fn in surv_train]).T
                surv_train = pd.DataFrame(surv_matrix_train, index=time_points_train)

                # Calcola C-index di Antolini
                ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
                ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')
                c_train = ev_train.concordance_td()
                c_test = ev_test.concordance_td()

                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")
                c_index_train_scores_tab_tuned_rsf.append(c_train)
                c_index_test_scores_tab_tuned_rsf.append(c_test)
                
            # Simple Deepsurv on the embeddings (without tuning)
            simple_path = ckpt_dir / f"{model}_deepsurv_simple.pt"

            in_features = train_embeddings.shape[1]
            net_simple  = nn.Linear(in_features, 1)
            deepsurv_simple = CoxPH(net_simple, tt.optim.Adam)
            deepsurv_simple.optimizer.set_lr(0.01)

            if simple_path.exists():
                print(f"  [ckpt] Loading Deepsurv Simple from {simple_path}")
                load_net_state(deepsurv_simple.net, simple_path, device=deepsurv_simple.device, model=deepsurv_simple)

            else:
                callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path=str(simple_path.with_name(simple_path.stem + "_tmp" + simple_path.suffix)),
                                    load_best=True,
                                    )]

                deepsurv_simple.fit(
                    train_embeddings, (t_train , np.asarray(y_train)),
                    256, 100,
                    callbacks,
                    val_data=(eval_embeddings, (t_eval, np.asarray(y_eval))),
                    verbose=True
                )
                deepsurv_simple.compute_baseline_hazards()
                save_net_state(
                    deepsurv_simple.net,
                    simple_path,
                    baseline_hazards=deepsurv_simple.baseline_hazards_,
                    baseline_cumulative_hazards=deepsurv_simple.baseline_cumulative_hazards_
                )

            surv_test = deepsurv_simple.predict_surv_df(test_embeddings)
            surv_train = deepsurv_simple.predict_surv_df(train_embeddings)

            ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
            ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')

            c_index_test_scores_tab_deepsurv_simple.append(ev_test.concordance_td())
            c_index_train_scores_tab_deepsurv_simple.append(ev_train.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())

            #MLP VANILLA on the embeddings (without tuning)
            vanilla_path = ckpt_dir / f"{model}_deepsurv_vanilla.pt"
            in_features  = train_embeddings.shape[1]
            num_nodes    = [32, 32]
            out_features = 1
            batch_norm   = True
            dropout      = 0.1
            net = tt.practical.MLPVanilla(
                in_features, num_nodes, out_features,
                batch_norm=batch_norm, dropout=dropout
            )

            deepsurv_vanilla = CoxPH(net, tt.optim.Adam)
            deepsurv_vanilla.optimizer.set_lr(0.01)

            if vanilla_path.exists():
                print(f"  [ckpt] Loading Deepsurv Vanilla from {vanilla_path}")
                load_net_state(deepsurv_vanilla.net, vanilla_path, device=deepsurv_vanilla.device, model=deepsurv_vanilla)
            else:
                callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path=str(vanilla_path.with_name(vanilla_path.stem + "_tmp" + vanilla_path.suffix)), 
                                    load_best=True
                                    )]

                deepsurv_vanilla.fit(
                    train_embeddings, (t_train , np.asarray(y_train)),
                    256, 100,
                    callbacks,
                    val_data=(eval_embeddings, (t_eval, np.asarray(y_eval))),
                    verbose=True
                )
            
                deepsurv_vanilla.compute_baseline_hazards()
                save_net_state(
                    deepsurv_vanilla.net,
                    vanilla_path,
                    baseline_hazards=deepsurv_vanilla.baseline_hazards_,
                    baseline_cumulative_hazards=deepsurv_vanilla.baseline_cumulative_hazards_
                )

            surv_test = deepsurv_vanilla.predict_surv_df(test_embeddings)
            surv_train = deepsurv_vanilla.predict_surv_df(train_embeddings)

            ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
            ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')

            c_index_test_scores_tab_deepsurv_vanilla.append(ev_test.concordance_td())
            c_index_train_scores_tab_deepsurv_vanilla.append(ev_train.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())

            rsf_path = ckpt_dir / f"{model}_rsf.pkl"

            if rsf_path.exists():
                print(f"  [ckpt] Loading {model.upper()} RSF from {ckpt_dir}")
                rsf = load(rsf_path)

            else:   
                rsf = RandomSurvivalForest(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=15,
                    min_samples_leaf=10,
                    max_features="log2",
                    n_jobs=-1,                        
                    random_state=seed
                    )

                rsf.fit(train_embeddings, y_train_structured)
                dump(rsf, rsf_path)
                print(f"  [ckpt] RSF saved → {rsf_path}")

            #c_train = rsf.score(train_embeddings, y_train_structured)
            #c_test  = rsf.score(test_embeddings, y_test_structured)

            #print(f"C-index train: {c_train:.4f}")
            #print(f"C-index test:  {c_test:.4f}")
            #c_index_train_scores_tabpfn_rsf.append(c_train)
            #c_index_test_scores_tabpfn_rsf.append(c_test)

            surv_test = rsf.predict_survival_function(test_embeddings)
            surv_train = rsf.predict_survival_function(train_embeddings)

            # Costruisci un DataFrame compatibile con pycox
            # righe = time points, colonne = pazienti
            time_points_test = surv_test[0].x
            surv_matrix_test = np.vstack([fn(time_points_test) for fn in surv_test]).T
            surv_test = pd.DataFrame(surv_matrix_test, index=time_points_test)
            time_points_train = surv_train[0].x
            surv_matrix_train = np.vstack([fn(time_points_train) for fn in surv_train]).T
            surv_train = pd.DataFrame(surv_matrix_train, index=time_points_train)

            # Calcola C-index di Antolini
            ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
            ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')
            c_train = ev_train.concordance_td()
            c_test = ev_test.concordance_td()

            print(f"C-index train: {c_train:.4f}")
            print(f"C-index test:  {c_test:.4f}")
            c_index_train_scores_tab_rsf.append(c_train)
            c_index_test_scores_tab_rsf.append(c_test)

            #COX PATH
            cox_path = ckpt_dir / f"{model}_tab_cox.pkl"
            if cox_path.exists():
                print(f"  [ckpt] Loading {model.upper()} Cox from {ckpt_dir}")
                cph = load(cox_path)
            else:
                print(f" Fitting CoxPH on the embeddings...")
                df_fit = pd.DataFrame(train_embeddings)
                df_fit['__t__'] = t_train
                df_fit['__e__'] = np.asarray(y_train)

                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(df_fit, duration_col='__t__', event_col='__e__')
                dump(cph, cox_path)

            surv_test = cph.predict_survival_function(pd.DataFrame(test_embeddings))
            surv_train = cph.predict_survival_function(pd.DataFrame(train_embeddings))

            ev_test = EvalSurv(surv_test, durations=t_test, events=np.asarray(y_test), censor_surv='km')
            ev_train = EvalSurv(surv_train, durations=t_train, events=np.asarray(y_train), censor_surv='km')
            c_test = ev_test.concordance_td()
            c_train = ev_train.concordance_td()

            print(f"C-index test: {c_test:.4f}")
            print(f"C-index train: {c_train:.4f}")
            c_index_train_scores_tab_cox.append(c_train)
            c_index_test_scores_tab_cox.append(c_test)

            if compute_shap:
                _shap_ds_simple_emb  = deepsurv_simple
                _shap_ds_vanilla_emb = deepsurv_vanilla
                _shap_rsf_emb        = rsf
                _shap_cph_emb        = cph
                if tuning:
                    _shap_ds_tuned_emb = deepsurv

            if preprocess_type != "NaN":
                deepsurv_vanilla_base_path = ckpt_dir / "deepsurv_vanilla_baseline.pt"
                in_features  = X_train.shape[1]
                num_nodes    = [32, 32]
                out_features = 1
                batch_norm   = True
                dropout      = 0.1

                net = tt.practical.MLPVanilla(
                    in_features, num_nodes, out_features,
                    batch_norm=batch_norm, dropout=dropout
                )

                deepsurv_vanilla = CoxPH(net, tt.optim.Adam)
                deepsurv_vanilla.optimizer.set_lr(0.01)

                if deepsurv_vanilla_base_path.exists():
                    print(f"  [ckpt] Loading DeepSurv baseline from {deepsurv_vanilla_base_path}")
                    #model_vanilla = load(deepsurv_vanilla_base_path)
                    load_net_state(deepsurv_vanilla.net, deepsurv_vanilla_base_path, device=deepsurv_vanilla.device, model=deepsurv_vanilla)
                else:
                    batch_size = 256
                    epochs     = 100
                    callbacks  = [tt.callbacks.EarlyStopping( 
                                        patience=20,          
                                        min_delta=1e-4,        
                                        checkpoint_model=True, 
                                        file_path=str(deepsurv_vanilla_base_path.with_name(deepsurv_vanilla_base_path.stem + "_tmp" + deepsurv_vanilla_base_path.suffix)), 
                                        load_best=True,
                                        )]

                    deepsurv_vanilla.fit(
                        np.asarray(X_train, dtype=np.float32), (t_train , np.asarray(y_train)),
                        batch_size, epochs,
                        callbacks,
                        val_data=(np.asarray(X_eval, dtype=np.float32), (t_eval, np.asarray(y_eval))),
                        verbose=True
                    )

                    deepsurv_vanilla.compute_baseline_hazards()
                    save_net_state(
                        deepsurv_vanilla.net,
                        deepsurv_vanilla_base_path,
                        baseline_hazards=deepsurv_vanilla.baseline_hazards_,
                        baseline_cumulative_hazards=deepsurv_vanilla.baseline_cumulative_hazards_
                    )

                surv_test = deepsurv_vanilla.predict_surv_df(np.asarray(X_test, dtype=np.float32))
                surv_train = deepsurv_vanilla.predict_surv_df(np.asarray(X_train, dtype=np.float32))

                ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
                ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')

                c_index_test_scores_deepsurv_vanilla.append(ev_test.concordance_td())
                c_index_train_scores_deepsurv_vanilla.append(ev_train.concordance_td())

                print("C-index TRAIN:", ev_train.concordance_td())
                print("C-index TEST :", ev_test.concordance_td())

                deepsurv_simple_base_path = ckpt_dir / "deepsurv_simple_baseline.pt"
                in_features = X_train.shape[1]
                net_simple  = nn.Linear(in_features, 1)
                deepsurv_simple = CoxPH(net_simple, tt.optim.Adam)
                deepsurv_simple.optimizer.set_lr(0.01)

                if deepsurv_simple_base_path.exists():
                    print(f"  [ckpt] Loading DeepSurv baseline from {deepsurv_simple_base_path}")
                    load_net_state(deepsurv_simple.net, deepsurv_simple_base_path, device=deepsurv_simple.device, model=deepsurv_simple)
                else:

                    batch_size = 256
                    epochs     = 100
                    callbacks  = [tt.callbacks.EarlyStopping( 
                                        patience=20,          
                                        min_delta=1e-4,        
                                        checkpoint_model=True, 
                                        file_path=str(deepsurv_simple_base_path.with_name(deepsurv_simple_base_path.stem + "_tmp" + deepsurv_simple_base_path.suffix)), 
                                        load_best=True,
                                        )]

                    deepsurv_simple.fit(
                        np.asarray(X_train, dtype=np.float32), (t_train , np.asarray(y_train)),
                        batch_size, epochs,
                        callbacks,
                        val_data=(np.asarray(X_eval, dtype=np.float32), (t_eval, np.asarray(y_eval))),
                        verbose=True
                    )
                    deepsurv_simple.compute_baseline_hazards()
                    save_net_state(
                        deepsurv_simple.net,
                        deepsurv_simple_base_path,
                        baseline_hazards=deepsurv_simple.baseline_hazards_,
                        baseline_cumulative_hazards=deepsurv_simple.baseline_cumulative_hazards_
                    )


                surv_test = deepsurv_simple.predict_surv_df(np.asarray(X_test, dtype=np.float32))#.values.astype(np.float32)))
                surv_train = deepsurv_simple.predict_surv_df(np.asarray(X_train, dtype=np.float32))#.values.astype(np.float32)))

                ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
                ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')

                c_index_test_scores_deepsurv_simple.append(ev_test.concordance_td())
                c_index_train_scores_deepsurv_simple.append(ev_train.concordance_td())

                print("C-index TRAIN:", ev_train.concordance_td())
                print("C-index TEST :", ev_test.concordance_td())

                rsf_base_path = ckpt_dir / "rsf_baseline.pkl"
                if rsf_base_path.exists():
                    print(f"  [ckpt] Loading RSF baseline from {rsf_base_path}")
                    rsf = load(rsf_base_path)
                else:
                    rsf = RandomSurvivalForest(
                        n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=seed
                    )

                    rsf.fit(np.asarray(X_train), y_train_structured)
                    dump(rsf, rsf_base_path)
                    print(f"  [ckpt] RSF baseline saved → {rsf_base_path}")
                
                #c_index_test = rsf.score(X_test.values.astype(np.float32), y_test_structured)
                #c_index_train = rsf.score(X_train.values.astype(np.float32), y_train_structured)
                #c_index_test_scores_rsf.append(c_index_test)
                #c_index_train_scores_rsf.append(c_index_train)
                #print(f"C-index TEST (RSF): {c_index_test}")
                #print(f"C-index TRAIN (RSF): {c_index_train}")

                surv_test = rsf.predict_survival_function(X_test)
                surv_train = rsf.predict_survival_function(X_train)

                # Costruisci un DataFrame compatibile con pycox
                # righe = time points, colonne = pazienti
                time_points_test = surv_test[0].x
                surv_matrix_test = np.vstack([fn(time_points_test) for fn in surv_test]).T
                surv_test = pd.DataFrame(surv_matrix_test, index=time_points_test)
                time_points_train = surv_train[0].x
                surv_matrix_train = np.vstack([fn(time_points_train) for fn in surv_train]).T
                surv_train = pd.DataFrame(surv_matrix_train, index=time_points_train)

                # Calcola C-index di Antolini
                ev_train = EvalSurv(surv_train, t_train, np.asarray(y_train), censor_surv='km')
                ev_test = EvalSurv(surv_test, t_test, np.asarray(y_test), censor_surv='km')
                c_train = ev_train.concordance_td()
                c_test = ev_test.concordance_td()

                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")
                c_index_train_scores_rsf.append(c_train)
                c_index_test_scores_rsf.append(c_test)


                #COX PATH
                cox_path = ckpt_dir / f"{model}_cox.pkl"
                if cox_path.exists():
                    print(f"  [ckpt] Loading Cox from {ckpt_dir}")
                    cph = load(cox_path)
                else:
                    print(f" Fitting CoxPH on the original features...")
                    if model == "tabdpt":
                        df_fit = pd.DataFrame(np.asarray(X_train))
                        df_fit['__t__'] = t_train
                        df_fit['__e__'] = np.asarray(y_train)
                    else:
                        df_fit = X_train.copy()
                        df_fit['__t__'] = t_train
                        df_fit['__e__'] = y_train.values
                    cph = CoxPHFitter(penalizer=0.1)
                    cph.fit(df_fit, duration_col='__t__', event_col='__e__')
                    dump(cph, cox_path)

                if model == "tabdpt":
                    surv_test = cph.predict_survival_function(pd.DataFrame(np.asarray(X_test)))
                    surv_train = cph.predict_survival_function(pd.DataFrame(np.asarray(X_train)))
                else:
                    surv_test = cph.predict_survival_function(X_test)
                    surv_train = cph.predict_survival_function(X_train) 

                ev_test = EvalSurv(surv_test, durations=t_test, events=np.asarray(y_test), censor_surv='km')
                ev_train = EvalSurv(surv_train, durations=t_train, events=np.asarray(y_train), censor_surv='km')
                c_test = ev_test.concordance_td()
                c_train = ev_train.concordance_td()

                print(f"C-index test: {c_test:.4f}")
                print(f"C-index train: {c_train:.4f}")
                c_index_train_scores_cox.append(c_train)
                c_index_test_scores_cox.append(c_test)

            if compute_shap:
                shap_cache_path = ckpt_dir / "shap_values.pkl"
                if shap_cache_path.exists():
                    print(f"  [SHAP] Fold {fold+1}: loading cached values from {shap_cache_path}")
                    cached = load(shap_cache_path)
                    for key, val in cached.items():
                        if key in shap_fold_values:
                            shap_fold_values[key].append(val)
                else:
                    print(f"\n[SHAP] Fold {fold+1}: computing SHAP values...")
                    if model == "tabdpt":
                        X_train_shap = pd.DataFrame(X_train, columns=feature_names_shap)
                        X_test_shap  = pd.DataFrame(X_test,  columns=feature_names_shap)
                    else:
                        X_train_shap = X_train.reset_index(drop=True)
                        X_test_shap  = X_test.reset_index(drop=True)

                    background  = shap.sample(X_train_shap, min(100, len(X_train_shap)))
                    X_shap_test = X_test_shap.iloc[:min(100, len(X_test_shap))]

                    def _embed_fn(X_np):
                        X_np = np.atleast_2d(X_np)
                        X_df_in = pd.DataFrame(X_np, columns=feature_names_shap)
                        if model == "tabpfn":
                            _, emb = get_tabpfn_embeddings(X_train_shap, y_train, X_df_in, np.zeros(len(X_df_in)), seed)
                        elif model == "tabicl":
                            y_tr = y_train.values if hasattr(y_train, 'values') else y_train
                            _, emb = get_tabicl_embeddings(X_train_shap, y_tr, X_df_in, device=device, random_state=seed)
                        elif model == "tabdpt":
                            y_tr = y_train if isinstance(y_train, np.ndarray) else y_train.to_numpy()
                            _, emb = get_tabdpt_embeddings(X_train_shap.to_numpy(), y_tr, X_np, device=device)
                        return np.atleast_2d(emb)

                    def _emb_net_risk(net, X_np):
                        emb = _embed_fn(X_np)
                        net.eval()
                        with torch.no_grad():
                            dev = next(net.parameters()).device
                            t = torch.FloatTensor(emb.astype(np.float32)).to(dev)
                            if t.dim() == 1:
                                t = t.unsqueeze(0)
                            return net(t).cpu().numpy().flatten()

                    def _raw_net_risk(net, X_np):
                        net.eval()
                        with torch.no_grad():
                            dev = next(net.parameters()).device
                            t = torch.FloatTensor(np.asarray(X_np, dtype=np.float32)).to(dev)
                            if t.dim() == 1:
                                t = t.unsqueeze(0)
                            return net(t).cpu().numpy().flatten()

                    for key, net_ref in [("deepsurv_simple",  _shap_ds_simple_emb.net),
                                         ("deepsurv_vanilla", _shap_ds_vanilla_emb.net)]:
                        def _fn(X_np, _n=net_ref):
                            return _emb_net_risk(_n, X_np)
                        print(f"  [SHAP] {key}...")
                        sv = shap.KernelExplainer(_fn, background).shap_values(X_shap_test,nsamples=100)
                        shap_fold_values[key].append(np.abs(sv).mean(axis=0))

                    if tuning:
                        def _fn_tuned(X_np, _n=_shap_ds_tuned_emb.net):
                            return _emb_net_risk(_n, X_np)
                        print("  [SHAP] deepsurv_tuned...")
                        sv = shap.KernelExplainer(_fn_tuned, background).shap_values(X_shap_test,nsamples=100)
                        shap_fold_values["deepsurv_tuned"].append(np.abs(sv).mean(axis=0))

                    print("  [SHAP] rsf (embedding)...")
                    _rsf_emb_ref = _shap_rsf_emb
                    sv = shap.KernelExplainer(
                        lambda X, _r=_rsf_emb_ref: _r.predict(_embed_fn(X)), background
                    ).shap_values(X_shap_test,nsamples=100)
                    shap_fold_values["rsf"].append(np.abs(sv).mean(axis=0))

                    print("  [SHAP] cox (embedding)...")
                    _cph_emb_ref = _shap_cph_emb
                    sv = shap.KernelExplainer(
                        lambda X, _c=_cph_emb_ref: _c.predict_partial_hazard(pd.DataFrame(_embed_fn(X))).values,
                        background
                    ).shap_values(X_shap_test,nsamples=100)
                    shap_fold_values["cox"].append(np.abs(sv).mean(axis=0))

                    if preprocess_type != "NaN":
                        for key, net_ref in [("deepsurv_vanilla_baseline", deepsurv_vanilla.net),
                                             ("deepsurv_simple_baseline",  deepsurv_simple.net)]:
                            def _fn_base(X_np, _n=net_ref):
                                return _raw_net_risk(_n, X_np)
                            print(f"  [SHAP] {key}...")
                            sv = shap.KernelExplainer(_fn_base, background).shap_values(X_shap_test,nsamples=100)
                            shap_fold_values[key].append(np.abs(sv).mean(axis=0))

                        print("  [SHAP] rsf_baseline...")
                        _rsf_base_ref = rsf
                        sv = shap.KernelExplainer(
                            lambda X, _r=_rsf_base_ref: _r.predict(np.atleast_2d(np.asarray(X))), background
                        ).shap_values(X_shap_test,nsamples=100)
                        shap_fold_values["rsf_baseline"].append(np.abs(sv).mean(axis=0))

                        if model == "tabdpt":
                            _cph_base_ref = cph
                            def _cph_base_fn(X_np, _c=_cph_base_ref):
                                return _c.predict_partial_hazard(pd.DataFrame(np.atleast_2d(np.asarray(X_np)))).values
                        else:
                            _cph_base_ref = cph
                            def _cph_base_fn(X_np, _c=_cph_base_ref):
                                return _c.predict_partial_hazard(pd.DataFrame(np.atleast_2d(np.asarray(X_np, dtype=float)), columns=feature_names_shap)).values
                        print("  [SHAP] cox_baseline...")
                        sv = shap.KernelExplainer(_cph_base_fn, background).shap_values(X_shap_test,nsamples=100)
                        shap_fold_values["cox_baseline"].append(np.abs(sv).mean(axis=0))

                    fold_shap = {key: shap_fold_values[key][-1] for key in shap_fold_values if shap_fold_values[key]}
                    dump(fold_shap, shap_cache_path)
                    print(f"  [SHAP] Fold {fold+1}: values cached → {shap_cache_path}")

            # ── Predict ──────────────────────────────────────────────────────────────
            #survival_df = cox.predict_survival(test_embeddings)
            
            #print("Predicted survival probabilities for test set:")
            #print(survival_df)

            #print("Baseline Hazard: ",cox.model.baseline_hazards_)

            # Check the distribution of the predicted logits (risk scores)
            #logits = cox.model.predict(test_embeddings)
            #print(f"min: {logits.min():.3f}, max: {logits.max():.3f}, std: {logits.std():.3f}")
            # If std ≈ 0 → the model has not learned anything (lr too high/low, not enough epochs, etc.)
            
            '''
            # 5. Visualize embeddings via t-SNE
            if train_embeddings.ndim == 3:
                train_embeddings = train_embeddings[0]

            print(f"Running t-SNE on {train_embeddings.shape} points...")
            tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
            X_2d = tsne.fit_transform(train_embeddings)

            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.sans-serif'] = ['Arial']
            
            fig, ax = setup_figure(figsize=(8, 8), style='seaborn-v0_8-paper')
            
            palette = ['#1f77b4', '#d62728'] # Blue for negatives, Red for positives
            labels = {0: 'Negative (Alive)', 1: 'Positive (Deceased)'}
            
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
                    linewidths=0.1
                )
                
            ax.set_xlabel("t-SNE Component 1", fontsize=24, fontweight='bold')
            ax.set_ylabel("t-SNE Component 2", fontsize=24, fontweight='bold')
            
            ax.legend(
                bbox_to_anchor=(1.0, 1.0), 
                loc='upper right', 
                title="Mortality Status", 
                frameon=True, 
                shadow=True,
                fontsize=10
            )
            ax.grid(True, alpha=0.3, linestyle='--')
            
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            
            save_name_base = "tabpfn_mortality_tsne_dataset" + str(dataset_name)+ "_preprocess" + str(preprocess_type) + "_seed"+ str(seed) + "_fold" + str(fold + 1)
            savefig = create_savefig_partial(
                fig_dir=out_dir,
                fig_fmt='pdf',
                fig_size=(12, 12),
                save=True,
                dpi=300
            )
            savefig(fig, save_name_base)
            plt.close(fig)
            print(f"Visualization saved to {os.path.join(out_dir, save_name_base)}.pdf")
            '''

        if compute_shap and shap_fold_values:
            shap_dir  = f"results/{dataset_name}/{model}"
            os.makedirs(shap_dir, exist_ok=True)
            shap_path = os.path.join(shap_dir, f"shap_{dataset_name}_{model}_{preprocess_type}_seed{seed}.txt")
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

        results.append( (preprocess_type,
                        (c_index_train_scores_tab_tuned_deepsurv, c_index_test_scores_tab_tuned_deepsurv),
                        (c_index_train_scores_tab_tuned_rsf, c_index_test_scores_tab_tuned_rsf),
                        (c_index_train_scores_tab_deepsurv_vanilla, c_index_test_scores_tab_deepsurv_vanilla),
                        (c_index_train_scores_tab_deepsurv_simple, c_index_test_scores_tab_deepsurv_simple),
                        (c_index_train_scores_tab_rsf, c_index_test_scores_tab_rsf),
                        (c_index_train_scores_tab_cox, c_index_test_scores_tab_cox),
                        (c_index_train_scores_deepsurv_vanilla, c_index_test_scores_deepsurv_vanilla),
                        (c_index_train_scores_deepsurv_simple, c_index_test_scores_deepsurv_simple),
                        (c_index_train_scores_rsf, c_index_test_scores_rsf),
                        (c_index_train_scores_cox, c_index_test_scores_cox),
                        ))
    return results

def get_or_create_folds(
    X,
    y=None,
    dataset_name="dataset",
    seed=42,
    n_splits=5,
    base_path="tmp/splits/"
):
    # Crea directory se non esiste
    os.makedirs(base_path, exist_ok=True)

    # Nome file
    file_path = os.path.join(base_path, f"{dataset_name}_seed{seed}.pkl")

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


def get_ckpt_dir(dataset_name: str, preprocess_type: str, seed: int, fold: int, model: str) -> Path:
    """Restituisce (e crea) la directory di checkpoint per questa combinazione."""
    p = Path("checkpoints/") / model / dataset_name / preprocess_type / f"seed{seed}" / f"fold{fold}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_params(ckpt_dir: Path, model_name: str, params: dict):
    """Salva i migliori iperparametri in JSON."""
    path = ckpt_dir / f"{model_name}_params.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  [ckpt] Params saved → {path}")


def load_params(ckpt_dir: Path, model_name: str) -> dict | None:
    """Carica i parametri se esistono, altrimenti None."""
    path = ckpt_dir / f"{model_name}_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

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


def ckpt_exists(ckpt_dir: Path, model_name: str) -> bool:
    """True se esiste almeno il file del modello."""
    return (ckpt_dir / f"{model_name}.pt").exists() or \
           (ckpt_dir / f"{model_name}.pkl").exists()


def print_stats(label, values):
        arr = np.array(values)
        print(f"  {label:10s} → mean: {arr.mean():.6f} | std: {arr.std():.6f} | min: {arr.min():.6f} | max: {arr.max():.6f}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--model", type=str, default="tabpfn", choices=["tabpfn", "tabicl", "tabdpt"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shap", action="store_true", default=False)

    args = parser.parse_args()

    tuning = args.tuning
    seed = args.seed
    model = args.model
    compute_shap = args.shap

    print("STARTED")
    print(tuning)
    print(seed)
    print(model)
    set_seed(seed)

    dataset = [("OrmoniTirodei", "Total mortality", "Follow Up Data"),("HURRAH", "STATO_AL_FU", "FU")]

    for dataset_name, time_col, event_col in dataset:
        res = [[]]
        res[0].append((main(dataset_name, time_col, event_col, seed, model, tuning, compute_shap=compute_shap)))

        for exp_idx, experiment in enumerate(res):

            per_preprocess = defaultdict(lambda: {
                f'train_tab_tuned_deepsurv':    [],     'test_tab_tuned_deepsurv':  [],
                f'train_tab_tuned_rsf':         [],     'test_tab_tuned_rsf':       [],
                f'train_tab_deepsurv_vanilla':  [],     'test_tab_deepsurv_vanilla':[],
                f'train_tab_deepsurv_simple':   [],     'test_tab_deepsurv_simple':[],
                f'train_tab_rsf':               [],     'test_tab_rsf':             [],
                f'train_tab_cox':               [],     'test_tab_cox':             [],
                f'train_deepsurv_vanilla':          [],     'test_deepsurv_vanilla':        [],
                f'train_deepsurv_simple':           [],     'test_deepsurv_simple':         [],
                f'train_rsf':                       [],     'test_rsf':                     [],
                f'train_cox':                       [],     'test_cox':                     [],
            })
            
            for result_list in experiment:

                output_dir = f"results/{dataset_name}/{model}"
                os.makedirs(output_dir, exist_ok=True)
                filepath = os.path.join(output_dir, f"results_cv_{dataset_name}_{model}_seed{seed}.txt")
                
                tee = Tee(filepath)
                sys.stdout = tee

                for preprocess_type,(c_train_tab_tuned_deepsurv, c_test_tab_tuned_deepsurv),(c_train_tab_tuned_rsf, c_test_tab_tuned_rsf),(c_train_tab_deepsurv_vanilla, c_test_tab_deepsurv_vanilla),(c_train_tab_deepsurv_simple, c_test_tab_deepsurv_simple),(c_train_tab_rsf, c_test_tab_rsf),(c_train_tab_cox,c_test_tab_cox),(c_train_deepsurv_vanilla, c_test_deepsurv_vanilla),(c_train_deepsurv_simple, c_test_deepsurv_simple), (c_train_rsf, c_test_rsf),(c_train_cox, c_test_cox) in result_list:

                    print(f"\n  Preprocess: {preprocess_type} | Seed: {seed}")
                    if c_train_tab_tuned_deepsurv:
                        print_stats(f"{model.upper()} Train Tuned Deepsurv", c_train_tab_tuned_deepsurv)
                        print_stats(f"{model.upper()} Test Tuned Deepsurv",  c_test_tab_tuned_deepsurv)
                    if c_train_tab_tuned_rsf:
                        print_stats(f"{model.upper()} Train Tuned RSF", c_train_tab_tuned_rsf)
                        print_stats(f"{model.upper()} Test Tuned RSF",  c_test_tab_tuned_rsf)
                    print_stats(f"{model.upper()} Train Vanilla", c_train_tab_deepsurv_vanilla)
                    print_stats(f"{model.upper()} Test Vanilla",  c_test_tab_deepsurv_vanilla)
                    print_stats(f"{model.upper()} Train Simple", c_train_tab_deepsurv_simple)
                    print_stats(f"{model.upper()} Test Simple",  c_test_tab_deepsurv_simple)
                    print_stats(f"{model.upper()} Train RSF", c_train_tab_rsf)
                    print_stats(f"{model.upper()} Test RSF",  c_test_tab_rsf)
                    print_stats(f"{model.upper()} Train Cox", c_train_tab_cox)
                    print_stats(f"{model.upper()} Test Cox",  c_test_tab_cox)
                    if c_train_deepsurv_vanilla:
                        print_stats("Train Deepsurv Vanilla", c_train_deepsurv_vanilla)
                        print_stats("Test Deepsurv Vanilla",  c_test_deepsurv_vanilla)
                    if c_train_deepsurv_simple:
                        print_stats("Train Deepsurv Simple", c_train_deepsurv_simple)
                        print_stats("Test Deepsurv Simple",  c_test_deepsurv_simple)
                    if c_train_rsf:
                        print_stats("Train RSF", c_train_rsf)
                        print_stats("Test RSF",  c_test_rsf)
                    if c_train_cox:
                        print_stats("Train Cox", c_train_cox)
                        print_stats("Test Cox",  c_test_cox)
        sys.stdout = tee.console
        tee.close()
        print(f"✅ Result saved in 'results_cv_{dataset_name}_{model}_seed{seed}.txt'")



