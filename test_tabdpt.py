import sys
import os
import argparse
import json
from joblib import dump, load
import pickle
from pathlib import Path
from unittest import case
import numpy as np
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


from src.data_loader import load_data
from src.preprocessing import clean_and_impute, prepare_cox_data_cv, prepare_cox_data_hurrah_cv
from src.tabdpt import get_tabdpt_embeddings
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

def main(dataset_name, feature_event, feature_time, seed, tuning=False):    
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

        c_index_train_scores_tabdpt_vanilla = []
        c_index_test_scores_tabdpt_vanilla = []
        c_index_train_scores_tabdpt_tuned_cox = []
        c_index_test_scores_tabdpt_tuned_cox = []
        c_index_train_scores_tabdpt_tuned_rsf = []
        c_index_test_scores_tabdpt_tuned_rsf = []
        c_index_train_scores_tabdpt_simple = []
        c_index_test_scores_tabdpt_simple = []
        c_index_train_scores_cox_vanilla = []
        c_index_test_scores_cox_vanilla = []
        c_index_train_scores_cox_simple = []
        c_index_test_scores_cox_simple = []
        c_index_test_scores_rsf = []
        c_index_train_scores_rsf = []

        X = df_tmp.drop(columns=['__event__', '__duration__'])
        y = df_tmp['__event__']
        t = df_tmp['__duration__'].values.astype(np.float32)

        X = X.to_numpy()
        y = y.to_numpy()

        X_eval = df_tmp_eval.drop(columns=['__event__', '__duration__'])
        y_eval = df_tmp_eval['__event__']
        t_eval = df_tmp_eval['__duration__'].values.astype(np.float32)
        
        X_eval = X_eval.to_numpy()
        y_eval = y_eval.to_numpy()

        folds = get_or_create_folds(X, dataset_name=dataset_name, seed=seed, n_splits=5, base_path="tmp/splits/")

        for fold, (train_idx, test_idx) in enumerate(folds["folds"]): 

            X_train, X_test   = X[train_idx], X[test_idx]
            y_train, y_test   = y[train_idx], y[test_idx]
            t_train, t_test   = t[train_idx], t[test_idx]

            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 4. Generate the embeddings!
            print("Generating TabDPT Embeddings...")
            train_embeddings, test_embeddings = get_tabdpt_embeddings(X_train, y_train, X_test, device=device)
            _, eval_embeddings = get_tabdpt_embeddings(X_train, y_train, X_eval, device=device)
            print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")

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
            
            ckpt_dir = get_ckpt_dir(dataset_name, preprocess_type, seed, fold + 1)

            if tuning:
                
                cox_params = load_params(ckpt_dir, "tabdpt_cox_tuned")

                if cox_params is not None and ckpt_exists(ckpt_dir, "tabdpt_cox_tuned"):
                    print(f"  [ckpt] Loading Tabdpt CoxPH Tuned from {ckpt_dir}")

                    cox = EmbeddingCoxPH(**cox_params)
                    load_net_state(cox.model.net, ckpt_dir / "tabdpt_cox_tuned.pt", device=cox.model.device, model=cox.model)
                else:

                    study_name = "cox_tuning_tabdpt_{fold}_{type}_seed{seed}".format(fold=fold+1, type=preprocess_type, seed=seed)
                    study = optuna.create_study(
                        study_name=study_name,
                        direction="maximize",
                    )
                    
                    def objective_cox(trial):
                        num_nodes = trial.suggest_categorical('num_nodes', [[32], [64], [128], [64, 32], [128, 64]])
                        dropout = trial.suggest_float('dropout', 0.0, 0.3)
                        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                        batch_norm = trial.suggest_categorical('batch_norm', [True, False])

                        model = EmbeddingCoxPH(
                            embedding_dim=train_embeddings.shape[1],
                            num_nodes=num_nodes,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            batch_norm=batch_norm
                        )

                        model.fit(train_embeddings, t_train, y_train,
                                epochs=100,
                                batch_size=128)
                        model.compute_baseline()
                        return model.concordance_index(eval_embeddings, t_eval, y_eval)

                    study.optimize(objective_cox, n_trials=100,n_jobs=1)
                    print(f"Best parameters: {study.best_params}")
                    best_params = {
                        "embedding_dim": train_embeddings.shape[1],
                        **study.best_params
                    }
                    save_params(ckpt_dir, "tabdpt_cox_tuned", best_params)
                    
                    cox = EmbeddingCoxPH(**best_params)

                    callbacks = [tt.callbacks.EarlyStopping(patience=20,          
                                                            min_delta=1e-4,        
                                                            checkpoint_model=True, 
                                                            file_path=str(ckpt_dir / "tabdpt_cox_tuned.pt"), 
                                                            load_best=True
                                                            )]
                    cox.fit(
                                train_embeddings,
                                durations=t_train,
                                events=y_train,
                                val_data=(eval_embeddings, t_eval, y_eval),
                                epochs=200,
                                callbacks=callbacks,
                                batch_size=128,
                            )

                    cox.compute_baseline()
                    save_net_state(
                        cox.model.net,
                        ckpt_dir / "tabdpt_cox_tuned.pt",
                        baseline_hazards=cox.model.baseline_hazards_,
                        baseline_cumulative_hazards=cox.model.baseline_cumulative_hazards_,
                        params=best_params,
                    )
                    print(f"  [ckpt] Cox Tuned saved → {ckpt_dir / 'tabdpt_cox_tuned.pt'}")

                c_train = cox.concordance_index(train_embeddings, t_train, y_train)
                c_test  = cox.concordance_index(test_embeddings, t_test, y_test)

                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")
                c_index_train_scores_tabdpt_tuned_cox.append(c_train)
                c_index_test_scores_tabdpt_tuned_cox.append(c_test)

                rsf_params = load_params(ckpt_dir, "tabdpt_rsf_tuned")
                rsf_path   = ckpt_dir / "tabdpt_rsf_tuned.pkl"

                if rsf_params is not None and rsf_path.exists():
                    print(f"  [ckpt] Loading TabDPT RSF Tuned from {ckpt_dir}")
                    rsf_tuned = load(rsf_path)

                else:

                    study_name = "rsf_tuning_tabdpt_{fold}_{type}_seed{seed}".format(fold=fold+1, type=preprocess_type, seed=seed)
                    study_rsf = optuna.create_study(
                        study_name=study_name,
                        direction="maximize",
                    )
                    
                    def objective_rsf(trial):
                        print(f"[Trial {trial.number}] START")

                        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
                        max_depth = trial.suggest_int("max_depth", 3, 20)
                        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
                        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
                        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

                        rsf = RandomSurvivalForest(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            n_jobs=1,
                            random_state=seed
                        )

                        print(f"[Trial {trial.number}] fitting...") 
                        rsf.fit(train_embeddings, y_train_structured)

                        print(f"[Trial {trial.number}] done")
                        return rsf.score(eval_embeddings, y_eval_structured)

                    study_rsf.optimize(objective_rsf, n_trials=100,n_jobs=1,timeout=3600)

                    print(f"Best parameters: {study.best_params}")
                    best_rsf_params = {
                        "n_jobs": -1,
                        "random_state": seed,
                        **study_rsf.best_params
                    }
                    save_params(ckpt_dir, "tabdpt_rsf_tuned", best_rsf_params)
                    
                    rsf_tuned = RandomSurvivalForest(**best_rsf_params)
                    rsf_tuned.fit(train_embeddings, y_train_structured)
                    dump(rsf_tuned, rsf_path)
                    print(f"  [ckpt] RSF Tuned saved → {rsf_path}")
                
                c_train = rsf_tuned.score(train_embeddings, y_train_structured)
                c_test  = rsf_tuned.score(test_embeddings, y_test_structured)

                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")
                c_index_train_scores_tabdpt_tuned_rsf.append(c_train)
                c_index_test_scores_tabdpt_tuned_rsf.append(c_test)


            simple_path = ckpt_dir / "tabdpt_cox_simple.pt"
            in_features  = train_embeddings.shape[1]
            net = nn.Linear(in_features, 1)
            model_simple = CoxPH(net, tt.optim.Adam)
            model_simple.optimizer.set_lr(0.01)

            if simple_path.exists():
                print(f"  [ckpt] Loading CoxPH Simple from {simple_path}")
                load_net_state(model_simple.net, simple_path, device=model_simple.device, model=model_simple)
            else:
                callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path=str(simple_path.with_name(simple_path.stem + "_tmp" + simple_path.suffix)),
                                    load_best=True,
                                    )]

                model_simple.fit(
                    train_embeddings, (t_train , y_train),
                    256, 100,
                    callbacks,
                    val_data=(eval_embeddings, (t_eval, y_eval)),
                    verbose=True
                    )

                model.compute_baseline_hazards()
                save_net_state(
                    model_simple.net,
                    simple_path,
                    baseline_hazards=model_simple.baseline_hazards_,
                    baseline_cumulative_hazards=model_simple.baseline_cumulative_hazards_
                )

            surv_test = model_simple.predict_surv_df(test_embeddings)
            surv_train = model_simple.predict_surv_df(train_embeddings)

            ev_train = EvalSurv(surv_train, t_train, y_train, censor_surv='km')
            ev_test = EvalSurv(surv_test, t_test, y_test, censor_surv='km')

            c_index_test_scores_tabdpt_simple.append(ev_test.concordance_td())
            c_index_train_scores_tabdpt_simple.append(ev_train.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())
            
            #MLP VANILLA on the embeddings (without tuning)
            vanilla_path = ckpt_dir / "tabdpt_cox_vanilla.pt"
            in_features  = train_embeddings.shape[1]
            num_nodes    = [32, 32]
            out_features = 1
            batch_norm   = True
            dropout      = 0.1
            net = tt.practical.MLPVanilla(
                in_features, num_nodes, out_features,
                batch_norm=batch_norm, dropout=dropout
            )

            model_vanilla = CoxPH(net, tt.optim.Adam)
            model_vanilla.optimizer.set_lr(0.01)

            if vanilla_path.exists():
                print(f"  [ckpt] Loading CoxPH Vanilla from {vanilla_path}")
                load_net_state(model_vanilla.net, vanilla_path, device=model_vanilla.device, model=model_vanilla)
            else:
                callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path=str(vanilla_path),
                                    load_best=True,
                                    rm_file=False
                                    )]

                model_vanilla.fit(
                    train_embeddings, (t_train , y_train),
                    batch_size, epochs,
                    callbacks,
                    val_data=(eval_embeddings, (t_eval, y_eval)),
                    verbose=True
                )

                model_vanilla.compute_baseline_hazards()
                save_net_state(
                    model_vanilla.net,
                    vanilla_path,
                    baseline_hazards=model_vanilla.baseline_hazards_,
                    baseline_cumulative_hazards=model_vanilla.baseline_cumulative_hazards_
                )

            surv_test = model_vanilla.predict_surv_df(test_embeddings)
            surv_train = model_vanilla.predict_surv_df(train_embeddings)

            ev_train = EvalSurv(surv_train, t_train, y_train, censor_surv='km')

            ev_test = EvalSurv(surv_test, t_test, y_test, censor_surv='km')

            c_index_test_scores_tabdpt_vanilla.append(ev_test.concordance_td())
            c_index_train_scores_tabdpt_vanilla.append(ev_train.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())

            if preprocess_type != "NaN":
                cox_vanilla_base_path = ckpt_dir / "cox_vanilla_baseline.pkl"
                in_features  = X_train.shape[1]
                num_nodes    = [32, 32]
                out_features = 1
                batch_norm   = True
                dropout      = 0.1

                net = tt.practical.MLPVanilla(
                    in_features, num_nodes, out_features,
                    batch_norm=batch_norm, dropout=dropout
                )

                model_vanilla = CoxPH(net, tt.optim.Adam)
                model_vanilla.optimizer.set_lr(0.01)

                if cox_vanilla_base_path.exists():
                    print(f"  [ckpt] Loading Cox baseline from {cox_vanilla_base_path}")
                    load_net_state(model_vanilla.net, cox_vanilla_base_path, device=model_vanilla.device, model=model_vanilla)
                else:
                    batch_size = 256
                    epochs     = 100
                    callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path=str(cox_vanilla_base_path.with_name(cox_vanilla_base_path.stem + "_tmp" + cox_vanilla_base_path.suffix)), 
                                    load_best=True,
                                    )]

                    model_vanilla.fit(
                        X_train.astype(np.float32), (t_train , y_train),
                        batch_size, epochs,
                        callbacks,
                        val_data=(X_eval.astype(np.float32), (t_eval, y_eval)),
                        verbose=True
                    )

                    model_vanilla.compute_baseline_hazards()
                    save_net_state(
                        model_vanilla.net,
                        cox_vanilla_base_path,
                        baseline_hazards=model_vanilla.baseline_hazards_,
                        baseline_cumulative_hazards=model_vanilla.baseline_cumulative_hazards_
                    )

                surv_test = model_vanilla.predict_surv_df(X_test.astype(np.float32))
                surv_train = model_vanilla.predict_surv_df(X_train.astype(np.float32))

                ev_train = EvalSurv(surv_train, t_train, y_train, censor_surv='km')

                ev_test = EvalSurv(surv_test, t_test, y_test, censor_surv='km')

                c_index_test_scores_cox_vanilla.append(ev_test.concordance_td())
                c_index_train_scores_cox_vanilla.append(ev_train.concordance_td())

                print("C-index TRAIN:", ev_train.concordance_td())
                print("C-index TEST :", ev_test.concordance_td())

                cox_simple_base_path = ckpt_dir / "cox_simple_baseline.pkl"
                in_features = train_embeddings.shape[1]
                net_simple  = nn.Linear(in_features, 1)
                model_simple = CoxPH(net_simple, tt.optim.Adam)
                model_simple.optimizer.set_lr(0.01)

                if cox_simple_base_path.exists():
                    print(f"  [ckpt] Loading Cox baseline from {cox_simple_base_path}")
                    load_net_state(model_simple.net, cox_simple_base_path, device=model_simple.device, model=model_simple)
                else:

                    batch_size = 256
                    epochs     = 100
                    callbacks  = [tt.callbacks.EarlyStopping( 
                                        patience=20,          
                                        min_delta=1e-4,        
                                        checkpoint_model=True, 
                                        file_path=str(cox_simple_base_path.with_name(cox_simple_base_path.stem + "_tmp" + cox_simple_base_path.suffix)), 
                                        load_best=True,
                                        )]

                    model_simple.fit(
                        X_train.values.astype(np.float32), (t_train , y_train.values),
                        batch_size, epochs,
                        callbacks,
                        val_data=(X_eval.values.astype(np.float32), (t_eval, y_eval.values)),
                        verbose=True
                    )

                    model_simple.compute_baseline_hazards()
                    save_net_state(
                        model_simple.net,
                        cox_simple_base_path,
                        baseline_hazards=model_simple.baseline_hazards_,
                        baseline_cumulative_hazards=model_simple.baseline_cumulative_hazards_
                    )

                surv_test = model_simple.predict_surv_df(X_test.values.astype(np.float32))
                surv_train = model_simple.predict_surv_df(X_train.values.astype(np.float32))

                ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

                ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

                c_index_test_scores_cox_simple.append(ev_test.concordance_td())
                c_index_train_scores_cox_simple.append(ev_train.concordance_td())

                print("C-index TRAIN:", ev_train.concordance_td())
                print("C-index TEST :", ev_test.concordance_td())


                #RANDOM SURVIVAL FOREST
                rsf_base_path = ckpt_dir / "rsf_baseline.pkl"
                if rsf_base_path.exists():
                    print(f"  [ckpt] Loading RSF baseline from {rsf_base_path}")
                    rsf = load(rsf_base_path)
                else:
                    rsf = RandomSurvivalForest(
                        n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=1, random_state=seed
                    )

                    rsf.fit(X_train.values.astype(np.float32), y_train_structured)
                    dump(rsf, rsf_base_path)
                    print(f"  [ckpt] RSF baseline saved → {rsf_base_path}")
                
                c_index_test = rsf.score(X_test.astype(np.float32), y_test_structured)
                c_index_train = rsf.score(X_train.astype(np.float32), y_train_structured)
                c_index_test_scores_rsf.append(c_index_test)
                c_index_train_scores_rsf.append(c_index_train)
                print(f"C-index TEST (RSF): {c_index_test:.5f}")
                print(f"C-index TRAIN (RSF): {c_index_train:.5f}")



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
                loc='upper right', b
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
        results.append( (preprocess_type, 
                        (c_index_train_scores_tabdpt_vanilla, c_index_test_scores_tabdpt_vanilla), 
                        (c_index_train_scores_tabdpt_simple, c_index_test_scores_tabdpt_simple), 
                        (c_index_train_scores_tabdpt_tuned_cox, c_index_test_scores_tabdpt_tuned_cox),
                        (c_index_train_scores_tabdpt_tuned_rsf, c_index_test_scores_tabdpt_tuned_rsf), 
                        (c_index_train_scores_cox_vanilla, c_index_test_scores_cox_vanilla),
                        (c_index_train_scores_cox_simple, c_index_test_scores_cox_simple),
                        (c_index_train_scores_rsf, c_index_test_scores_rsf)))
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
        folds = load(file_path)
        return folds

    # 🆕 Altrimenti crea
    print(f"Creating new folds and saving to {file_path}")

    #if stratified and y is not None:
    #    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    #    splits = kf.split(X, y)
    #else:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = kf.split(X)

    folds = [(train_idx, test_idx) for train_idx, test_idx in splits]

    data = {
        "seed": seed,
        "n_splits": n_splits,
        "folds": folds
    }

    dump(data, file_path)

    return folds


def get_ckpt_dir(dataset_name: str, preprocess_type: str, seed: int, fold: int) -> Path:
    """Restituisce (e crea) la directory di checkpoint per questa combinazione."""
    p = Path("checkpoints/tabdpt") / dataset_name / preprocess_type / f"seed{seed}" / f"fold{fold}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning", action="store_true")
    args = parser.parse_args()
    
    tuning = args.tuning

    print("STARTED")
    seeds = [42, 123, 456, 789, 2024]
    res = [[],[]]
    for index, seed in enumerate(seeds): 
        set_seed(seed)

        res[0].append((seed, main("OrmoniTirodei", "Total mortality", "Follow Up Data", seed, tuning)))
        res[1].append((seed, main("HURRAH", "STATO_AL_FU", "FU", seed, tuning)))

    tee = Tee("results_cv_tabdpt.txt")
    sys.stdout = tee

    for exp_idx, experiment in enumerate(res):
        print(f"\n{'='*60}")
        print(f"Dataset {exp_idx + 1}")
        print(f"{'='*60}")

        all_train_tabdpt_vanilla = []
        all_test_tabdpt_vanilla  = []
        all_train_tabdpt_simple = []
        all_test_tabdpt_simple  = []
        all_train_tabdpt_tuned_cox = []
        all_test_tabdpt_tuned_cox  = []
        all_train_tabdpt_tuned_rsf = []
        all_test_tabdpt_tuned_rsf  = []
        all_train_cox    = []
        all_test_cox     = []
        all_train_rsf    = []
        all_test_rsf     = []

        for seed, result_list in experiment:
            for preprocess_type, (c_train_tabdpt_vanilla, c_test_tabdpt_vanilla),(c_train_tabdpt_simple, c_test_tabdpt_simple), (c_train_tabdpt_tuned_cox, c_test_tabdpt_tuned_cox), (c_train_tabdpt_tuned_rsf, c_test_tabdpt_tuned_rsf), (c_train_cox, c_test_cox), (c_train_rsf, c_test_rsf) in result_list:
                print(f"\n  Preprocess: {preprocess_type} | Seed: {seed}")
                print_stats("TabDPT Train Vanilla", c_train_tabdpt_vanilla)
                print_stats("TabDPT Test Vanilla",  c_test_tabdpt_vanilla)
                print_stats("TabDPT Train Simple", c_train_tabdpt_simple)
                print_stats("TabDPT Test Simple",  c_test_tabdpt_simple)
                if c_train_tabdpt_tuned_cox:
                    print_stats("TabDPT Train Tuned Cox", c_train_tabdpt_tuned_cox)
                    print_stats("TabDPT Test Tuned Cox",  c_test_tabdpt_tuned_cox)
                if c_train_tabdpt_tuned_rsf:
                    print_stats("TabDPT Train Tuned RSF", c_train_tabdpt_tuned_rsf)
                    print_stats("TabDPT Test Tuned RSF",  c_test_tabdpt_tuned_rsf)
                if c_train_cox:
                    print_stats("Cox Train", c_train_cox)
                    print_stats("Cox Test",  c_test_cox)
                if c_train_rsf:
                    print_stats("RSF Train", c_train_rsf)
                    print_stats("RSF Test",  c_test_rsf)

                all_train_tabdpt_vanilla.extend(c_train_tabdpt_vanilla)
                all_test_tabdpt_vanilla.extend(c_test_tabdpt_vanilla)
                all_train_tabdpt_simple.extend(c_train_tabdpt_simple)
                all_test_tabdpt_simple.extend(c_test_tabdpt_simple)
                all_train_tabdpt_tuned_cox.extend(c_train_tabdpt_tuned_cox)
                all_test_tabdpt_tuned_cox.extend(c_test_tabdpt_tuned_cox)
                all_train_tabdpt_tuned_rsf.extend(c_train_tabdpt_tuned_rsf)
                all_test_tabdpt_tuned_rsf.extend(c_test_tabdpt_tuned_rsf)
                all_train_cox.extend(c_train_cox)
                all_test_cox.extend(c_test_cox)
                all_train_rsf.extend(c_train_rsf)
                all_test_rsf.extend(c_test_rsf)

        print(f"\n  {'─'*50}")
        print(f"  TOTAL:")
        print_stats("TabDPT Train Vanilla", all_train_tabdpt_vanilla)
        print_stats("TabDPT Test Vanilla",  all_test_tabdpt_vanilla)
        print_stats("TabDPT Train Simple", all_train_tabdpt_simple)
        print_stats("TabDPT Test Simple",  all_test_tabdpt_simple)
        if len(all_train_tabdpt_tuned_cox) > 0:
            print_stats("TabDPT Train Tuned Cox", all_train_tabdpt_tuned_cox)
            print_stats("TabDPT Test Tuned Cox",  all_test_tabdpt_tuned_cox)
        if len(all_train_tabdpt_tuned_rsf) > 0:
            print_stats("TabDPT Train Tuned RSF", all_train_tabdpt_tuned_rsf)
            print_stats("TabDPT Test Tuned RSF",  all_test_tabdpt_tuned_rsf)
        print_stats("Cox Train",    all_train_cox)
        print_stats("Cox Test",     all_test_cox)
        print_stats("RSF Train",    all_train_rsf)
        print_stats("RSF Test",     all_test_rsf)
    sys.stdout = tee.console
    tee.close()
    print("✅ Result saved in 'results_cv_tabdpt.txt'")