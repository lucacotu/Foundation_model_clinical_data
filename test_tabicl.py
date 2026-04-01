import sys
import os
import argparse
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
from sklearn.impute import SimpleImputer


from src.data_loader import load_data
from src.preprocessing import clean_and_impute, prepare_cox_data_cv, prepare_cox_data_hurrah_cv
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

    preprocess = ["-1","NaN"]
    for preprocess_type in preprocess:
        df_tmp = df_tmp_original.copy()
        df_tmp_eval = df_tmp_eval_original.copy()
        match preprocess_type:
            case "drop":
                print("\n\nPreprocess: Drop NaN")
                df_tmp = df_tmp.dropna()
                df_tmp_eval = df_tmp_eval.dropna()
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

        c_index_train_scores_tabicl_vanilla = []
        c_index_test_scores_tabicl_vanilla = []
        c_index_train_scores_tabicl_tuned = []
        c_index_test_scores_tabicl_tuned = []
        c_index_train_scores_tabicl_simple = []
        c_index_test_scores_tabicl_simple = []
        c_index_train_scores_cox = []
        c_index_test_scores_cox = []
        c_index_test_scores_rsf = []
        c_index_train_scores_rsf = []

        X = df_tmp.drop(columns=['__event__', '__duration__'])
        y = df_tmp['__event__']
        t = df_tmp['__duration__'].values.astype(np.float32)

        X_eval = df_tmp_eval.drop(columns=['__event__', '__duration__'])
        y_eval = df_tmp_eval['__event__']
        t_eval = df_tmp_eval['__duration__'].values.astype(np.float32)

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = kf.split(X)
        
        for fold, (train_idx, test_idx) in enumerate(splits): 

            X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test   = y.iloc[train_idx], y.iloc[test_idx]
            t_train, t_test   = t[train_idx], t[test_idx]

            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 4. Generate the embeddings!
            print("Generating Tabicl Embeddings...")
            train_embeddings, test_embeddings = get_tabicl_embeddings(X_train, y_train.values, X_test, device=device, random_state=seed)
            _, eval_embeddings = get_tabicl_embeddings(X_train, y_train.values, X_eval, device=device, random_state=seed)
            print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")

            if tuning:
                os.makedirs("results/optuna/", exist_ok=True)
                log_name = f"optuna_cox_tabicl.log"
                log_file = os.path.join("results/optuna/", log_name)
                storage = JournalStorage(JournalFileBackend(log_file))

                study_name = "cox_tuning_tabicl_{fold}_{type}_seed{seed}".format(fold=fold+1, type=preprocess_type, seed=seed)
                study = optuna.create_study(
                    study_name=study_name,
                    direction="maximize",
                    storage=storage,
                    load_if_exists=True
                )

                params ={ 'embedding_dim':train_embeddings.shape[1],
                        'num_nodes': [128, 64],
                        'dropout': 0.1,
                        'learning_rate': 1e-3,
                        'batch_norm': True
                        }
            
                def objective(trial):
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

                    model.fit(train_embeddings, t_train, y_train.values,
                            epochs=100,
                            batch_size=128)
                    model.compute_baseline()
                    return model.concordance_index(eval_embeddings, t_eval, y_eval.values)

                study.optimize(objective, n_trials=100)
                params.update(study.best_params)
                print(f"Best parameters: {study.best_params}")
                
                cox = EmbeddingCoxPH(**params)

                callbacks = [tt.callbacks.EarlyStopping(patience=20,          
                                                        min_delta=1e-4,        
                                                        checkpoint_model=True, 
                                                        file_path="models/best_models_tabicl_tuned.pt", 
                                                        load_best=True
                                                        )]
                cox.fit(
                        train_embeddings,
                        durations=t_train,
                        events=y_train.values,
                        val_data=(eval_embeddings, t_eval, y_eval.values),
                        epochs=200,
                        callbacks=callbacks,
                        batch_size=128,
                    )

                cox.compute_baseline()

                c_train = cox.concordance_index(train_embeddings, t_train, y_train.values)
                c_test  = cox.concordance_index(test_embeddings, t_test, y_test.values)

                print(f"C-index train: {c_train:.4f}")
                print(f"C-index test:  {c_test:.4f}")
                c_index_train_scores_tabicl_tuned.append(c_train)
                c_index_test_scores_tabicl_tuned.append(c_test)
            
            
            #num_nodes    = [32, 32]
            #out_features = 1
            #batch_norm   = True
            #dropout      = 0.1
            
            # Simple CoxPH on the embeddings (without tuning)
            in_features  = train_embeddings.shape[1]
            net = nn.Linear(in_features, 1)

            model = CoxPH(net, tt.optim.Adam)
            model.optimizer.set_lr(0.01)

            batch_size = 256
            epochs     = 100
            callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path="models/best_models_tabicl_cox_simple.pt", 
                                    load_best=True
                                    )]

            log = model.fit(
                    train_embeddings, (t_train , y_train.values),
                    batch_size, epochs,
                    callbacks,
                    val_data=(eval_embeddings, (t_eval, y_eval.values)),
                    verbose=True
                )

            _ = model.compute_baseline_hazards()

            surv_test = model.predict_surv_df(test_embeddings)
            surv_train = model.predict_surv_df(train_embeddings)

            ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

            ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

            c_index_test_scores_tabicl_simple.append(ev_test.concordance_td())
            c_index_train_scores_tabicl_simple.append(ev_train.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())
            

            #MLP VANILLA on the embeddings (without tuning)
            in_features  = train_embeddings.shape[1]
            num_nodes    = [32, 32]
            out_features = 1
            batch_norm   = True
            dropout      = 0.1
            net = tt.practical.MLPVanilla(
                in_features, num_nodes, out_features,
                batch_norm=batch_norm, dropout=dropout
            )

            model = CoxPH(net, tt.optim.Adam)
            model.optimizer.set_lr(0.01)

            batch_size = 256
            epochs     = 100
            callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path="models/best_models_tabicl_cox_vanilla.pt", 
                                    load_best=True
                                    )]

            log = model.fit(
                    train_embeddings, (t_train , y_train.values),
                    batch_size, epochs,
                    callbacks,
                    val_data=(eval_embeddings, (t_eval, y_eval.values)),
                    verbose=True
                )

            _ = model.compute_baseline_hazards()

            surv_test = model.predict_surv_df(test_embeddings)
            surv_train = model.predict_surv_df(train_embeddings)

            ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

            ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

            c_index_test_scores_tabicl_vanilla.append(ev_test.concordance_td())
            c_index_train_scores_tabicl_vanilla.append(ev_train.concordance_td())

            print("C-index TRAIN:", ev_train.concordance_td())
            print("C-index TEST :", ev_test.concordance_td())
            

            if preprocess_type != "NaN":
                in_features  = X_train.shape[1]
                num_nodes    = [32, 32]
                out_features = 1
                batch_norm   = True
                dropout      = 0.1

                net = tt.practical.MLPVanilla(
                    in_features, num_nodes, out_features,
                    batch_norm=batch_norm, dropout=dropout
                )

                model = CoxPH(net, tt.optim.Adam)
                model.optimizer.set_lr(0.01)

                batch_size = 256
                epochs     = 100
                callbacks  = [tt.callbacks.EarlyStopping( 
                                    patience=20,          
                                    min_delta=1e-4,        
                                    checkpoint_model=True, 
                                    file_path="models/best_models_tabicl_cox_vanilla_nan.pt", 
                                    load_best=True
                                    )]

                log = model.fit(
                    X_train.values.astype(np.float32), (t_train , y_train.values),
                    batch_size, epochs,
                    callbacks,
                    val_data=(X_eval.values.astype(np.float32), (t_eval, y_eval.values)),
                    verbose=True
                )

                _ = model.compute_baseline_hazards()

                surv_test = model.predict_surv_df(X_test.values.astype(np.float32))
                surv_train = model.predict_surv_df(X_train.values.astype(np.float32))

                ev_train = EvalSurv(surv_train, t_train, y_train.values, censor_surv='km')

                ev_test = EvalSurv(surv_test, t_test, y_test.values, censor_surv='km')

                c_index_test_scores_cox.append(ev_test.concordance_td())
                c_index_train_scores_cox.append(ev_train.concordance_td())

                print("C-index TRAIN:", ev_train.concordance_td())
                print("C-index TEST :", ev_test.concordance_td())

                #RANDOM SURVIVAL FOREST
                rsf = RandomSurvivalForest(
                    n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=seed
                )

                y_train_structured = np.array(
                    [(bool(e), t) for e, t in zip(y_train, t_train)],
                    dtype=[('event', bool), ('time', float)]
                )

                y_test_structured = np.array(
                    [(bool(e), t) for e, t in zip(y_test, t_test)],
                    dtype=[('event', bool), ('time', float)]
                )

                rsf.fit(X_train.values.astype(np.float32), y_train_structured)
                
                c_index_test = rsf.score(X_test.values.astype(np.float32), y_test_structured)
                c_index_train = rsf.score(X_train.values.astype(np.float32), y_train_structured)
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
        results.append( (preprocess_type, 
                        (c_index_train_scores_tabicl_vanilla, c_index_test_scores_tabicl_vanilla),
                        (c_index_train_scores_tabicl_simple, c_index_test_scores_tabicl_simple),
                        (c_index_train_scores_tabicl_tuned, c_index_test_scores_tabicl_tuned), 
                        (c_index_train_scores_cox, c_index_test_scores_cox), 
                        (c_index_train_scores_rsf, c_index_test_scores_rsf)))
    return results


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
        res[1].append((seed, main("OrmoniTirodei", "Total mortality", "Follow Up Data", seed, tuning)))

    tee = Tee("results_cv_tabicl.txt")
    sys.stdout = tee

    for exp_idx, experiment in enumerate(res):
        print(f"\n{'='*60}")
        print(f"Dataset {exp_idx + 1}")
        print(f"{'='*60}")

        all_train_tabicl_vanilla = []
        all_test_tabicl_vanilla = []
        all_train_tabicl_simple = []
        all_test_tabicl_simple = []
        all_train_tabicl_tuned = []
        all_test_tabicl_tuned = []
        all_train_cox    = []
        all_test_cox     = []
        all_train_rsf    = []
        all_test_rsf     = []

        for seed, result_list in experiment:
            for preprocess_type, (c_train_tabicl_vanilla, c_test_tabicl_vanilla), (c_train_tabicl_simple, c_test_tabicl_simple), (c_train_tabicl_tuned, c_test_tabicl_tuned), (c_train_cox, c_test_cox), (c_train_rsf, c_test_rsf) in result_list:
                print(f"\n  Preprocess: {preprocess_type} | Seed: {seed}")
                print_stats("TabICL Train Vanilla", c_train_tabicl_vanilla)
                print_stats("TabICL Test Vanilla",  c_test_tabicl_vanilla)
                print_stats("TabICL Train Simple", c_train_tabicl_simple)
                print_stats("TabICL Test Simple",  c_test_tabicl_simple)
                if c_train_tabicl_tuned:
                    print_stats("TabICL Train Tuned", c_train_tabicl_tuned)
                    print_stats("TabICL Test Tuned",  c_test_tabicl_tuned)
                if c_train_cox:
                    print_stats("Cox Train", c_train_cox)
                    print_stats("Cox Test",  c_test_cox)
                if c_train_rsf:
                    print_stats("RSF Train", c_train_rsf)
                    print_stats("RSF Test",  c_test_rsf)

                all_train_tabicl_vanilla.extend(c_train_tabicl_vanilla)
                all_test_tabicl_vanilla.extend(c_test_tabicl_vanilla)
                all_train_tabicl_simple.extend(c_train_tabicl_simple)
                all_test_tabicl_simple.extend(c_test_tabicl_simple)
                all_train_tabicl_tuned.extend(c_train_tabicl_tuned)
                all_test_tabicl_tuned.extend(c_test_tabicl_tuned)
                all_train_cox.extend(c_train_cox)
                all_test_cox.extend(c_test_cox)
                all_train_rsf.extend(c_train_rsf)
                all_test_rsf.extend(c_test_rsf)

        print(f"\n  {'─'*50}")
        print(f"  TOTAL:")
        print_stats("TabICL Train Vanilla", all_train_tabicl_vanilla)
        print_stats("TabICL Test Vanilla",  all_test_tabicl_vanilla)
        print_stats("TabICL Train Simple", all_train_tabicl_simple)
        print_stats("TabICL Test Simple",  all_test_tabicl_simple)
        if len(all_train_tabicl_tuned) > 0:
            print_stats("TabICL Train Tuned", all_train_tabicl_tuned)
            print_stats("TabICL Test Tuned",  all_test_tabicl_tuned)
        print_stats("Cox Train",    all_train_cox)
        print_stats("Cox Test",     all_test_cox)
        print_stats("RSF Train",    all_train_rsf)
        print_stats("RSF Test",     all_test_rsf)
    sys.stdout = tee.console
    tee.close()
    print("✅ Result saved in 'results_cv_tabicl.txt'")