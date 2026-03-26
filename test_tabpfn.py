import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
import torch
import random


from src.data_loader import load_and_merge_data,load_dataset
from src.preprocessing import clean_and_impute, prepare_cox_data, prepare_cox_data_hurrah, preprocess_data,prepare_cox_data_cv, prepare_cox_data_hurrah_cv
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

def main(seed):
    # 1. Load and clean the data
    data = load_and_merge_data("Dataset Sirbu")
    df_main = clean_and_impute(data)

    # 2. Extract specific features and targets (e.g. Mortality data)
    # And split into Train, Eval and Test sets
    #df_mortality_train, df_mortality_eval, df_mortality_test = prepare_data(df)
    df_mortality_train, df_mortality_eval = prepare_cox_data_cv(df_main)

    X_eval = df_mortality_eval.drop(columns=["Follow Up Data", "Total mortality"])
    y_eval = df_mortality_eval["Total mortality"]
    t_eval = df_mortality_eval["Follow Up Data"].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    c_index_train_scores = []
    c_index_test_scores = []

    X = df_mortality_train.drop(columns=["Follow Up Data", "Total mortality"])
    y = df_mortality_train["Total mortality"]
    t = df_mortality_train["Follow Up Data"].values.astype(np.float32)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):

        X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test   = y.iloc[train_idx], y.iloc[test_idx]
        t_train, t_test   = t[train_idx], t[test_idx]

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
        # 4. Generate the embeddings!
        print("Generating TabPFN Embeddings...")
        train_embeddings, test_embeddings = get_tabpfn_embeddings(X_train, y_train, X_test, y_test,seed)
        _, eval_embeddings = get_tabpfn_embeddings(X_train, y_train, X_eval, y_eval,seed)
        print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")

        cox = EmbeddingCoxPH(
            embedding_dim=train_embeddings.shape[1],
            num_nodes=[128, 64],
            dropout=0.1,
            learning_rate=1e-3,
            early_stopping=True,
            patience=20,
            min_delta=1e-4,
        )

        cox.fit(
            train_embeddings,
            durations=t_train,
            events=y_train.values,
            val_data=(eval_embeddings, t_eval, y_eval.values),
            epochs=200,
            batch_size=128,
        )

        cox.compute_baseline()

        c_train = cox.concordance_index(train_embeddings, t_train, y_train.values)
        c_test  = cox.concordance_index(test_embeddings, t_test, y_test.values)

        print(f"C-index train: {c_train:.4f}")
        print(f"C-index test:  {c_test:.4f}")
        c_index_train_scores.append(c_train)
        c_index_test_scores.append(c_test)

        # ── Predict ──────────────────────────────────────────────────────────────
        #survival_df = cox.predict_survival(test_embeddings)
        
        #print("Predicted survival probabilities for test set:")
        #print(survival_df)

        #print("Baseline Hazard: ",cox.model.baseline_hazards_)

        # Check the distribution of the predicted logits (risk scores)
        #logits = cox.model.predict(test_embeddings)
        #print(f"min: {logits.min():.3f}, max: {logits.max():.3f}, std: {logits.std():.3f}")
        # If std ≈ 0 → the model has not learned anything (lr too high/low, not enough epochs, etc.)
        
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
        
        save_name_base = "tabpfn_mortality_tsne_seed"+ str(seed) + "_fold" + str(fold + 1)
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
    return (c_index_train_scores, c_index_test_scores)

def main2(seed):

    df = load_dataset("Dataset Sirbu")
    df = preprocess_data(df)
    
    # 2. Extract specific features and targets (e.g. Mortality data)
    # And split into Train, Eval and Test sets
    df_mortality_train, df_mortality_eval = prepare_cox_data_hurrah_cv(df)

    X_eval = df_mortality_eval.drop(columns=["FU", "STATO_AL_FU"])
    y_eval = df_mortality_eval["STATO_AL_FU"]
    t_eval = df_mortality_eval["FU"].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    c_index_train_scores = []
    c_index_test_scores = []

    X = df_mortality_train.drop(columns=["FU", "STATO_AL_FU"])
    y = df_mortality_train["STATO_AL_FU"]
    t = df_mortality_train["FU"].values.astype(np.float32)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test   = y.iloc[train_idx], y.iloc[test_idx]
        t_train, t_test   = t[train_idx], t[test_idx]


        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
        # 4. Generate the embeddings!
        print("Generating TabPFN Embeddings...")
        train_embeddings, test_embeddings = get_tabpfn_embeddings(X_train, y_train, X_test, y_test,seed)
        _, eval_embeddings = get_tabpfn_embeddings(X_train, y_train, X_eval, y_eval,seed)
        print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")

        cox = EmbeddingCoxPH(
            embedding_dim=train_embeddings.shape[1],
            num_nodes=[128, 64],
            dropout=0.1,
            learning_rate=1e-3,
            early_stopping=True,
            patience = 20,
            min_delta=1e-4,
        )

        cox.fit(
            train_embeddings,
            durations=t_train,
            events=y_train.values,
            val_data=(eval_embeddings, t_eval, y_eval.values),
            epochs=200,
            batch_size=128,
        )

        cox.compute_baseline()

        c_train = cox.concordance_index(train_embeddings, t_train, y_train.values)
        c_test  = cox.concordance_index(test_embeddings, t_test, y_test.values)

        print(f"C-index train: {c_train:.4f}")
        print(f"C-index test:  {c_test:.4f}")
        c_index_train_scores.append(c_train)
        c_index_test_scores.append(c_test)

        # ── Predict ──────────────────────────────────────────────────────────────
        #survival_df = cox.predict_survival(test_embeddings)
        
        #print("Predicted survival probabilities for test set:")
        #print(survival_df)

        #print("Baseline Hazard: ",cox.model.baseline_hazards_)

        # Check the distribution of the predicted logits (risk scores)
        #logits = cox.model.predict(test_embeddings)
        #print(f"min: {logits.min():.3f}, max: {logits.max():.3f}, std: {logits.std():.3f}")
        # If std ≈ 0 → the model has not learned anything (lr too high/low, not enough epochs, etc.)
    
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
        
        save_name_base = "tabpfn_mortality_tsne_HURRAH_seed" + str(seed) + "_fold" + str(fold + 1)
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
        
    return (c_index_train_scores, c_index_test_scores)

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
    print("STARTED")
    seeds = [42, 123, 456, 789, 2024]
    res = [[],[]]
    for index, seed in enumerate(seeds): 
        set_seed(seed)

        res[0].append((seed,main(seed)))
        res[1].append((seed,main2(seed)))

    tee = Tee("results_cv.txt")
    sys.stdout = tee

    for exp_idx, experiment in enumerate(res):
        print(f"\n{'='*60}")
        print(f"Dataset {exp_idx + 1}")
        print(f"{'='*60}")

        all_train = []
        all_test  = []

        for seed, (c_train_list, c_test_list) in experiment:
            print(f"\n  Seed {seed}:")
            print_stats("Train", c_train_list)
            print_stats("Test",  c_test_list)

            all_train.extend(c_train_list)
            all_test.extend(c_test_list)

        print(f"\n  {'─'*50}")
        print(f" Total: ")
        print_stats("Train", all_train)
        print_stats("Test",  all_test)
    sys.stdout = tee.console
    tee.close()
    print("✅ Result saved in 'results_cv.txt'")
    
    
    '''for experiment in res:
    all_means = []
    
    for fold_idx, c_train_list, c_eval_list in experiment:
        means = [np.mean([t, e]) for t, e in zip(c_train_list, c_eval_list)]
        all_means.append(means)
    
    all_means = np.array(all_means)  # shape: (n_folds, n_valori)
    
    print("\nMedia globale per colonna:")
    for i, col_mean in enumerate(all_means.mean(axis=0)):
        print(f"  Colonna {i}: {col_mean:.4f}")
    '''
    #print(res)
    #for idx, data in enumerate(res):
    #    for r in data:
    #        print(r[0][0] ": ")
    #        for train, test in r:
    #            print("TRAIN: ", train)
    #            prinnt("TEST: ", test)


    '''print(f"\n{'='*40}")
    print(f"Mean C-index train  : {np.mean(res[0][0]):.4f}")
    print(f"Std train           : {np.std(res[0][0]):.4f}")
    print(f"Min / Max train     : {np.min(res[0][0]):.4f} / {np.max(res[0][0]):.4f}")

    print(f"\n{'='*40}")
    print(f"Mean C-index  test : {np.mean(res[0][1]):.4f}")
    print(f"Std test           : {np.std(res[0][1]):.4f}")
    print(f"Min / Max test     : {np.min(res[0][1]):.4f} / {np.max(res[0][1]):.4f}")

    print(f"\n{'='*40}")
    print(f"Mean C-index train HURRAH  : {np.mean(res[1][0]):.4f}")
    print(f"Std train HURRAH           : {np.std(res[1][0]):.4f}")
    print(f"Min / Max train HURRAH     : {np.min(res[1][0]):.4f} / {np.max(res[1][0]):.4f}")

    print(f"\n{'='*40}")
    print(f"Mean C-index test HURRAH  : {np.mean(res[1][1]):.4f}")
    print(f"Std test HURRAH           : {np.std(res[1][1]):.4f}")
    print(f"Min / Max test HURRAH     : {np.min(res[1][1]):.4f} / {np.max(res[1][1]):.4f}")'''