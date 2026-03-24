import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


from src.data_loader import load_and_merge_data,load_dataset
from src.preprocessing import clean_and_impute, prepare_cox_data, preprocess_data
from src.tabpfn import (
	get_tabpfn_embeddings,
	setup_figure, create_savefig_partial
)
from src.embedding_cox import EmbeddingCoxPH

def main():
    # 1. Load and clean the data
    data = load_and_merge_data("Dataset Sirbu")
    df_main = clean_and_impute(data)

    # 2. Extract specific features and targets (e.g. Mortality data)
    # And split into Train, Eval and Test sets
    df_mortality_train, df_mortality_eval, df_mortality_test = prepare_cox_data(df_main)
    
    X_train = df_mortality_train.drop(columns=["Follow Up Data", "Total mortality"])
    y_train = df_mortality_train["Total mortality"]
    t_train = df_mortality_train["Follow Up Data"].values.astype(np.float32)
    
    X_test = df_mortality_test.drop(columns=["Follow Up Data", "Total mortality"])
    y_test = df_mortality_test["Total mortality"]
    t_test = df_mortality_test["Follow Up Data"].values.astype(np.float32)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # 4. Generate the embeddings!
    print("Generating TabPFN Embeddings...")
    train_embeddings, test_embeddings = get_tabpfn_embeddings(X_train, y_train[0:20], X_test[0:20], y_test[0:20])
    print(f"Successfully generated embeddings with shape: {train_embeddings.shape}")

    cox = EmbeddingCoxPH(
        embedding_dim=train_embeddings.shape[1],
        num_nodes=[128, 64],
        dropout=0.1,
        learning_rate=1e-3,
    )

    cox.fit(
        train_embeddings,
        durations=t_train[0:20],
        events=y_train.values[0:20],
        epochs=200,
        batch_size=128,
    )

    cox.compute_baseline()

    c_train = cox.concordance_index(train_embeddings, t_train[0:20], y_train.values[0:20])
    c_test  = cox.concordance_index(test_embeddings, t_test[0:20], y_test.values[0:20])

    print(f"C-index train: {c_train:.4f}")
    print(f"C-index test:  {c_test:.4f}")


    # ── Predict ──────────────────────────────────────────────────────────────
    survival_df = cox.predict_survival(test_embeddings)
    
    print("Predicted survival probabilities for test set:")
    print(survival_df)

    print("Baseline Hazard: ",cox.model.baseline_hazards_)

    # Check the distribution of the predicted logits (risk scores)
    logits = cox.model.predict(test_embeddings)
    print(f"min: {logits.min():.3f}, max: {logits.max():.3f}, std: {logits.std():.3f}")
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
        mask = (y_train[0:200].values == cls)
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
    
    save_name_base = "tabpfn_mortality_tsne"
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

def main2():
    df = load_dataset("Dataset Sirbu")
    print(df.head())
    df = preprocess_data(df)


if __name__ == "__main__":
    main2()