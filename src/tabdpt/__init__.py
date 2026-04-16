"""survpfn.models.tabdpt — TabDPT embedding extraction and survival wrappers."""
import os
import numpy as np
from typing import Optional
import sys

import torch
from src.tabdpt.embedding import TabDPTEmbeddingExtractor

def get_tabdpt_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    checkpoint_path: Optional[str] = None,
    context_size: int = 128,
    use_retrieval: bool = True,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frozen TabDPT embeddings for train and test sets.

    Parameters
    ----------
    X_train, X_test : np.ndarray  — raw (unscaled) features
    y_train         : np.ndarray  — binary event labels (0=censored, 1=event)
    checkpoint_path : str | None  — path to TabDPT checkpoint;
                                    falls back to TABDPT_CHECKPOINT env var
    device          : str         — torch device

    Returns
    -------
    train_emb : (n_train, ninp)
    test_emb  : (n_test,  ninp)
    """
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("TABDPT_CHECKPOINT", "")
        print("ENV CHECKPOINT PATH:", checkpoint_path, file=sys.stderr)
    if not checkpoint_path:
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src/models_diff/tabdpt1_1.pth")
        print("DEFAULT PATH:", default_path, file=sys.stderr)
        if os.path.exists(default_path):
            checkpoint_path = default_path
        else:
            # Absolute path fallback from user
            fallback_abs = "/home/lcotugno/Foundation_model_clinical_data/src/models_diff/tabdpt1_1.pth"
            if os.path.exists(fallback_abs):
                checkpoint_path = fallback_abs

    if not checkpoint_path:
        raise ValueError(
            "No TabDPT checkpoint found.  Pass checkpoint_path= or set "
            "the TABDPT_CHECKPOINT environment variable."
        )


    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(type(checkpoint))  # deve essere un dict, non un errore

    extractor = TabDPTEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
        context_size=context_size,
        use_retrieval=use_retrieval,
    )
    extractor.fit(X_train, y_train)

    train_emb, test_emb = extractor._embed_all(X_test)
    return train_emb, test_emb
