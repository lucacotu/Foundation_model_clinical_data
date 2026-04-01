"""survpfn.models.tabicl — TabICL embedding extraction and survival wrappers."""
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

from typing import Optional

from src.tabicl.embedding import TabICLEmbeddingExtractor
import numpy as np

from src.tabicl.tabicl.sklearn.preprocessing import TransformToNumerical


def get_tabicl_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: str = "cpu",
    random_state: int = 42,
    model_path: Optional[str] = None,
    checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frozen TabICL embeddings for train and test sets.

    Parameters
    ----------
    X_train, X_test  : raw (unscaled) feature arrays
    y_train          : binary event labels (0/1)
    device           : torch device
    model_path       : local checkpoint path; None = auto-download from HuggingFace
    context_size     : max training samples used as ICL context (default 1000)
    hook_point       : ``"post_icl"`` (default) or ``"pre_icl"``

    Returns
    -------
    train_emb : (n_train, emb_dim)
    test_emb  : (n_test,  emb_dim)
    """
    extractor = TabICLEmbeddingExtractor(
        device=device,
        model_path=model_path,
        checkpoint_version=checkpoint_version,
    )

    # Preprocess with l'imputer di TabICL
    encoder = TransformToNumerical()
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    emb = extractor._embed_batch(X_train_enc, y_train, X_test_enc)
    train_emb = emb[:, :X_train.shape[0]].squeeze(0)
    test_emb  = emb[:, X_train.shape[0]:].squeeze(0)
    return train_emb, test_emb
    