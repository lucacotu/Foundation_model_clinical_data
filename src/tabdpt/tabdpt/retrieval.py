import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class FAISS:
    """Nearest-neighbor search backed by sklearn, replacing faiss for CPU portability."""

    def __init__(
        self, X: np.ndarray, use_hnsw: bool = False, hnsw_m: int = 32, metric: str = "L2"
    ) -> None:
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = np.ascontiguousarray(X, dtype=np.float32)

        if metric == "L2":
            sk_metric = "euclidean"
        elif metric == "IP":
            sk_metric = "cosine"
        else:
            raise NotImplementedError(f"Unsupported metric: {metric}")

        self.index = NearestNeighbors(metric=sk_metric, algorithm="auto")
        self.index.fit(np.nan_to_num(X, nan=0.0))

    def get_knn_indices(self, queries: np.ndarray | torch.Tensor, k: int) -> np.ndarray:
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        assert isinstance(k, int)

        _, indices = self.index.kneighbors(np.nan_to_num(queries, nan=0.0), n_neighbors=k)
        return indices
