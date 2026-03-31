import torch
import torch.nn as nn
import numpy as np
from torchtuples import practical
import torchtuples as tt
from pycox.models import CoxPH
import pandas as pd
from pycox.evaluation import EvalSurv
from torchtuples import callbacks as cb

class MLPVanilla(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_nodes: list,
        out_features: int,
        batch_norm: bool = True,
        dropout: float = None,
        activation=nn.ReLU,
        output_activation=None,
        output_bias: bool = True,
    ):
        super().__init__()

        nodes = [in_features] + list(num_nodes)
        layers = []

        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))

        # Layer di output — NO batch norm, NO dropout
        out_layer = nn.Linear(nodes[-1], out_features, bias=output_bias)
        nn.init.kaiming_normal_(out_layer.weight, nonlinearity='relu')
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

        # Init pesi layer nascosti
        self._init_weights()

    def _init_weights(self):
        layers = list(self.net.children())
        output_layer = layers[-1] if not isinstance(layers[-1], nn.Module.__class__) else None

        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear) and m is not list(self.net.children())[-1]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        #for m in self.net:
        #    if isinstance(m, nn.Linear):
        #        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        #        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingCoxPH:
    def __init__(
        self,
        embedding_dim: int,
        num_nodes: list = [64, 64],
        out_features: int = 1,
        batch_norm: bool = True,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0
    ):
        net = MLPVanilla(
            in_features=embedding_dim,
            num_nodes=num_nodes,
            out_features=out_features,
            batch_norm=False,#batch_norm,
            dropout=dropout,
            output_activation=None,  
        )

        self.model = CoxPH(net, tt.optim.Adam(lr=learning_rate))
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self._x_train = None
        self._y_train = None

    def fit(
        self,
        embeddings: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: tuple = None,
        epochs: int = 100,
        batch_size: int = 256,
        callbacks=None,
        verbose: bool = True,
    ):
        x = embeddings.astype(np.float32)
        y = (durations.astype(np.float32), events.astype(np.int32))

        self._x_train = x
        self._y_train = y

        val = None
        if val_data is not None:
            x_val, dur_val, ev_val = val_data
            val = (
                x_val.astype(np.float32),
                (dur_val.astype(np.float32), ev_val.astype(np.float32)),
            )
        
        callback_list = callbacks or []

        if self.early_stopping:
            if val is None:
                raise ValueError(
                    "val_data it's missing"
                )
            callback_list = callback_list + [
                cb.EarlyStopping(
                    patience=self.patience,          
                    min_delta=self.min_delta,        
                    checkpoint_model=True, 
                    file_path="models/best_models.pt", 
                    load_best=True
                ),
            ]

        self.log = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callback_list,
            verbose=verbose,
            val_data=val,
        )

        return self

    def compute_baseline(self):
        if self._x_train is None:
            raise RuntimeError("Call fit() before compute_baseline().")


        self.model.compute_baseline_hazards(
            input=self._x_train,
            target=self._y_train,
        )
        return self

    def predict_survival(self, embeddings: np.ndarray) -> "pd.DataFrame":
        """
        Restituisce un DataFrame (timepoints × soggetti) con le survival
        probabilities. Valori attesi: float in (0, 1].
        """
        if self.model.baseline_hazards_ is None:
            raise RuntimeError("Chiama compute_baseline() prima di predict_survival().")

        x = embeddings.astype(np.float32)
        return self.model.predict_surv_df(x)
    

    def concordance_index(
        self,
        embeddings: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        method: str = "antolini",
    ) -> float:
        """
        Calcola il concordance index sul set fornito.

        Parameters
        ----------
        embeddings : np.ndarray
            Feature matrix (n_samples, embedding_dim)
        durations : np.ndarray
            Tempi di osservazione
        events : np.ndarray
            Event indicator (1 = evento, 0 = censurato)
        method : str
            'antolini' per il C-index time-dependent (default),
            oppure altri metodi supportati da EvalSurv.

        Returns
        -------
        float
            Concordance index in [0, 1]. 0.5 = random, 1.0 = perfetto.
        """
        surv_df = self.predict_survival(embeddings)

        ev = EvalSurv(
            surv=surv_df,
            durations=durations,
            events=events,
            censor_surv="km",
        )

        return ev.concordance_td(method=method)