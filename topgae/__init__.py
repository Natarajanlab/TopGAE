"""TopGAE: a topology-informed regulatory influence-weighted graph autoencoder
for gene regulatory network reconstruction from single-cell transcriptomics.

Typical use::

    from topgae import (build_adjacency, prepare_inputs, make_splits,
                        seed_everything, TopGAE, train_topgae)

    adj, rows, cols = build_adjacency(net, genes, genes)          # prior network
    X_row, X_col, A = prepare_inputs(expr, adj, rows, cols)        # genes x cells
    seed_everything(42)
    splits = make_splits(A, rows, cols, seed=42, split_mode="gaedgrn")
    model = TopGAE(in_features=X_row.shape[1], out_feature=64, latent_dim=32)
    S1, S2, A_hat, metrics = train_topgae(model, X_row, X_col, A, rows, cols,
                                          splits=splits, seed=42)

Requires Python >= 3.10, PyTorch >= 2.0, NumPy, pandas and scikit-learn
(no torch_geometric / torch_scatter dependency).
"""
from .data import build_adjacency, prepare_inputs
from .layers import DualScoreLayer
from .losses import mask_expression, masked_expression_loss
from .model import TopGAE
from .splits import gaedgrn_pairs, make_splits, validate_splits
from .structure import structural_features
from .train import link_metrics, train_topgae
from .utils import seed_everything

__version__ = "1.0.0"

__all__ = [
    "TopGAE",
    "DualScoreLayer",
    "train_topgae",
    "build_adjacency",
    "prepare_inputs",
    "make_splits",
    "validate_splits",
    "gaedgrn_pairs",
    "structural_features",
    "mask_expression",
    "masked_expression_loss",
    "link_metrics",
    "seed_everything",
]
