"""Small shared helpers: seeding, gene-list checks and scatter operations.

The scatter helpers replace torch_scatter / torch_geometric so that TopGAE
only depends on plain PyTorch.
"""
import random

import numpy as np
import torch


def seed_everything(seed=42):
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Call this BEFORE constructing a model as well as before training, so that
    weight initialisation is reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_names(names):
    """Return ``names`` as a list after checking it is nonempty and duplicate-free."""
    names = list(names)
    if len(names) != len(set(names)) or not names:
        raise ValueError("Gene lists must be nonempty and contain unique IDs.")
    return names


def scatter_sum(values, index, n):
    """Sum rows of ``values`` into ``n`` buckets given by ``index``."""
    return values.new_zeros((n,) + values.shape[1:]).index_add(0, index, values)


def scatter_softmax(scores, index, n):
    """Numerically stable softmax of ``scores`` within each group of ``index``."""
    if scores.numel() == 0:
        return scores
    maxima = scores.new_full((n,), -torch.inf)
    maxima.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
    exp = (scores - maxima[index]).exp()
    return exp / scatter_sum(exp, index, n)[index].clamp_min(1e-12)
