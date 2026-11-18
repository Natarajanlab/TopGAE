"""Masked-expression corruption and reconstruction loss."""
import torch
from torch.nn import functional as F


def mask_expression(X_row, X_col, row_genes, col_genes, ratio=0.2):
    """Zero out a random ``ratio`` of unique genes consistently in BOTH roles.

    A gene masked as a source is also masked as a target, so it cannot be
    recovered through its other role.

    Returns
    -------
    masked_row, masked_col, row_mask, col_mask
    """
    if not 0 < ratio < 1:
        raise ValueError("mask_ratio must lie strictly between zero and one.")
    genes = list(dict.fromkeys(list(row_genes) + list(col_genes)))
    lookup = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    selected = torch.zeros(n, dtype=torch.bool, device=X_row.device)
    k = min(n, max(1, round(ratio * n)))
    selected[torch.randperm(n, device=X_row.device)[:k]] = True
    mr = selected[torch.tensor([lookup[g] for g in row_genes], device=X_row.device)]
    mc = selected[torch.tensor([lookup[g] for g in col_genes], device=X_row.device)]
    return X_row.masked_fill(mr[:, None], 0), X_col.masked_fill(mc[:, None], 0), mr, mc


def masked_expression_loss(pred_row, pred_col, X_row, X_col, mr, mc,
                           row_genes, col_genes, kind="mse"):
    """Reconstruction loss over masked genes only.

    ``kind="mse"`` is the mean squared error; ``kind="cosine"`` is the squared
    cosine error (used for the paper results), falling back to MSE for
    all-zero target profiles. Each unique masked gene has equal weight, even
    when it is present in both roles.
    """
    def per_gene(pred, target):
        if kind == "mse":
            return (pred - target).square().mean(-1)
        if kind == "cosine":
            # Zero expression profiles have no cosine direction; use MSE for them.
            cosine = (1 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)).clamp_min(0).square()
            return torch.where(target.norm(dim=-1) > 1e-8, cosine,
                               (pred - target).square().mean(-1))
        raise ValueError("reconstruction_loss must be mse or cosine.")
    lr, lc = per_gene(pred_row, X_row), per_gene(pred_col, X_col)
    shared = set(row_genes).intersection(col_genes)
    wr = lr.new_tensor([0.5 if g in shared else 1.0 for g in row_genes])
    wc = lc.new_tensor([0.5 if g in shared else 1.0 for g in col_genes])
    return ((lr[mr] * wr[mr]).sum() + (lc[mc] * wc[mc]).sum()) / (wr[mr].sum() + wc[mc].sum()).clamp_min(1)
