"""Input preparation: prior network -> adjacency, expression -> role tensors."""
import warnings

import numpy as np
import pandas as pd
import torch

from .utils import unique_names


def build_adjacency(df, row_genes=None, col_genes=None):
    """Build a binary source x target adjacency DataFrame from an edge list.

    Parameters
    ----------
    df : pandas.DataFrame
        POSITIVE interactions only, with columns ``Gene1`` (source / TF) and
        ``Gene2`` (target).
    row_genes, col_genes : list of str, optional
        Source and target gene universes. Pass the benchmark TF / gene lists
        explicitly to keep genes that have no labelled edges. Edges outside the
        supplied universe are dropped; self-pairs are always excluded.

    Returns
    -------
    adj : pandas.DataFrame (float32, 0/1), rows : list, cols : list
    """
    if row_genes is None or col_genes is None:
        warnings.warn("Inferring missing gene lists from positives changes the candidate universe; "
                      "prefer explicit benchmark TF and gene lists.", stacklevel=2)
    rows = unique_names(sorted(df.Gene1.unique()) if row_genes is None else row_genes)
    cols = unique_names(sorted(df.Gene2.unique()) if col_genes is None else col_genes)
    adj = pd.DataFrame(0.0, index=rows, columns=cols, dtype=np.float32)
    ri, ci = {g: i for i, g in enumerate(rows)}, {g: i for i, g in enumerate(cols)}
    for a, b in df[["Gene1", "Gene2"]].itertuples(index=False, name=None):
        if a in ri and b in ci and a != b:
            adj.iat[ri[a], ci[b]] = 1.0
    return adj, rows, cols


def prepare_inputs(expression, adj, row_genes, col_genes, device="cpu", normalization="cell"):
    """Turn a genes x cells expression table into TopGAE input tensors.

    Parameters
    ----------
    expression : pandas.DataFrame
        Genes x cells matrix with unique gene IDs as the index (NOT an AnnData
        object; see ``run_topgae.py`` for AnnData input).
    adj : pandas.DataFrame
        Adjacency returned by :func:`build_adjacency`.
    row_genes, col_genes : list of str
        Source-role and target-role genes.
    normalization : {"cell", "gene", "none"}
        ``"cell"`` (paper default) standardises each cell across genes,
        ``"gene"`` standardises each gene across cells and ``"none"`` keeps the
        supplied values. No implicit log transform is applied. Unique genes are
        normalised once and then extracted for both roles.

    Returns
    -------
    X_row : (n_sources, n_cells) tensor, X_col : (n_targets, n_cells) tensor,
    A : (n_sources, n_targets) binary tensor
    """
    rows, cols = unique_names(row_genes), unique_names(col_genes)
    if not expression.index.is_unique:
        raise ValueError("Expression gene IDs must be unique.")
    genes = list(dict.fromkeys(rows + cols))
    x = expression.loc[genes].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all() or x.shape[1] == 0:
        raise ValueError("Expression must have finite values and at least one cell.")
    if normalization in ("cell", "gene"):
        axis = 0 if normalization == "cell" else 1
        x = (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True).clip(1e-6)
    elif normalization != "none":
        raise ValueError("normalization must be cell, gene, or none.")
    lookup = {g: i for i, g in enumerate(genes)}
    xr = torch.as_tensor(x[[lookup[g] for g in rows]], device=device)
    xc = torch.as_tensor(x[[lookup[g] for g in cols]], device=device)
    a = torch.as_tensor(adj.loc[rows, cols].to_numpy(dtype=np.float32, copy=True), device=device)
    return xr, xc, a
