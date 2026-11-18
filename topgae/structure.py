"""Structural descriptors of the training graph (degree + HITS hub/authority)."""
import torch
from torch.nn import functional as F

from .utils import scatter_sum


@torch.no_grad()
def structural_features(edges, rows, cols, device, iterations=100):
    """Per-gene structural descriptors computed from training-positive edges.

    All source and target genes are mapped onto one unique-gene directed graph.
    Each gene gets ``[log1p out-degree, log1p in-degree, HITS hub, HITS
    authority]``, each column divided by its maximum across genes. HITS is a
    structural feature, not a claim of biological importance.

    Returns
    -------
    (row_features, col_features) : tensors of shape (n_sources, 4), (n_targets, 4)
    """
    genes = list(dict.fromkeys(list(rows) + list(cols)))
    lookup = {g: i for i, g in enumerate(genes)}
    ri = torch.tensor([lookup[g] for g in rows], device=device)
    ci = torch.tensor([lookup[g] for g in cols], device=device)
    src, dst = ri[edges[0]], ci[edges[1]]
    n = len(genes)
    one = torch.ones(src.numel(), device=device)
    outdeg, indeg = scatter_sum(one, src, n), scatter_sum(one, dst, n)
    h = torch.ones(n, device=device) / n ** 0.5
    a = torch.zeros_like(h)
    for _ in range(iterations):
        a = F.normalize(scatter_sum(h[src], dst, n), dim=0, eps=1e-12)
        new_h = F.normalize(scatter_sum(a[dst], src, n), dim=0, eps=1e-12)
        if torch.max(torch.abs(new_h - h)) < 1e-7:
            h = new_h
            break
        h = new_h
    a = F.normalize(scatter_sum(h[src], dst, n), dim=0, eps=1e-12)
    f = torch.stack([outdeg.log1p(), indeg.log1p(), h, a], dim=-1)
    f = f / f.amax(dim=0, keepdim=True).clamp_min(1e-12)
    return f[ri], f[ci]
