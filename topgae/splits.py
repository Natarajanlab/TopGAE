"""Train / validation / test edge partitioning with balanced, fixed negatives."""
import numpy as np
import torch

from .utils import unique_names


def gaedgrn_pairs(adjacency, seed=42, val_fraction=0.05, test_fraction=0.10):
    """GAEDGRN-like global split on a square gene x gene adjacency.

    Returns ``[(train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg)]``
    as arrays of (row, col) index pairs.

    Positives: floor(5% E) validation, floor(10% E) test, remainder train.
    Negatives: drawn uniformly without replacement from all off-diagonal absent
    directed gene-gene pairs, excluding ALL known positives. Test negatives are
    reserved first, then validation, then training. All three splits are
    balanced and disjoint.
    """
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("gaedgrn requires a square gene x gene adjacency.")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("Adjacency must be binary.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Invalid validation/test fractions.")
    # Keep both directions distinct. Ignore diagonal labels, as the reference does.
    valid = ~np.eye(a.shape[0], dtype=bool)
    positives = np.argwhere((a == 1) & valid)
    n_val = int(np.floor(len(positives) * val_fraction))
    n_test = int(np.floor(len(positives) * test_fraction))
    if min(n_val, n_test, len(positives) - n_val - n_test) < 1:
        raise ValueError("Too few positives for nonempty GAEDGRN floor-based splits; "
                         "at defaults at least 20 non-self positive edges are required.")
    # Legacy RNG reproduces the reference positive shuffle for the same initial
    # state, ordered adjacency and seed.
    rng = np.random.RandomState(seed)
    order = np.arange(len(positives))
    rng.shuffle(order)
    val_pos = positives[order[:n_val]]
    test_pos = positives[order[n_val:n_val + n_test]]
    train_pos = positives[np.delete(np.arange(len(positives)), order[:n_val + n_test])]
    negatives = np.argwhere((a == 0) & valid)
    if len(negatives) < len(positives):
        raise ValueError("Insufficient unique negatives for balanced disjoint sampled training "
                         "and evaluation. This trainer does not use full-matrix reconstruction.")
    selected = negatives[rng.choice(len(negatives), size=len(positives), replace=False)]
    test_neg = selected[:n_test]
    val_neg = selected[n_test:n_test + n_val]
    train_neg = selected[n_test + n_val:]
    return [(train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg)]


def make_splits(A, row_genes, col_genes, seed=42, split_mode="gaedgrn",
                val_fraction=0.05, test_fraction=0.10):
    """Split the positive edges of ``A`` and pair them with fixed sampled negatives.

    Returns six tensors ``(train_edges, train_labels, val_edges, val_labels,
    test_edges, test_labels)``; each ``*_edges`` is a ``(2, n)`` long tensor of
    (source index, target index) and each ``*_labels`` a float 0/1 tensor.

    split_mode
        ``"gaedgrn"``  global floor-based positive split; negatives from all
                       ordered gene x gene pairs. Requires identical, identically
                       ordered ``row_genes`` and ``col_genes`` (paper benchmark
                       protocol for BEELINE, dyngen and SERGIO).
        ``"global"``   global positive split; negatives sampled uniformly from
                       source x target pairs (used for the perturbation analyses,
                       where sources are the TFs of the prior network).
        ``"per_tf"``   degree-stratified positives and balanced negatives within
                       each source gene.
    """
    rows, cols = unique_names(row_genes), unique_names(col_genes)
    if A.shape != (len(rows), len(cols)) or not torch.all((A == 0) | (A == 1)):
        raise ValueError("A must be binary and match the gene lists.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Invalid split fractions.")
    rng = np.random.default_rng(seed)
    a = A.detach().cpu().numpy()
    valid = np.array([[r != c for c in cols] for r in rows])
    splits = [[], [], []]

    def allocate(pos, neg):
        pos, neg = rng.permutation(pos), rng.permutation(neg)
        n = len(pos)
        if n == 0:
            return
        # Keep a training edge for low-degree sources; expose changed proportions.
        if n < 3:
            nv, nt = (1 if n == 2 else 0), 0
        else:
            nv = max(1, int(n * val_fraction))
            nt = max(1, int(n * test_fraction))
            while nv + nt >= n:
                if nt >= nv and nt > 1:
                    nt -= 1
                else:
                    nv -= 1
        parts = [pos[nv + nt:], pos[:nv], pos[nv:nv + nt]]
        if len(neg) < n:
            raise ValueError("Not enough unique non-self negatives for balanced disjoint splits. "
                             "Use a larger candidate universe or a different sampling protocol.")
        offset = 0
        for dest, p in zip(splits, parts):
            q = neg[offset:offset + len(p)]
            offset += len(p)
            dest.extend((int(i), int(j), 1.0) for i, j in p)
            dest.extend((int(i), int(j), 0.0) for i, j in q)

    if split_mode == "gaedgrn":
        if rows != cols:
            raise ValueError("gaedgrn requires identical ordered row_genes/col_genes; "
                             "build_adjacency(net, all_genes, all_genes).")
        for dest, (p, q) in zip(splits, gaedgrn_pairs(a, seed, val_fraction, test_fraction)):
            dest.extend((int(i), int(j), 1.0) for i, j in p)
            dest.extend((int(i), int(j), 0.0) for i, j in q)
    elif split_mode == "global":
        allocate(np.argwhere((a == 1) & valid), np.argwhere((a == 0) & valid))
    elif split_mode == "per_tf":
        for i in range(len(rows)):
            p = np.flatnonzero((a[i] == 1) & valid[i])
            q = np.flatnonzero((a[i] == 0) & valid[i])
            allocate(np.column_stack([np.full(len(p), i), p]),
                     np.column_stack([np.full(len(q), i), q]))
    else:
        raise ValueError("split_mode must be global, per_tf, or gaedgrn.")
    output = []
    for s in splits:
        arr = np.asarray(s, dtype=np.float64).reshape(-1, 3)
        output.extend([torch.tensor(arr[:, :2].T, dtype=torch.long, device=A.device),
                       torch.tensor(arr[:, 2], dtype=torch.float32, device=A.device)])
    validate_splits(tuple(output), A, rows, cols)
    return tuple(output)


def validate_splits(splits, A, rows, cols):
    """Check a six-tensor split for shape errors, label mismatches and leakage.

    Also used on user-supplied splits; leakage is never silently accepted.
    """
    if len(splits) != 6:
        raise ValueError("splits must contain train/val/test edge_index and labels.")
    seen = set()
    for name, e, y in zip(("train", "val", "test"), splits[::2], splits[1::2]):
        if e.ndim != 2 or e.shape[0] != 2 or y.shape != (e.shape[1],) or e.dtype != torch.long:
            raise ValueError(f"Invalid shapes or edge dtype in {name}.")
        if set(y.detach().cpu().tolist()) != {0.0, 1.0}:
            raise ValueError(f"{name} must contain both classes. Use more edges or global splitting.")
        if (e < 0).any() or (e[0] >= len(rows)).any() or (e[1] >= len(cols)).any():
            raise ValueError("Edge index out of range.")
        if not torch.equal(A[e[0], e[1]].float(), y.float()):
            raise ValueError("Split labels disagree with A.")
        pairs = list(zip(*e.detach().cpu().tolist()))
        if len(set(pairs)) != len(pairs) or seen.intersection(pairs):
            raise ValueError("Duplicate or overlapping train/validation/test pairs.")
        if any(rows[i] == cols[j] for i, j in pairs):
            raise ValueError("Self-pairs are excluded from this protocol.")
        seen.update(pairs)
