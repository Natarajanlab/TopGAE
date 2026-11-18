"""Training loop, early stopping and evaluation for TopGAE."""
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .losses import mask_expression, masked_expression_loss
from .splits import make_splits, validate_splits
from .structure import structural_features
from .utils import seed_everything, unique_names


def link_metrics(logits, labels):
    """Return ``(AUROC, AUPRC)``.

    Rank-based metrics are computed directly from logits, which avoids ties
    caused by sigmoid saturation.
    """
    y, scores = labels.detach().cpu().numpy(), logits.detach().cpu().numpy()
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def train_topgae(model, X_row, X_col, A_target, row_genes, col_genes,
                 num_epochs=2000, lr=5e-4, patience=50, val_check_freq=5,
                 device=None, verbose=True, *, seed=42, splits=None,
                 split_mode="global", lambda_expr=0.1, mask_ratio=0.2,
                 reconstruction_loss="mse", edge_dropout=0.0, weight_decay=5e-4,
                 selection_metric="auprc"):
    """Train a :class:`~topgae.model.TopGAE` model and evaluate it on held-out edges.

    Parameters
    ----------
    model : TopGAE
    X_row, X_col : tensors
        Source-role and target-role expression from :func:`topgae.prepare_inputs`.
    A_target : tensor
        Binary reference adjacency (labels).
    row_genes, col_genes : list of str
    splits : tuple of six tensors, optional
        Pre-computed splits from :func:`topgae.make_splits`. If ``None`` the
        splits are created here with ``split_mode`` and ``seed``.
    lambda_expr : float
        Weight of the masked-expression loss (0 disables the second pass).
    mask_ratio : float
        Fraction of unique genes masked per epoch.
    reconstruction_loss : {"mse", "cosine"}
    selection_metric : {"auprc", "auroc", "sum"}
        Validation metric used for early stopping / checkpoint selection.

    Notes
    -----
    Training uses fixed, balanced training negatives; validation and test
    negatives never enter training. Only training positives form the
    message-passing graph and the structural descriptors (computed once).
    The link loss uses an unmasked encoder pass and the expression loss a
    separate masked pass with the same parameters.

    Returns
    -------
    S1 : (n_sources, n_targets) tensor
        S1 scores on training message edges (zero elsewhere).
    S2 : (n_sources, n_targets) tensor
        S2 scores on training message edges (zero elsewhere).
    A_hat : (n_sources, n_targets) tensor
        Dense link LOGITS for every candidate pair - these are the predictions.
    metrics : dict
        ``test_auroc``, ``test_auprc``, ``best_epoch``, ``best_val_score``,
        ``best_val_loss``, ``history``, ``splits``, ``split_counts``,
        ``message_edge_index``, ``importance``, ``structure`` and ``config``.
    """
    if num_epochs < 1 or patience < 1 or val_check_freq < 1:
        raise ValueError("Epochs, patience and validation frequency must be positive.")
    if lambda_expr < 0 or not 0 <= edge_dropout < 1 or not 0 < mask_ratio < 1:
        raise ValueError("Invalid loss weight, dropout or mask ratio.")
    if selection_metric not in ("auprc", "auroc", "sum"):
        raise ValueError("selection_metric must be auprc, auroc or sum.")
    if reconstruction_loss not in ("mse", "cosine"):
        raise ValueError("reconstruction_loss must be mse or cosine.")
    rows, cols = unique_names(row_genes), unique_names(col_genes)
    seed_everything(seed)
    device = X_row.device if device is None else torch.device(device)
    model.to(device)
    X_row, X_col, A = X_row.to(device), X_col.to(device), A_target.to(device)
    if X_row.shape != (len(rows), model.in_features) or X_col.shape != (len(cols), model.in_features):
        raise ValueError("Expression shapes do not match model or gene lists.")
    if A.shape != (len(rows), len(cols)) or not torch.all((A == 0) | (A == 1)):
        raise ValueError("A must be binary and match gene lists.")
    if not torch.isfinite(X_row).all() or not torch.isfinite(X_col).all():
        raise ValueError("Nonfinite expression values.")
    ci = {g: j for j, g in enumerate(cols)}
    for i, g in enumerate(rows):
        if g in ci and not torch.allclose(X_row[i], X_col[ci[g]], atol=1e-6):
            raise ValueError("Shared genes must have the same expression in both roles.")
    split_origin = split_mode if splits is None else "provided"
    splits = make_splits(A, rows, cols, seed, split_mode) if splits is None else tuple(t.to(device) for t in splits)
    validate_splits(splits, A, rows, cols)
    te, ty, ve, vy, ee, ey = splits
    message_edges = te[:, ty == 1]
    structure = structural_features(message_edges, rows, cols, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {k: [] for k in ("train_loss", "train_link_loss", "train_expr_loss", "val_epoch",
                               "val_loss", "val_auroc", "val_auprc", "train_auroc", "train_auprc")}
    best_score, best_state, stale = -float("inf"), None, 0
    best_epoch = 0
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        keep = torch.rand(message_edges.shape[1], device=device) >= edge_dropout
        epoch_edges = message_edges[:, keep]
        zr, zc = model.encode(X_row, X_col, epoch_edges, rows, cols, structure=structure, store_scores=False)
        logits = model.decode_link(zr, zc, te)
        link_loss = F.binary_cross_entropy_with_logits(logits, ty)
        expr_loss = link_loss.new_zeros(())
        if lambda_expr > 0:
            xr, xc, mr, mc = mask_expression(X_row, X_col, rows, cols, mask_ratio)
            # Gate heads only see corrupted expression in this pass.
            zmr, zmc = model.encode(xr, xc, epoch_edges, rows, cols, structure=structure, store_scores=False)
            pr, pc = model.decode_feat(zmr, zmc)
            expr_loss = masked_expression_loss(pr, pc, X_row, X_col, mr, mc, rows, cols, reconstruction_loss)
        loss = link_loss + lambda_expr * expr_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history["train_loss"].append(loss.item())
        history["train_link_loss"].append(link_loss.item())
        history["train_expr_loss"].append(expr_loss.item())
        if epoch % val_check_freq == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                zr, zc = model.encode(X_row, X_col, message_edges, rows, cols, structure=structure, store_scores=False)
                vl = model.decode_link(zr, zc, ve)
                val_loss = F.binary_cross_entropy_with_logits(vl, vy).item()
                auc, ap = link_metrics(vl, vy)
                train_auc, train_ap = link_metrics(model.decode_link(zr, zc, te), ty)
            for key, value in zip(("val_epoch", "val_loss", "val_auroc", "val_auprc", "train_auroc", "train_auprc"),
                                  (epoch, val_loss, auc, ap, train_auc, train_ap)):
                history[key].append(value)
            score = {"auprc": ap, "auroc": auc, "sum": auc + ap}[selection_metric]
            if verbose:
                print(f"Epoch {epoch}: link={link_loss.item():.4f}, expr={expr_loss.item():.4f}, "
                      f"val AUROC={auc:.4f}, AP={ap:.4f}")
            if score > best_score:
                best_score, best_epoch, best_val_loss, stale = score, epoch, val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        zr, zc = model.encode(X_row, X_col, message_edges, rows, cols, structure=structure)
        auc, ap = link_metrics(model.decode_link(zr, zc, ee), ey)
        A_hat = model.decode_all(zr, zc).cpu()
        s1, s2 = model.S1_output.cpu(), model.S2_output.cpu()
        importance = model.ds_layer.last_importance
    counts = {name: {"positive": int((y == 1).sum()), "negative": int((y == 0).sum())}
              for name, y in zip(("train", "val", "test"), splits[1::2])}
    metrics = dict(test_auroc=auc, test_auprc=ap, best_epoch=best_epoch,
                   best_val_score=best_score, best_val_loss=best_val_loss,
                   history=history, splits=tuple(t.detach().cpu() for t in splits), split_counts=counts,
                   message_edge_index=message_edges.cpu(),
                   importance=None if importance is None else tuple(t.cpu() for t in importance),
                   structure=tuple(t.cpu() for t in structure),
                   config=dict(seed=seed, split_mode=split_origin,
                               lambda_expr=lambda_expr, mask_ratio=mask_ratio, reconstruction_loss=reconstruction_loss,
                               edge_dropout=edge_dropout, selection_metric=selection_metric,
                               use_importance=model.ds_layer.use_importance,
                               reverse_messages=model.ds_layer.reverse_messages,
                               use_s1=model.ds_layer.use_s1, use_s2=model.ds_layer.use_s2))
    if verbose:
        print(f"Best epoch {best_epoch}; test AUROC={auc:.4f}, AP={ap:.4f}")
    return s1, s2, A_hat, metrics
