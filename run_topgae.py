#!/usr/bin/env python
"""Run TopGAE on your own single-cell data (.h5ad) and prior network.

TopGAE needs two inputs:

1. an AnnData file (cells x genes) with raw counts or log-normalised expression;
2. a directed prior network (CSV/TSV edge list, one regulator -> target per row),
   e.g. a ChIP-seq, motif or curated TF-target network.

It trains TopGAE on the prior edges, reports held-out AUROC / AUPRC, and
scores every candidate regulator -> target pair. The main output is
``ranked_edges.csv``; rows with ``InPrior == False`` are new candidate
interactions not in your prior.

Minimal example::

    python run_topgae.py --h5ad my_data.h5ad --network prior.csv --out-dir topgae_results

See ``python run_topgae.py --help`` and README.md ("Run TopGAE on your own data").
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import expit

from topgae import (TopGAE, build_adjacency, make_splits, prepare_inputs,
                    seed_everything, train_topgae)

SOURCE_COLUMNS = ("Gene1", "Source", "source", "from", "TF", "tf", "Regulator", "regulator")
TARGET_COLUMNS = ("Gene2", "Target", "target", "to", "Target_gene", "target_gene")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run TopGAE on an .h5ad file and a prior regulatory network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io = p.add_argument_group("inputs / outputs")
    io.add_argument("--h5ad", required=True, help="AnnData file, cells x genes.")
    io.add_argument("--network", required=True,
                    help="Prior network edge list (CSV or TSV, with a header row).")
    io.add_argument("--source-col", default=None,
                    help=f"Regulator column in --network (auto-detected from {', '.join(SOURCE_COLUMNS)}).")
    io.add_argument("--target-col", default=None,
                    help=f"Target column in --network (auto-detected from {', '.join(TARGET_COLUMNS)}).")
    io.add_argument("--sep", default=None,
                    help="Network delimiter. Default: tab for .tsv/.txt, comma otherwise.")
    io.add_argument("--out-dir", default="topgae_results", help="Output folder.")

    ex = p.add_argument_group("expression")
    ex.add_argument("--layer", default=None,
                    help="Use adata.layers[LAYER] instead of adata.X (e.g. 'counts').")
    ex.add_argument("--gene-column", default=None,
                    help="Column of adata.var holding gene names that match the network "
                         "(default: adata.var_names).")
    ex.add_argument("--uppercase", action="store_true",
                    help="Upper-case gene names in both the expression data and the network.")
    ex.add_argument("--obs-key", default=None,
                    help="Column of adata.obs used to select cells (e.g. 'cell_type').")
    ex.add_argument("--obs-values", nargs="+", default=None,
                    help="Values of --obs-key to keep (e.g. 'HSC' or 'control').")
    ex.add_argument("--max-cells", type=int, default=None,
                    help="Randomly subsample to at most this many cells (memory / speed).")
    ex.add_argument("--input", choices=("auto", "counts", "log"), default="auto",
                    help="'counts': library-size normalise to --target-sum then log1p; "
                         "'log': data are already log-normalised; 'auto': counts if all values are integers.")
    ex.add_argument("--target-sum", type=float, default=1e4,
                    help="Library size after normalisation of raw counts.")
    ex.add_argument("--min-cells", type=int, default=10,
                    help="Keep target genes detected (> 0) in at least this many cells.")
    ex.add_argument("--n-top-genes", type=int, default=None,
                    help="Keep only the N most variable target genes (all regulators are always kept). "
                         "Recommended for --candidates all-pairs (the paper used 500 or 1000).")
    ex.add_argument("--normalization", choices=("cell", "gene", "none"), default="cell",
                    help="Standardisation applied by TopGAE (paper: per cell across genes).")

    md = p.add_argument_group("candidate edges and splitting")
    md.add_argument("--candidates", choices=("tf-targets", "all-pairs"), default="tf-targets",
                    help="'tf-targets': sources are the regulators of the prior network, targets are all "
                         "kept genes (used for the hESC perturbation analyses). 'all-pairs': every gene is "
                         "both a source and a target, negatives from all gene pairs (BEELINE / simulation "
                         "benchmark protocol; memory grows with genes^2).")
    md.add_argument("--val-fraction", type=float, default=0.05, help="Fraction of prior edges for validation.")
    md.add_argument("--test-fraction", type=float, default=0.10, help="Fraction of prior edges for testing.")

    tr = p.add_argument_group("model and training (defaults = values used in the paper's experiment scripts)")
    tr.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="One model per seed; scores are averaged. The paper used 42 43 44 45 46.")
    tr.add_argument("--hidden-dim", type=int, default=64)
    tr.add_argument("--latent-dim", type=int, default=32)
    tr.add_argument("--dropout", type=float, default=0.2)
    tr.add_argument("--lambda-expr", type=float, default=0.5,
                    help="Weight of the masked-expression reconstruction loss (0 disables it).")
    tr.add_argument("--mask-ratio", type=float, default=0.2)
    tr.add_argument("--recon-loss", choices=("cosine", "mse"), default="cosine")
    tr.add_argument("--no-structure", action="store_true",
                    help="Disable the structural (degree + HITS) gates, as for the SERGIO runs.")
    tr.add_argument("--epochs", type=int, default=2000)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--weight-decay", type=float, default=5e-4)
    tr.add_argument("--patience", type=int, default=50,
                    help="Early stopping: validation checks without improvement.")
    tr.add_argument("--val-check-freq", type=int, default=5)
    tr.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda' or e.g. 'cuda:1'.")

    out = p.add_argument_group("outputs")
    out.add_argument("--top-edges", type=int, default=None,
                     help="Write only the N highest-scoring edges to ranked_edges.csv (default: all).")
    out.add_argument("--save-matrix", action="store_true",
                     help="Also save the full mean score matrix as score_matrix.npz.")
    out.add_argument("--save-message-scores", action="store_true",
                     help="Also save S1 / S2 message weights on the training prior edges for each seed.")
    out.add_argument("--quiet", action="store_true", help="Do not print training progress.")

    args = p.parse_args(argv)
    if (args.obs_key is None) != (args.obs_values is None):
        p.error("--obs-key and --obs-values must be used together.")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must be unique.")
    return args


def log(message):
    print(message, flush=True)


# --------------------------------------------------------------------------- inputs

def load_network(path, source_col=None, target_col=None, sep=None, uppercase=False):
    """Read a directed edge list and return a clean Gene1 -> Gene2 DataFrame."""
    path = Path(path)
    if sep is None:
        name = path.name.lower().removesuffix(".gz")
        sep = "\t" if name.endswith((".tsv", ".txt")) else ","
    net = pd.read_csv(path, sep=sep)
    source_col = source_col or next((c for c in SOURCE_COLUMNS if c in net.columns), None)
    target_col = target_col or next((c for c in TARGET_COLUMNS if c in net.columns), None)
    if source_col not in net.columns or target_col not in net.columns or source_col == target_col:
        sys.exit(f"Could not identify regulator/target columns in {path} (columns: {list(net.columns)}). "
                 "Use --source-col and --target-col.")
    net = net[[source_col, target_col]].dropna()
    net.columns = ["Gene1", "Gene2"]
    for column in net:
        net[column] = net[column].astype(str).str.strip()
        if uppercase:
            net[column] = net[column].str.upper()
    n_rows = len(net)
    net = net[net.Gene1.ne("") & net.Gene2.ne("") & net.Gene1.ne(net.Gene2)]
    net = net.drop_duplicates(["Gene1", "Gene2"]).reset_index(drop=True)
    log(f"Network: {n_rows} rows -> {len(net)} unique directed edges "
        f"({net.Gene1.nunique()} regulators, {net.Gene2.nunique()} targets) "
        f"[columns '{source_col}' -> '{target_col}']")
    return net


def _sample_values(values, limit=5_000_000):
    flat = np.asarray(values).ravel()
    return flat[::max(1, flat.size // limit)]


def load_expression(args):
    """Return (log-normalised cells x genes matrix, gene names, info)."""
    try:
        import anndata as ad
    except ImportError:
        sys.exit("Reading .h5ad files needs the 'anndata' package: pip install anndata")
    adata = ad.read_h5ad(args.h5ad)
    info = {"input_cells": int(adata.n_obs), "input_genes": int(adata.n_vars)}
    log(f"Expression: {adata.n_obs} cells x {adata.n_vars} genes")

    if args.obs_key is not None:
        if args.obs_key not in adata.obs:
            sys.exit(f"'{args.obs_key}' is not a column of adata.obs (available: {list(adata.obs.columns)}).")
        keep = adata.obs[args.obs_key].astype(str).isin(args.obs_values).to_numpy()
        if not keep.any():
            sys.exit(f"No cells with {args.obs_key} in {args.obs_values}.")
        adata = adata[keep]
        log(f"Selected {adata.n_obs} cells with {args.obs_key} in {args.obs_values}")
    if args.max_cells is not None and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seeds[0])
        adata = adata[np.sort(rng.choice(adata.n_obs, size=args.max_cells, replace=False))]
        log(f"Subsampled to {adata.n_obs} cells (seed {args.seeds[0]})")
    info["used_cells"] = int(adata.n_obs)

    if args.layer is not None:
        if args.layer not in adata.layers:
            sys.exit(f"Layer '{args.layer}' not found (available: {list(adata.layers.keys())}).")
        X = adata.layers[args.layer]
    else:
        X = adata.X
    X = sparse.csr_matrix(X, dtype=np.float32) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)

    if args.gene_column is not None:
        if args.gene_column not in adata.var:
            sys.exit(f"'{args.gene_column}' is not a column of adata.var (available: {list(adata.var.columns)}).")
        names = pd.Index(adata.var[args.gene_column].astype(str))
    else:
        names = pd.Index(adata.var_names.astype(str))
    names = names.str.strip()
    if args.uppercase:
        names = names.str.upper()
    bad = (names == "") | (names.str.lower() == "nan")
    duplicated = names.duplicated(keep="first")
    keep_genes = np.flatnonzero(~(bad | duplicated))
    if len(keep_genes) < len(names):
        log(f"Dropping {int(bad.sum())} genes without a name and {int((duplicated & ~bad).sum())} "
            "duplicated gene names (first occurrence kept)")
        X, names = X[:, keep_genes], names[keep_genes]

    values = X.data if sparse.issparse(X) else X
    if not np.isfinite(values).all():
        sys.exit("Expression contains NaN or infinite values.")
    if (values < 0).any():
        sys.exit("Expression contains negative values (scaled data?). Provide raw counts or "
                 "log-normalised expression, e.g. --layer counts.")
    sample = _sample_values(values)
    mode = args.input
    if mode == "auto":
        mode = "counts" if np.allclose(sample, np.rint(sample), atol=1e-4) else "log"
    if mode == "counts":
        library = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        scale = (args.target_sum / np.where(library > 0, library, 1.0)).astype(np.float32)
        if sparse.issparse(X):
            X = sparse.csr_matrix(sparse.diags(scale) @ X)
            X.data = np.log1p(X.data)
        else:
            X = np.log1p(X * scale[:, None])
        log(f"Treated input as raw counts: normalised to {args.target_sum:g} per cell, then log1p")
    else:
        log("Treated input as log-normalised expression (no further transform)")
    info["input_type"] = mode
    return X, names, info


def select_candidates(X, names, net, args):
    """Choose source / target genes and the prior edges between them."""
    measured = set(names)
    net = net[net.Gene1.isin(measured) & net.Gene2.isin(measured)]
    if net.empty:
        sys.exit("No network edge has both genes in the expression data. Check that gene identifiers "
                 "match (symbols vs Ensembl IDs, --gene-column, --uppercase for case differences).\n"
                 f"  expression genes, e.g.: {list(names[:5])}")
    detected = np.asarray((X > 0).sum(axis=0)).ravel()
    target_mask = detected >= args.min_cells
    if args.n_top_genes is not None and target_mask.sum() > args.n_top_genes:
        index = np.flatnonzero(target_mask)
        sub = X[:, index]
        mean = np.asarray(sub.mean(axis=0)).ravel()
        second = (np.asarray(sub.multiply(sub).mean(axis=0)).ravel() if sparse.issparse(sub)
                  else np.square(sub).mean(axis=0))
        variance = np.maximum(0.0, second - mean ** 2)
        top = index[np.argsort(-variance, kind="stable")[:args.n_top_genes]]
        target_mask = np.zeros_like(target_mask)
        target_mask[top] = True
    targets = [g for g, keep in zip(names, target_mask) if keep]
    if not targets:
        sys.exit(f"No gene is detected in at least {args.min_cells} cells; lower --min-cells.")
    regulators = sorted(net.Gene1.unique())

    if args.candidates == "all-pairs":
        chosen = set(regulators) | set(targets)
        genes = [g for g in names if g in chosen]
        rows, cols, split_mode = genes, list(genes), "gaedgrn"
    else:
        rows, cols, split_mode = regulators, targets, "global"
    net = net[net.Gene1.isin(set(rows)) & net.Gene2.isin(set(cols))].reset_index(drop=True)
    if net.empty:
        sys.exit("No prior edge points to a kept target gene; lower --min-cells or raise --n-top-genes.")
    rows = sorted(set(net.Gene1)) if args.candidates == "tf-targets" else rows
    pairs = len(rows) * len(cols)
    log(f"Candidates ({args.candidates}): {len(rows)} sources x {len(cols)} targets = {pairs:,} pairs; "
        f"{len(net)} prior edges")
    if args.candidates == "all-pairs" and len(rows) > 5000:
        log("WARNING: all-pairs with more than 5,000 genes needs a lot of memory and time; "
            "consider --n-top-genes 1000.")
    return rows, cols, net, split_mode


def to_gene_table(X, names, genes):
    lookup = {g: i for i, g in enumerate(names)}
    sub = X[:, [lookup[g] for g in genes]]
    sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    return pd.DataFrame(sub.T, index=genes, dtype=np.float32)


# --------------------------------------------------------------------------- outputs

def write_ranked_edges(mean, sd, rows, cols, prior, path, top=None):
    R, C = mean.shape
    valid = np.ones(R * C, dtype=bool)
    col_index = {g: j for j, g in enumerate(cols)}
    for i, g in enumerate(rows):
        if g in col_index:
            valid[i * C + col_index[g]] = False  # no self-pairs
    flat = mean.ravel()
    index = np.flatnonzero(valid)
    order = index[np.argsort(-flat[index], kind="stable")]
    if top is not None:
        order = order[:top]
    rows_arr, cols_arr = np.asarray(rows, dtype=object), np.asarray(cols, dtype=object)
    table = pd.DataFrame({
        "Rank": np.arange(1, len(order) + 1),
        "Source": rows_arr[order // C],
        "Target": cols_arr[order % C],
        "Score": flat[order],
        "Probability": expit(flat[order].astype(np.float64)),
        "InPrior": prior.ravel()[order].astype(bool),
    })
    if sd is not None:
        table.insert(4, "ScoreSD", sd.ravel()[order])
    table.to_csv(path, index=False)
    return table


def main(argv=None):
    args = parse_args(argv)
    start = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log(f"TopGAE | device: {device} | output: {out.resolve()}")

    net = load_network(args.network, args.source_col, args.target_col, args.sep, args.uppercase)
    X, names, info = load_expression(args)
    rows, cols, net, split_mode = select_candidates(X, names, net, args)
    genes = list(dict.fromkeys(rows + cols))
    expression = to_gene_table(X, names, genes)
    del X
    adjacency, rows, cols = build_adjacency(net, rows, cols)
    X_row, X_col, A = prepare_inputs(expression, adjacency, rows, cols, "cpu",
                                     normalization=args.normalization)
    del expression, adjacency
    prior = A.numpy().astype(bool)

    config = dict(vars(args), split_mode=split_mode, n_sources=len(rows), n_targets=len(cols),
                  n_cells=int(X_row.shape[1]), n_prior_edges=int(prior.sum()), **info)
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    mean = m2 = None
    records = []
    for k, seed in enumerate(args.seeds, start=1):
        log(f"\n=== Seed {seed} ({k}/{len(args.seeds)}) ===")
        seed_everything(seed)
        try:
            splits = make_splits(A, rows, cols, seed=seed, split_mode=split_mode,
                                 val_fraction=args.val_fraction, test_fraction=args.test_fraction)
        except ValueError as exc:
            sys.exit(f"Could not split the prior edges: {exc}")
        model = TopGAE(in_features=X_row.shape[1], out_feature=args.hidden_dim,
                       latent_dim=args.latent_dim, dropout=args.dropout,
                       use_importance=not args.no_structure, reverse_messages=True,
                       use_s1=True, use_s2=True)
        s1, s2, a_hat, metrics = train_topgae(
            model, X_row, X_col, A, rows, cols, splits=splits, seed=seed, device=device,
            num_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, val_check_freq=args.val_check_freq,
            lambda_expr=args.lambda_expr, mask_ratio=args.mask_ratio,
            reconstruction_loss=args.recon_loss, edge_dropout=0.0,
            selection_metric="auprc", verbose=not args.quiet)
        scores = a_hat.numpy().astype(np.float64)
        if mean is None:
            mean, m2 = scores.copy(), np.zeros_like(scores)
        else:  # Welford running mean / variance over seeds
            delta = scores - mean
            mean += delta / k
            m2 += delta * (scores - mean)
        counts = metrics["split_counts"]
        records.append(dict(Seed=seed, TestAUROC=metrics["test_auroc"], TestAUPRC=metrics["test_auprc"],
                            BestEpoch=metrics["best_epoch"], BestValAUPRC=metrics["best_val_score"],
                            TrainPositive=counts["train"]["positive"], ValPositive=counts["val"]["positive"],
                            TestPositive=counts["test"]["positive"]))
        pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
        torch.save(dict(state_dict={name: t.detach().cpu() for name, t in model.state_dict().items()},
                        seed=seed, row_genes=rows, col_genes=cols, in_features=int(X_row.shape[1]),
                        splits=metrics["splits"], config=config),
                   out / f"model_seed_{seed}.pt")
        if args.save_message_scores:
            edges = metrics["message_edge_index"]
            pd.DataFrame({"Source": np.asarray(rows, dtype=object)[edges[0].numpy()],
                          "Target": np.asarray(cols, dtype=object)[edges[1].numpy()],
                          "S1": s1[edges[0], edges[1]].numpy(),
                          "S2": s2[edges[0], edges[1]].numpy()}).to_csv(
                out / f"message_scores_seed_{seed}.csv", index=False)
        log(f"Seed {seed}: test AUROC={metrics['test_auroc']:.4f}, AUPRC={metrics['test_auprc']:.4f} "
            f"(best epoch {metrics['best_epoch']})")
        del model, s1, s2, a_hat
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sd = np.sqrt(m2 / (len(args.seeds) - 1)) if len(args.seeds) > 1 else None
    table = write_ranked_edges(mean.astype(np.float32), None if sd is None else sd.astype(np.float32),
                               rows, cols, prior, out / "ranked_edges.csv", args.top_edges)
    if args.save_matrix:
        np.savez_compressed(out / "score_matrix.npz", score=mean.astype(np.float32),
                            source_genes=np.asarray(rows), target_genes=np.asarray(cols),
                            in_prior=prior)

    summary = pd.DataFrame(records)
    log("\n=== Done ===")
    log(f"Held-out prior edges: AUROC {summary.TestAUROC.mean():.4f}, AUPRC {summary.TestAUPRC.mean():.4f} "
        f"(mean over {len(summary)} seed(s))")
    log(f"Wrote {len(table):,} ranked edges ({int((~table.InPrior).sum()):,} not in the prior) "
        f"to {out / 'ranked_edges.csv'}")
    log(f"Elapsed: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
