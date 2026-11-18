"""TopGAE on dyngen simulations (Fig. 2A-B, Table S1).

Install the package first (pip install -e . in the repository root), then run
from this folder:  python run_dyngen.py   (or call main() in a notebook).

dyngen version: expects pairs of files in BASE_DIR named
    dyngen_<name>_mat.csv   raw counts, cells x genes, first column = cell IDs
    dyngen_<name>_net.csv   ground-truth network with 'from' and 'to' columns
"""
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topgae import (
    build_adjacency, prepare_inputs, seed_everything,
    make_splits, TopGAE, train_topgae,
)

# --- CONFIG ---
BASE_DIR = Path("./dyngen_data")
OUT_DIR = Path("Dyngen_TopGAE_Results")
SEEDS = [42, 43, 44, 45, 46]                                        # 5 runs per network
N_RUNS = len(SEEDS)

RESULTS_FILE = OUT_DIR / "Summary.csv"
RUN_RESULTS_FILE = OUT_DIR / "Per_Run.csv"
SPLITS_DIR = OUT_DIR / "Splits"
AHAT_DIR = OUT_DIR / "A_hat"

# File naming: dyngen_<name>_mat.csv / dyngen_<name>_net.csv
MAT_PATTERN = re.compile(r"^dyngen_(?P<name>.+)_mat\.csv$")
NET_TEMPLATE = "dyngen_{name}_net.csv"
NET_SOURCE_COL, NET_TARGET_COL = "from", "to"

# Raw-count preprocessing, applied BEFORE prepare_inputs():
#   "log1p_libsize" : scale each cell to the median library size, then log1p (recommended for raw counts)
#   "log1p"         : log1p only
#   "none"          : pass raw counts through (use if prepare_inputs already handles counts)
COUNT_TRANSFORM = "log1p_libsize"
DROP_ZERO_GENES = True        # genes with zero counts in every cell carry no signal
MIN_CELL_COUNTS = 1           # cells with fewer total counts are dropped

HIDDEN_DIM = 64
LATENT_DIM = 32
DROPOUT = 0.2
NUM_EPOCHS = 2000
PATIENCE = 50
VAL_CHECK_FREQ = 5
LEARNING_RATE = 5e-4

USE_IMPORTANCE = True
REVERSE_MESSAGES = True
LAMBDA_EXPR = 0.5
MASK_RATIO = 0.2
RECONSTRUCTION_LOSS = "cosine"
EDGE_DROPOUT = 0.0
NORMALIZATION = "cell"
SPLIT_MODE = "gaedgrn"
SELECTION_METRIC = "auprc"
INCLUDE_ALL_EXPRESSION_TARGETS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_datasets(base):
    """Each dyngen_<name>_mat.csv that has a matching dyngen_<name>_net.csv."""
    pairs = []
    for exp_path in sorted(base.glob("dyngen_*_mat.csv")):
        m = MAT_PATTERN.match(exp_path.name)
        if not m:
            continue
        name = m.group("name")
        net_path = base / NET_TEMPLATE.format(name=name)
        if net_path.is_file():
            pairs.append((name, net_path, exp_path))
        else:
            print(f"Skipping {name}: missing network file {net_path.name}")
    return pairs


def transform_counts(exp):
    """exp: genes x cells raw counts -> transformed genes x cells."""
    if (exp.values < 0).any():
        raise ValueError("Negative values found; expected raw counts.")
    lib = exp.sum(axis=0)
    keep_cells = lib >= MIN_CELL_COUNTS
    if (~keep_cells).any():
        print(f"  Dropping {int((~keep_cells).sum())} cells with < {MIN_CELL_COUNTS} total counts")
        exp, lib = exp.loc[:, keep_cells], lib[keep_cells]
    if DROP_ZERO_GENES:
        nonzero = exp.sum(axis=1) > 0
        if (~nonzero).any():
            print(f"  Dropping {int((~nonzero).sum())} all-zero genes")
            exp = exp.loc[nonzero]
    if COUNT_TRANSFORM == "log1p_libsize":
        exp = np.log1p(exp.div(lib, axis=1) * lib.median())
    elif COUNT_TRANSFORM == "log1p":
        exp = np.log1p(exp)
    elif COUNT_TRANSFORM != "none":
        raise ValueError(f"Unknown COUNT_TRANSFORM: {COUNT_TRANSFORM}")
    return exp


def load_dataset(network_path, expression_path):
    net = pd.read_csv(network_path)
    # dyngen matrix is cells x genes with cell IDs in the first column -> transpose to genes x cells.
    exp = pd.read_csv(expression_path, index_col=0).T

    if not {NET_SOURCE_COL, NET_TARGET_COL}.issubset(net.columns):
        raise ValueError(f"Network file needs '{NET_SOURCE_COL}' and '{NET_TARGET_COL}' columns; "
                         f"found {list(net.columns)}")
    net = net.rename(columns={NET_SOURCE_COL: "Gene1", NET_TARGET_COL: "Gene2"})
    if net[["Gene1", "Gene2"]].isna().any().any():
        raise ValueError("Network contains missing gene identifiers.")
    net["Gene1"] = net["Gene1"].astype(str).str.strip()
    net["Gene2"] = net["Gene2"].astype(str).str.strip()

    exp.index = exp.index.astype(str).str.strip()
    if not exp.index.is_unique:
        raise ValueError("Expression gene identifiers are duplicated after normalization.")
    exp = exp.apply(pd.to_numeric, errors="raise").astype(float)
    n_cells_raw = exp.shape[1]
    exp = transform_counts(exp)

    n_edges_raw = len(net)
    net = net[net.Gene1.ne(net.Gene2)].drop_duplicates(["Gene1", "Gene2"])
    net = net[net.Gene1.isin(exp.index) & net.Gene2.isin(exp.index)].reset_index(drop=True)
    if net.empty:
        raise ValueError("No positive interactions match expression genes. Check gene IDs/orientation.")
    if len(net) < n_edges_raw:
        print(f"  Kept {len(net)}/{n_edges_raw} network edges "
              "(self-loops, duplicates, or genes absent from expression removed)")

    if SPLIT_MODE == "gaedgrn":
        row_genes = exp.index.tolist()
        col_genes = list(row_genes)
    else:
        row_genes = sorted(net.Gene1.unique().tolist())
        col_genes = exp.index.tolist() if INCLUDE_ALL_EXPRESSION_TARGETS else sorted(net.Gene2.unique().tolist())
    adj, row_genes, col_genes = build_adjacency(net, row_genes, col_genes)
    xr, xc, a = prepare_inputs(exp, adj, row_genes, col_genes, "cpu", normalization=NORMALIZATION)
    print(f"  {exp.shape[0]} genes x {exp.shape[1]} cells (of {n_cells_raw}), {len(net)} true edges")
    return xr, xc, a, row_genes, col_genes


def save_ahat(a_hat, rows, cols, dataset_name, seed):
    """Save A_hat (logits) as a labelled matrix and as a ranked edge list."""
    arr = a_hat.detach().cpu().numpy() if torch.is_tensor(a_hat) else np.asarray(a_hat)
    if arr.shape != (len(rows), len(cols)):
        raise ValueError(f"A_hat shape {arr.shape} != ({len(rows)}, {len(cols)})")
    out = AHAT_DIR / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    matrix = pd.DataFrame(arr, index=rows, columns=cols)
    matrix.to_csv(out / f"A_hat_seed_{seed}.csv")

    # Ranked edges (BEELINE rankedEdges format), self-loops removed.
    edges = matrix.stack().reset_index()
    edges.columns = ["Gene1", "Gene2", "EdgeWeight"]
    edges = edges[edges.Gene1.ne(edges.Gene2)].sort_values("EdgeWeight", ascending=False)
    edges.to_csv(out / f"rankedEdges_seed_{seed}.csv", index=False)
    return str(out / f"A_hat_seed_{seed}.csv")


def run_training_once(data, seed, split_path, dataset_name, splits=None):
    xr, xc, a, rows, cols = data
    seed_everything(seed)
    if splits is None:
        splits = make_splits(a, rows, cols, seed=seed, split_mode=SPLIT_MODE)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"splits": splits, "row_genes": rows, "col_genes": cols,
                "seed": seed, "split_mode": SPLIT_MODE,
                "negative_universe": "all_gene_pairs" if SPLIT_MODE == "gaedgrn" else "tf_gene_pairs",
                "training_objective": "balanced_sampled_bce"}, split_path)
    model = TopGAE(
        in_features=xr.shape[1], out_feature=HIDDEN_DIM, latent_dim=LATENT_DIM,
        dropout=DROPOUT, use_importance=USE_IMPORTANCE,
        reverse_messages=REVERSE_MESSAGES, use_s1=True, use_s2=True,
    )
    s1, s2, a_hat, metrics = train_topgae(
        model, xr, xc, a, rows, cols,
        device=device, seed=seed, splits=splits,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
        patience=PATIENCE, val_check_freq=VAL_CHECK_FREQ, verbose=False,
        lambda_expr=LAMBDA_EXPR, mask_ratio=MASK_RATIO,
        reconstruction_loss=RECONSTRUCTION_LOSS,
        edge_dropout=EDGE_DROPOUT, selection_metric=SELECTION_METRIC,
    )
    ahat_file = save_ahat(a_hat, rows, cols, dataset_name, seed)
    return {"AUROC": metrics["test_auroc"], "AUPRC": metrics["test_auprc"],
            "BestEpoch": metrics["best_epoch"], "BestValScore": metrics["best_val_score"],
            "TrainPositive": metrics["split_counts"]["train"]["positive"],
            "ValPositive": metrics["split_counts"]["val"]["positive"],
            "TestPositive": metrics["split_counts"]["test"]["positive"],
            "AhatFile": ahat_file}


def save_results(records):
    """Save successes AND failures; never report only successful run counts."""
    runs = pd.DataFrame(records)
    summaries = []
    for dataset, group in runs.groupby("Dataset", sort=False):
        ok = group[group.Status.eq("ok")]
        n = len(ok)
        summaries.append({
            "Dataset": dataset, "RequestedRuns": N_RUNS, "AttemptedRuns": len(group),
            "SplitMode": SPLIT_MODE, "CountTransform": COUNT_TRANSFORM,
            "Runs": n, "FailedRuns": int(group.Status.eq("error").sum()),
            "Mean_AUROC": ok.AUROC.mean() if n else np.nan,
            "SD_AUROC": ok.AUROC.std(ddof=1) if n > 1 else np.nan,
            "Mean_AUPRC": ok.AUPRC.mean() if n else np.nan,
            "SD_AUPRC": ok.AUPRC.std(ddof=1) if n > 1 else np.nan,
        })
    for path, frame in ((RUN_RESULTS_FILE, runs), (RESULTS_FILE, pd.DataFrame(summaries))):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


def main():
    pairs = find_datasets(Path(BASE_DIR))
    if not pairs:
        raise FileNotFoundError(f"No dyngen_*_mat.csv / _net.csv pairs under {Path(BASE_DIR).resolve()}")
    print(f"Using device: {device}. Found {len(pairs)} networks: {[p[0] for p in pairs]}. Seeds: {SEEDS}")
    print(f"Split protocol: {SPLIT_MODE}; count transform: {COUNT_TRANSFORM}; "
          f"training objective: balanced sampled BCE.")
    records = []
    for dataset_name, net_path, exp_path in pairs:
        print(f"\nRunning {N_RUNS} seeds on: {dataset_name}")
        data, load_error = None, None
        try:
            data = load_dataset(net_path, exp_path)
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
            print(f"  Load failed: {load_error}")
        for run, seed in enumerate(SEEDS):
            split_path = SPLITS_DIR / dataset_name / f"seed_{seed}.pt"
            record = dict(Dataset=dataset_name, Run=run + 1, Seed=seed, Status="error",
                          AUROC=np.nan, AUPRC=np.nan, Error="", SplitFile=str(split_path),
                          AhatFile="",
                          HiddenDim=HIDDEN_DIM, LatentDim=LATENT_DIM, Dropout=DROPOUT,
                          UseImportance=USE_IMPORTANCE, LambdaExpr=LAMBDA_EXPR,
                          ReverseMessages=REVERSE_MESSAGES, UseS1=True, UseS2=True,
                          MaskRatio=MASK_RATIO, ReconstructionLoss=RECONSTRUCTION_LOSS,
                          SplitMode=SPLIT_MODE, Normalization=NORMALIZATION,
                          CountTransform=COUNT_TRANSFORM, DropZeroGenes=DROP_ZERO_GENES,
                          AllExpressionTargets=(SPLIT_MODE == "gaedgrn" or INCLUDE_ALL_EXPRESSION_TARGETS),
                          AllGeneSources=(SPLIT_MODE == "gaedgrn"),
                          TrainingObjective="balanced_sampled_bce",
                          SelectionMetric=SELECTION_METRIC, LearningRate=LEARNING_RATE,
                          MaxEpochs=NUM_EPOCHS, EdgeDropout=EDGE_DROPOUT)
            try:
                if load_error:
                    raise ValueError(load_error)
                record.update(run_training_once(data, seed, split_path, dataset_name))
                record["Status"] = "ok"
                print(f"  Run {run+1}/{N_RUNS}, seed={seed}: "
                      f"AUROC={record['AUROC']:.4f}, AP={record['AUPRC']:.4f}")
            except Exception as exc:
                record["Error"] = f"{type(exc).__name__}: {exc}"
                print(f"  Run {run+1}/{N_RUNS} failed: {record['Error']}")
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            records.append(record)
            save_results(records)
    print(f"\nSummary: {RESULTS_FILE}\nPer-run metrics/errors: {RUN_RESULTS_FILE}\nA_hat: {AHAT_DIR}")
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()