"""TopGAE on the curated BEELINE benchmark (Fig. 3, Table S2).

Install the package first (pip install -e . in the repository root), then run
from this folder:  python run_beeline.py   (or call main() in a notebook).
Every BL--network.csv / BL--ExpressionData.csv pair found below BASE_DIR is
treated as one dataset configuration.
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topgae import (
    build_adjacency, prepare_inputs, seed_everything,
    make_splits, TopGAE, train_topgae,
)

# --- CONFIG ---
BASE_DIR = Path("./Dataset/Benchmark_Dataset")
N_RUNS = 20
BASE_SEED = 42
RESULTS_FILE = Path("Benchmark_TopGAE_Results.csv")
RUN_RESULTS_FILE = Path("Benchmark_TopGAE_Per_Run.csv")
SPLITS_DIR = Path("Benchmark_TopGAE_Splits")

# Keep your previous dimensions for the first controlled comparison.
HIDDEN_DIM = 64
LATENT_DIM = 32
DROPOUT = 0.2
NUM_EPOCHS = 2000
PATIENCE = 50
VAL_CHECK_FREQ = 5
LEARNING_RATE = 5e-4

USE_IMPORTANCE = True
REVERSE_MESSAGES = True
LAMBDA_EXPR = 0.5  # Set 0 to disable masked-expression regularization.
MASK_RATIO = 0.2
RECONSTRUCTION_LOSS = "cosine"
EDGE_DROPOUT = 0.0
NORMALIZATION = "cell"
SPLIT_MODE = "gaedgrn"  # Or 'global'/'per_tf' for the earlier TF-only protocol.
SELECTION_METRIC = "auprc"

# Used only for global/per_tf. gaedgrn always uses all expression genes in BOTH roles.
# In global/per_tf, preserve your original edge-derived candidate universe by default.
# If True, ALL rows of this benchmark expression file are candidate targets.
# Prefer a supplied benchmark TF list when available: edge-derived TFs omit
# TFs with no labelled outgoing interactions. Do not change this between methods.
INCLUDE_ALL_EXPRESSION_TARGETS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(network_path, expression_path):
    net = pd.read_csv(network_path)
    exp = pd.read_csv(expression_path, index_col=0)
    if not {"Gene1", "Gene2"}.issubset(net.columns):
        raise ValueError("Network CSV needs Gene1 and Gene2 columns.")
    if net[["Gene1", "Gene2"]].isna().any().any():
        raise ValueError("Network contains missing gene identifiers.")
    # This runner expects the same genes x cells orientation as your old runner.
    net["Gene1"] = net["Gene1"].astype(str).str.strip()
    net["Gene2"] = net["Gene2"].astype(str).str.strip()
    exp.index = exp.index.astype(str).str.strip()
    if not exp.index.is_unique:
        raise ValueError("Expression gene identifiers are duplicated after normalization.")
    exp = exp.apply(pd.to_numeric, errors="raise")
    net = net[net.Gene1.ne(net.Gene2)].drop_duplicates(["Gene1", "Gene2"])
    net = net[net.Gene1.isin(exp.index) & net.Gene2.isin(exp.index)].reset_index(drop=True)
    if net.empty:
        raise ValueError("No positive interactions match expression rows. Check gene IDs/orientation.")
    if SPLIT_MODE == "gaedgrn":
        # The expression file must be the chosen benchmark's selected gene set.
        # Source-role embeddings now include non-TFs to score square-matrix negatives.
        row_genes = exp.index.tolist()
        col_genes = list(row_genes)
    else:
        row_genes = sorted(net.Gene1.unique().tolist())
        col_genes = exp.index.tolist() if INCLUDE_ALL_EXPRESSION_TARGETS else sorted(net.Gene2.unique().tolist())
    adj, row_genes, col_genes = build_adjacency(net, row_genes, col_genes)
    # Keep prepared data on CPU between runs. Training moves it to selected device.
    xr, xc, a = prepare_inputs(exp, adj, row_genes, col_genes, "cpu", normalization=NORMALIZATION)
    return xr, xc, a, row_genes, col_genes


def run_training_once(data, seed, split_path, splits=None):
    xr, xc, a, rows, cols = data
    # Critical: seed BEFORE model initialization, and pass the SAME seed to train.
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
    # S1 and S2 remain enabled. A_hat is logits; S1/S2 cover message edges only.
    s1, s2, a_hat, metrics = train_topgae(
        model, xr, xc, a, rows, cols,
        device=device, seed=seed, splits=splits,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
        patience=PATIENCE, val_check_freq=VAL_CHECK_FREQ, verbose=False,
        lambda_expr=LAMBDA_EXPR, mask_ratio=MASK_RATIO,
        reconstruction_loss=RECONSTRUCTION_LOSS,
        edge_dropout=EDGE_DROPOUT, selection_metric=SELECTION_METRIC,
    )
    return {"AUROC": metrics["test_auroc"], "AUPRC": metrics["test_auprc"],
            "BestEpoch": metrics["best_epoch"], "BestValScore": metrics["best_val_score"],
            "TrainPositive": metrics["split_counts"]["train"]["positive"],
            "ValPositive": metrics["split_counts"]["val"]["positive"],
            "TestPositive": metrics["split_counts"]["test"]["positive"]}


def save_results(records):
    """Save successes AND failures; never report only successful run counts."""
    runs = pd.DataFrame(records)
    summaries = []
    for dataset, group in runs.groupby("Dataset", sort=False):
        ok = group[group.Status.eq("ok")]
        n = len(ok)
        summaries.append({
            "Dataset": dataset, "RequestedRuns": N_RUNS, "AttemptedRuns": len(group),
            "SplitMode": SPLIT_MODE,
            "Runs": n, "FailedRuns": int(group.Status.eq("error").sum()),
            "Mean_AUROC": ok.AUROC.mean() if n else np.nan,
            # Sample SD; undefined for fewer than two successful runs.
            "SD_AUROC": ok.AUROC.std(ddof=1) if n > 1 else np.nan,
            "Mean_AUPRC": ok.AUPRC.mean() if n else np.nan,
            "SD_AUPRC": ok.AUPRC.std(ddof=1) if n > 1 else np.nan,
        })
    for path, frame in ((RUN_RESULTS_FILE, runs), (RESULTS_FILE, pd.DataFrame(summaries))):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


def main():
    if N_RUNS < 1:
        raise ValueError("N_RUNS must be positive.")
    base = Path(BASE_DIR)
    pairs = [(p, p.with_name("BL--ExpressionData.csv"))
             for p in sorted(base.rglob("BL--network.csv"))
             if p.with_name("BL--ExpressionData.csv").is_file()]
    if not pairs:
        raise FileNotFoundError(f"No paired benchmark CSVs under {base.resolve()}")
    print(f"Using device: {device}. Found {len(pairs)} dataset pairs.")
    print(f"Split protocol: {SPLIT_MODE}; training objective: balanced sampled BCE.")
    print("This execution writes a new result table; it does not resume an earlier run.")
    records = []
    for net_path, exp_path in pairs:
        dataset_name = str(net_path.parent.relative_to(base))
        print(f"\nRunning {N_RUNS} times on: {dataset_name}")
        data, load_error = None, None
        try:
            data = load_dataset(net_path, exp_path)
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            split_path = Path(SPLITS_DIR) / net_path.parent.relative_to(base) / f"seed_{seed}.pt"
            record = dict(Dataset=dataset_name, Run=run + 1, Seed=seed, Status="error",
                          AUROC=np.nan, AUPRC=np.nan, Error="", SplitFile=str(split_path),
                          HiddenDim=HIDDEN_DIM, LatentDim=LATENT_DIM, Dropout=DROPOUT,
                          UseImportance=USE_IMPORTANCE, LambdaExpr=LAMBDA_EXPR,
                          ReverseMessages=REVERSE_MESSAGES, UseS1=True, UseS2=True,
                          MaskRatio=MASK_RATIO, ReconstructionLoss=RECONSTRUCTION_LOSS,
                          SplitMode=SPLIT_MODE, Normalization=NORMALIZATION,
                          AllExpressionTargets=(SPLIT_MODE == "gaedgrn" or INCLUDE_ALL_EXPRESSION_TARGETS),
                          AllGeneSources=(SPLIT_MODE == "gaedgrn"),
                          TrainingObjective="balanced_sampled_bce",
                          SelectionMetric=SELECTION_METRIC, LearningRate=LEARNING_RATE,
                          MaxEpochs=NUM_EPOCHS, EdgeDropout=EDGE_DROPOUT)
            try:
                if load_error:
                    raise ValueError(load_error)
                record.update(run_training_once(data, seed, split_path))
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
    print(f"\nSummary: {RESULTS_FILE}\nPer-run metrics/errors: {RUN_RESULTS_FILE}")
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
