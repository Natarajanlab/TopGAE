"""TopGAE on BEELINE with nested cell subsampling (10/25/50/75/100% of cells; Fig. S2).

Install the package first (pip install -e . in the repository root), then run
from this folder:  python run_beeline_cell_subset.py
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topgae import (
    build_adjacency,
    prepare_inputs,
    seed_everything,
    make_splits,
    TopGAE,
    train_topgae,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("./Dataset/Benchmark_Dataset")

# Cell fractions to evaluate
CELL_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]

# Five independent runs per cell fraction
N_RUNS = 5

BASE_SEED = 42

RESULTS_FILE = Path("Benchmark_TopGAE_CellFraction_Results.csv")
RUN_RESULTS_FILE = Path("Benchmark_TopGAE_CellFraction_Per_Run.csv")

SPLITS_DIR = Path("Benchmark_TopGAE_CellFraction_Splits")


# ------------------------------------------------------------
# Model / training configuration
# ------------------------------------------------------------

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


# Used only for global/per_tf
INCLUDE_ALL_EXPRESSION_TARGETS = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(network_path, expression_path):
    """
    Load the complete dataset.

    Returns:
        net
        exp

    We deliberately do NOT prepare the expression data here,
    because cell subsampling needs to happen independently
    for every run.
    """

    net = pd.read_csv(network_path)

    exp = pd.read_csv(expression_path, index_col=0)

    if not {"Gene1", "Gene2"}.issubset(net.columns):
        raise ValueError(
            "Network CSV needs Gene1 and Gene2 columns."
        )

    if net[["Gene1", "Gene2"]].isna().any().any():
        raise ValueError(
            "Network contains missing gene identifiers."
        )

    # Normalize identifiers
    net["Gene1"] = net["Gene1"].astype(str).str.strip()
    net["Gene2"] = net["Gene2"].astype(str).str.strip()

    exp.index = exp.index.astype(str).str.strip()

    if not exp.index.is_unique:
        raise ValueError(
            "Expression gene identifiers are duplicated "
            "after normalization."
        )

    exp = exp.apply(pd.to_numeric, errors="raise")

    # Remove self-loops and duplicate edges
    net = net[
        net.Gene1.ne(net.Gene2)
    ].drop_duplicates(
        ["Gene1", "Gene2"]
    )

    # Keep only genes present in expression
    net = net[
        net.Gene1.isin(exp.index)
        & net.Gene2.isin(exp.index)
    ].reset_index(drop=True)

    if net.empty:
        raise ValueError(
            "No positive interactions match expression rows. "
            "Check gene IDs/orientation."
        )

    return net, exp


# ============================================================
# PREPARE ONE CELL-FRACTION DATASET
# ============================================================

def prepare_fraction_data(
    net,
    exp,
    cell_fraction,
    seed,
):
    """
    Select a reproducible fraction of cells and prepare model inputs.

    Cells are columns of exp.

    For each seed, we generate ONE random permutation of all cells.
    Fractions are prefixes of that same permutation:

        10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100%

    This gives a nested cell-subsampling design.
    """

    if not (0 < cell_fraction <= 1.0):
        raise ValueError(
            f"Invalid cell fraction: {cell_fraction}"
        )

    n_cells = exp.shape[1]

    if n_cells < 1:
        raise ValueError("Expression matrix contains no cells.")

    # --------------------------------------------------------
    # Reproducible permutation of cells
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    cell_indices = rng.permutation(n_cells)

    # Number of cells for this fraction
    n_selected = int(np.floor(
        n_cells * cell_fraction
    ))

    # Always select at least one cell
    n_selected = max(1, n_selected)

    # 100% should use every cell
    if cell_fraction == 1.0:
        n_selected = n_cells

    selected_indices = cell_indices[:n_selected]

    # Preserve the selected permutation order.
    # If you prefer original cell ordering, use:
    #
    # selected_indices = np.sort(selected_indices)
    #
    # Either is valid, but consistency is important.
    selected_exp = exp.iloc[:, selected_indices].copy()

    # --------------------------------------------------------
    # Define graph universe
    # --------------------------------------------------------

    if SPLIT_MODE == "gaedgrn":

        # Square gene × gene graph.
        # All expression genes participate as both source
        # and target nodes.
        row_genes = selected_exp.index.tolist()
        col_genes = list(row_genes)

    else:

        row_genes = sorted(
            net.Gene1.unique().tolist()
        )

        col_genes = (
            selected_exp.index.tolist()
            if INCLUDE_ALL_EXPRESSION_TARGETS
            else sorted(net.Gene2.unique().tolist())
        )

    # --------------------------------------------------------
    # Build adjacency
    # --------------------------------------------------------

    adj, row_genes, col_genes = build_adjacency(
        net,
        row_genes,
        col_genes,
    )

    # --------------------------------------------------------
    # Prepare model inputs
    # --------------------------------------------------------

    xr, xc, a = prepare_inputs(
        selected_exp,
        adj,
        row_genes,
        col_genes,
        "cpu",
        normalization=NORMALIZATION,
    )

    return (
        xr,
        xc,
        a,
        row_genes,
        col_genes,
        n_selected,
        selected_exp.columns.tolist(),
    )


# ============================================================
# ONE TRAINING RUN
# ============================================================

def run_training_once(
    net,
    exp,
    cell_fraction,
    seed,
    split_path,
):
    """
    Run one experiment at a specific cell fraction and seed.
    """

    # --------------------------------------------------------
    # Critical: seed BEFORE model initialization
    # --------------------------------------------------------

    seed_everything(seed)

    # --------------------------------------------------------
    # Select cells and prepare inputs
    # --------------------------------------------------------

    (
        xr,
        xc,
        a,
        rows,
        cols,
        n_selected,
        selected_cells,
    ) = prepare_fraction_data(
        net=net,
        exp=exp,
        cell_fraction=cell_fraction,
        seed=seed,
    )

    # --------------------------------------------------------
    # Generate graph split
    #
    # Splits depend on graph/seed, NOT expression cell count.
    # Therefore the same seed produces the same graph split
    # across all cell fractions.
    # --------------------------------------------------------

    splits = make_splits(
        a,
        rows,
        cols,
        seed=seed,
        split_mode=SPLIT_MODE,
    )

    split_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "splits": splits,
            "row_genes": rows,
            "col_genes": cols,
            "seed": seed,
            "cell_fraction": cell_fraction,
            "n_cells": n_selected,
            "selected_cells": selected_cells,
            "split_mode": SPLIT_MODE,
            "negative_universe": (
                "all_gene_pairs"
                if SPLIT_MODE == "gaedgrn"
                else "tf_gene_pairs"
            ),
            "training_objective": "balanced_sampled_bce",
        },
        split_path,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TopGAE(
        in_features=xr.shape[1],
        out_feature=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
        use_importance=USE_IMPORTANCE,
        reverse_messages=REVERSE_MESSAGES,
        use_s1=True,
        use_s2=True,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    s1, s2, a_hat, metrics = train_topgae(
        model,
        xr,
        xc,
        a,
        rows,
        cols,

        device=device,

        seed=seed,
        splits=splits,

        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,

        patience=PATIENCE,
        val_check_freq=VAL_CHECK_FREQ,

        verbose=False,

        lambda_expr=LAMBDA_EXPR,
        mask_ratio=MASK_RATIO,

        reconstruction_loss=RECONSTRUCTION_LOSS,

        edge_dropout=EDGE_DROPOUT,

        selection_metric=SELECTION_METRIC,
    )

    return {
        "AUROC": metrics["test_auroc"],
        "AUPRC": metrics["test_auprc"],
        "BestEpoch": metrics["best_epoch"],
        "BestValScore": metrics["best_val_score"],

        "TrainPositive": (
            metrics["split_counts"]["train"]["positive"]
        ),
        "ValPositive": (
            metrics["split_counts"]["val"]["positive"]
        ),
        "TestPositive": (
            metrics["split_counts"]["test"]["positive"]
        ),

        "NCells": n_selected,
        "SelectedCells": selected_cells,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(records):

    runs = pd.DataFrame(records)

    summaries = []

    for (
        dataset,
        cell_fraction,
    ), group in runs.groupby(
        ["Dataset", "CellFraction"],
        sort=False,
    ):

        ok = group[
            group.Status.eq("ok")
        ]

        n = len(ok)

        summaries.append(
            {
                "Dataset": dataset,

                "CellFraction": cell_fraction,

                "NCells":
                    ok["NCells"].iloc[0]
                    if n
                    else np.nan,

                "RequestedRuns": N_RUNS,

                "AttemptedRuns": len(group),

                "SplitMode": SPLIT_MODE,

                "Runs": n,

                "FailedRuns": int(
                    group.Status.eq("error").sum()
                ),

                "Mean_AUROC":
                    ok.AUROC.mean()
                    if n
                    else np.nan,

                "SD_AUROC":
                    ok.AUROC.std(ddof=1)
                    if n > 1
                    else np.nan,

                "Mean_AUPRC":
                    ok.AUPRC.mean()
                    if n
                    else np.nan,

                "SD_AUPRC":
                    ok.AUPRC.std(ddof=1)
                    if n > 1
                    else np.nan,
            }
        )

    # --------------------------------------------------------
    # Save per-run results and summary
    # --------------------------------------------------------

    for path, frame in (
        (RUN_RESULTS_FILE, runs),
        (RESULTS_FILE, pd.DataFrame(summaries)),
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        frame.to_csv(
            temporary,
            index=False,
        )

        temporary.replace(path)


# ============================================================
# MAIN
# ============================================================

def main():

    if N_RUNS != 5:
        raise ValueError(
            "This experiment is configured for exactly "
            "5 runs per cell fraction."
        )

    if not CELL_FRACTIONS:
        raise ValueError(
            "CELL_FRACTIONS cannot be empty."
        )

    base = Path(BASE_DIR)

    pairs = [
        (
            p,
            p.with_name(
                "BL--ExpressionData.csv"
            ),
        )
        for p in sorted(
            base.rglob("BL--network.csv")
        )
        if p.with_name(
            "BL--ExpressionData.csv"
        ).is_file()
    ]

    if not pairs:
        raise FileNotFoundError(
            f"No paired benchmark CSVs under "
            f"{base.resolve()}"
        )

    print(
        f"Using device: {device}"
    )

    print(
        f"Found {len(pairs)} dataset pairs."
    )

    print(
        f"Cell fractions: {CELL_FRACTIONS}"
    )

    print(
        f"Runs per fraction: {N_RUNS}"
    )

    print(
        f"Total runs per dataset: "
        f"{len(CELL_FRACTIONS) * N_RUNS}"
    )

    print(
        f"Split protocol: {SPLIT_MODE}"
    )

    print(
        "Cell subsampling: nested within each seed"
    )

    print(
        "Training objective: balanced sampled BCE"
    )

    records = []

    # ========================================================
    # DATASETS
    # ========================================================

    for net_path, exp_path in pairs:

        dataset_name = str(
            net_path.parent.relative_to(base)
        )

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"Dataset: {dataset_name}"
        )

        print(
            f"{'=' * 70}"
        )

        # ----------------------------------------------------
        # Load complete dataset ONCE
        # ----------------------------------------------------

        try:

            net, exp = load_dataset(
                net_path,
                exp_path,
            )

            total_cells = exp.shape[1]

            print(
                f"Genes: {exp.shape[0]}"
            )

            print(
                f"Total cells/features: {total_cells}"
            )

        except Exception as exc:

            print(
                f"Dataset loading failed: "
                f"{type(exc).__name__}: {exc}"
            )

            # Still record all intended runs as failures
            for fraction in CELL_FRACTIONS:

                for run in range(N_RUNS):

                    seed = (
                        BASE_SEED + run
                    )

                    record = {
                        "Dataset": dataset_name,
                        "CellFraction": fraction,
                        "Run": run + 1,
                        "Seed": seed,
                        "Status": "error",

                        "NCells": np.nan,

                        "AUROC": np.nan,
                        "AUPRC": np.nan,

                        "BestEpoch": np.nan,
                        "BestValScore": np.nan,

                        "Error":
                            f"{type(exc).__name__}: {exc}",
                    }

                    records.append(record)

            save_results(records)

            continue

        # ====================================================
        # CELL FRACTIONS
        # ====================================================

        for fraction in CELL_FRACTIONS:

            n_cells = max(
                1,
                int(
                    np.floor(
                        total_cells * fraction
                    )
                ),
            )

            if fraction == 1.0:
                n_cells = total_cells

            print(
                f"\n--- {fraction * 100:.0f}% cells "
                f"({n_cells}/{total_cells}) ---"
            )

            # ------------------------------------------------
            # FIVE RUNS
            # ------------------------------------------------

            for run in range(N_RUNS):

                seed = (
                    BASE_SEED + run
                )

                split_path = (
                    Path(SPLITS_DIR)
                    / net_path.parent.relative_to(base)
                    / f"fraction_{fraction:.2f}"
                    / f"seed_{seed}.pt"
                )

                record = dict(

                    Dataset=dataset_name,

                    CellFraction=fraction,

                    Run=run + 1,

                    Seed=seed,

                    Status="error",

                    NCells=n_cells,

                    AUROC=np.nan,
                    AUPRC=np.nan,

                    BestEpoch=np.nan,
                    BestValScore=np.nan,

                    TrainPositive=np.nan,
                    ValPositive=np.nan,
                    TestPositive=np.nan,

                    Error="",

                    SplitFile=str(
                        split_path
                    ),

                    HiddenDim=HIDDEN_DIM,
                    LatentDim=LATENT_DIM,
                    Dropout=DROPOUT,

                    UseImportance=USE_IMPORTANCE,

                    LambdaExpr=LAMBDA_EXPR,

                    ReverseMessages=REVERSE_MESSAGES,

                    UseS1=True,
                    UseS2=True,

                    MaskRatio=MASK_RATIO,

                    ReconstructionLoss=(
                        RECONSTRUCTION_LOSS
                    ),

                    SplitMode=SPLIT_MODE,

                    Normalization=NORMALIZATION,

                    AllExpressionTargets=(
                        SPLIT_MODE == "gaedgrn"
                        or INCLUDE_ALL_EXPRESSION_TARGETS
                    ),

                    AllGeneSources=(
                        SPLIT_MODE == "gaedgrn"
                    ),

                    TrainingObjective=(
                        "balanced_sampled_bce"
                    ),

                    SelectionMetric=(
                        SELECTION_METRIC
                    ),

                    LearningRate=(
                        LEARNING_RATE
                    ),

                    MaxEpochs=NUM_EPOCHS,

                    EdgeDropout=EDGE_DROPOUT,
                )

                try:

                    result = run_training_once(
                        net=net,
                        exp=exp,
                        cell_fraction=fraction,
                        seed=seed,
                        split_path=split_path,
                    )

                    record.update(result)

                    record["Status"] = "ok"

                    print(
                        f"  Run {run + 1}/{N_RUNS} "
                        f"| seed={seed} "
                        f"| cells={n_cells} "
                        f"| AUROC="
                        f"{record['AUROC']:.4f} "
                        f"| AUPRC="
                        f"{record['AUPRC']:.4f}"
                    )

                except Exception as exc:

                    record["Error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    print(
                        f"  Run {run + 1}/{N_RUNS} "
                        f"| seed={seed} "
                        f"FAILED: "
                        f"{record['Error']}"
                    )

                finally:

                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                records.append(record)

                # Save after every run so an interruption
                # doesn't destroy previous results.
                save_results(records)

    # ========================================================
    # FINAL
    # ========================================================

    print(
        f"\n{'=' * 70}"
    )

    print(
        "Experiment complete."
    )

    print(
        f"Summary: {RESULTS_FILE}"
    )

    print(
        f"Per-run results: {RUN_RESULTS_FILE}"
    )

    print(
        f"Splits: {SPLITS_DIR}"
    )

    print(
        f"{'=' * 70}"
    )

    return pd.DataFrame(records)


if __name__ == "__main__":
    main()