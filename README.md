# TopGAE: A topology-informed regulatory influence-weighted graph autoencoder for gene regulatory network reconstruction from single-cell transcriptomics
Shreyansh Priyadarshi and Kedar Natarajan

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-276DC3)](https://www.r-project.org/)

> **Code accompanying the manuscript**

---

## Overview

Reconstructing directed gene regulatory networks (GRNs) from single-cell RNA-seq (scRNA-seq) data is challenging due to high sparsity, technical dropout, and cellular heterogeneity, while existing graph models often mask individual transcription factor (TF) influences or overfit incomplete priors. This repository provides TopGAE, a topology-informed graph autoencoder that preserves regulatory asymmetry and jointly reconstructs network links and masked expression profiles by scoring TF-gene edges via allocation coefficients and bounded interaction strengths.

**How TopGAE works (Fig. 1 of the manuscript)**

1. **Inputs.** A genes x cells expression matrix (standardised per cell) and a partially known, directed prior network. Every gene has a separate *source* (regulator) and *target* role. Prior edges are split 85% / 5% / 10% into train / validation / test, each paired with an equal number of sampled negative pairs; only training edges are used for message passing.
2. **Role-specific gates.** Four structural descriptors per gene (log out-degree, log in-degree, HITS hub and authority scores, from the training edges) are combined with the expression embedding to gate each role.
3. **Dual-score message passing.** Every training edge *i -> j* gets two scores from the pair of gated embeddings: **S1**, a softmax over the incoming regulators of target *j* (relative contribution), and **S2**, a bounded factor in (0, 2) (interaction strength). Messages are weighted by S1 x S2; a reverse pass lets targets send information back to their regulators.
4. **Decoding.** Source and target embeddings are projected to separate latent spaces and scored by an asymmetric bilinear decoder (so *i -> j* and *j -> i* differ). A second pass masks 20% of genes in both roles and a feature decoder reconstructs their expression. The link (BCE) and reconstruction losses are optimised together; the checkpoint with the best validation AUPRC is kept.

The predictions are the decoder logits for every candidate regulator -> target pair. Edges carry no sign (activation vs. repression).

---

## Repository structure

```
TopGAE/
├── topgae/                      # the TopGAE model as an installable Python package
│   ├── __init__.py              #   public API (import everything from here)
│   ├── model.py                 #   TopGAE: encoder, bilinear link decoder, feature decoder
│   ├── layers.py                #   DualScoreLayer: role-specific gates, S1/S2 scores, message passing
│   ├── structure.py             #   structural descriptors (degree + HITS hub/authority)
│   ├── losses.py                #   gene masking and masked-expression reconstruction loss
│   ├── train.py                 #   train_topgae(): training loop, early stopping, evaluation
│   ├── splits.py                #   train/val/test edge splits with fixed balanced negatives
│   ├── data.py                  #   build_adjacency(), prepare_inputs()
│   └── utils.py                 #   seeding and scatter helpers (no torch_geometric needed)
├── run_topgae.py                # run TopGAE on YOUR OWN .h5ad file + prior network
├── experiments/                 # code to reproduce the manuscript
│   ├── beeline/
│   │   ├── run_beeline.py                     # BEELINE benchmark, 42 configurations (Fig. 3, Table S2)
│   │   ├── run_beeline_cell_subset.py         # 10-100% of cells (Fig. S2, Table S2)
│   │   └── run_beeline_network_subset.py      # 10-100% of training edges (Fig. S3, Table S2)
│   ├── simulations/
│   │   ├── simulate_dyngen.R                  # generate the dyngen datasets (Fig. 2A, Fig. S1)
│   │   ├── run_dyngen.py                      # TopGAE on dyngen data (Fig. 2B, Table S1)
│   │   ├── simulate_sergio.ipynb              # generate the SERGIO datasets + plots (Fig. 2C-D)
│   │   └── run_sergio.py                      # TopGAE on SERGIO data (Fig. 2D, Table S1)
│   └── perturbation/
│       ├── perturb_knockdown.ipynb            # CRISPRi TF knockdown: subset, DE, training, evaluation (Fig. 4A, S6A, Table S3)
│       ├── perturb_overexpression.ipynb       # TF overexpression: subset, DE, training, evaluation (Fig. 4B-C, S6B-C, Table S3)
│       └── perturb_data_overview.ipynb        # overview plots of both perturbation datasets (Fig. S4, S5)
├── pyproject.toml               # makes `topgae` pip-installable
├── requirements.txt             # all Python dependencies (package + experiments)
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Natarajanlab/TopGAE.git
cd TopGAE
conda create -n topgae python=3.11 -y
conda activate topgae
pip install -e .                  # the topgae package (numpy, pandas, scipy, scikit-learn, torch)
pip install -r requirements.txt   # + anndata, scanpy, matplotlib, seaborn, jupyter for run_topgae.py and the experiments
```

Install the PyTorch build that matches your CUDA version first if you want GPU training (see https://pytorch.org). A GPU is recommended but not required. TopGAE does not need `torch_geometric` or `torch_scatter`.

For the simulations you also need:

* **dyngen** (R >= 4.0): `install.packages(c("tidyverse", "Matrix"))`, then `remotes::install_github(c("dynverse/dyngen", "dynverse/dyno"))` (installs `dynplot` too).
* **SERGIO**: `git clone https://github.com/PayamDiba/SERGIO` and add the cloned repository root to `PYTHONPATH` so that `from SERGIO.sergio import sergio` works.

---

## Run TopGAE on your own data

`run_topgae.py` takes an `.h5ad` file and a prior network, trains TopGAE and writes a ranked list of regulator -> target predictions.

### 1. Prepare the inputs

**Expression (`.h5ad`)**: an AnnData object with **cells as rows and genes as columns**.

* `adata.X` (or a layer, `--layer counts`) holds either **raw counts** (integers) or **log-normalised** expression. Raw counts are normalised to 10,000 counts per cell and `log1p`-transformed automatically; log-normalised data are used as they are. Do not pass scaled (z-scored) data.
* Gene names in `adata.var_names` (or in a column of `adata.var`, `--gene-column gene_symbols`) must use the same identifiers as the network (e.g. both HGNC symbols, or both Ensembl IDs).
* Use one coherent population of cells; TopGAE learns a context-specific network from them. Select a cell type or condition with `--obs-key` / `--obs-values` (in the paper's perturbation analyses the model was trained on control cells only).

**Prior network (CSV or TSV with a header)**: one directed edge per row, regulator first. Column names are detected automatically (`Gene1`/`Gene2`, `Source`/`Target`, `from`/`to`, `TF`/`target`, ...) or set with `--source-col` / `--target-col`. Extra columns are ignored. For example:

```
Source,Target
SOX2,NANOG
POU5F1,NANOG
MYC,NPM1
```

The prior can come from ChIP-seq (e.g. SC-MO-GRN-DB), motif or curated TF-target databases (e.g. DoRothEA, TRRUST) or a BEELINE-style ground truth. Self-loops and duplicate edges are removed; at least ~20 prior edges between measured genes are needed.

### 2. Run

```bash
python run_topgae.py --h5ad my_data.h5ad --network prior_network.csv --out-dir topgae_results
```

More examples:

```bash
# one cell type only, raw counts in a layer, 5 seeds as in the paper
python run_topgae.py --h5ad atlas.h5ad --network prior.tsv --layer counts --obs-key cell_type --obs-values "HSC" --seeds 42 43 44 45 46 --out-dir results_hsc
```

```bash
# mouse data with lower-case symbols vs. an upper-case network, at most 5,000 cells
python run_topgae.py --h5ad mouse.h5ad --network mouse_prior.csv --uppercase --max-cells 5000
```

```bash
# BEELINE-style protocol: every gene is a candidate source and target, 1,000 most variable genes + all regulators
python run_topgae.py --h5ad my_data.h5ad --network prior.csv --candidates all-pairs --n-top-genes 1000
```

Run `python run_topgae.py --help` for all options. The most important ones:

| Option | Default | Meaning |
|---|---|---|
| `--candidates` | `tf-targets` | `tf-targets`: sources are the regulators of the prior network, targets are all kept genes (scales to whole transcriptomes; used for the hESC perturbation analyses). `all-pairs`: every gene is both source and target and negatives come from all gene pairs (BEELINE / simulation protocol; memory grows with genes^2, use `--n-top-genes`). |
| `--obs-key`, `--obs-values` | all cells | Train on a subset of cells, e.g. `--obs-key condition --obs-values control`. |
| `--layer`, `--gene-column` | `X`, `var_names` | Where to find the expression values and gene names. |
| `--input` | `auto` | `counts` (normalise + log1p), `log` (use as is) or `auto` (counts if all values are integers). |
| `--min-cells` | `10` | Keep target genes detected in at least this many cells. |
| `--n-top-genes` | all | Keep only the N most variable target genes (regulators are always kept). |
| `--max-cells` | all | Randomly subsample cells. Each cell is an input feature, so this bounds memory and time. |
| `--seeds` | `42` | One model per seed; scores are averaged and their SD reported. |
| `--no-structure` | off | Disable the structural (degree + HITS) gates (used for the SERGIO benchmark). |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:1`, ... |
| `--top-edges` | all | Only write the N best-scoring edges. |

Model hyperparameters (`--hidden-dim`, `--latent-dim`, `--lambda-expr`, `--epochs`, ...) default to the values used in the paper's experiment scripts (see [Hyperparameters](#hyperparameters)).

### 3. Outputs (in `--out-dir`)

| File | Content |
|---|---|
| `ranked_edges.csv` | All candidate regulator -> target pairs (self-pairs excluded), best first: `Rank`, `Source`, `Target`, `Score` (decoder logit, mean over seeds), `ScoreSD` (with more than one seed), `Probability` (sigmoid of `Score`) and `InPrior` (edge is in your prior network). Rows with `InPrior == False` are new candidate interactions. |
| `metrics.csv` | Per seed: AUROC / AUPRC for recovering held-out prior edges (10% test split), best epoch, best validation AUPRC and split sizes. Use it to judge how well the model fits your data before trusting the ranking. |
| `model_seed_<seed>.pt` | Trained weights, gene lists, edge splits and the full configuration. |
| `run_config.json` | All options plus numbers of cells, source / target genes and prior edges actually used. |
| `score_matrix.npz` | (`--save-matrix`) dense source x target score matrix with the gene lists. |
| `message_scores_seed_<seed>.csv` | (`--save-message-scores`) S1 / S2 message weights on the training prior edges. These describe how the model weighted known edges; they are not predictions. |

Scores are relative rankings within one run, not calibrated probabilities. Because edges are unsigned, a high score does not tell activation from repression.

### Use TopGAE from Python

```python
import pandas as pd
from topgae import (build_adjacency, prepare_inputs, make_splits, seed_everything,
                    TopGAE, train_topgae)

expr = pd.read_csv("BL--ExpressionData.csv", index_col=0)   # genes x cells, log-normalised, unique gene IDs
net = pd.read_csv("BL--network.csv")                        # columns Gene1 (regulator) and Gene2 (target)
genes = expr.index.tolist()

adj, rows, cols = build_adjacency(net, genes, genes)        # edges outside the gene list are dropped
X_row, X_col, A = prepare_inputs(expr, adj, rows, cols, normalization="cell")

seed_everything(42)
splits = make_splits(A, rows, cols, seed=42, split_mode="gaedgrn")   # "global" for regulator x gene candidates
model = TopGAE(in_features=X_row.shape[1], out_feature=64, latent_dim=32, dropout=0.2)
S1, S2, A_hat, metrics = train_topgae(model, X_row, X_col, A, rows, cols, splits=splits, seed=42,
                                      lambda_expr=0.5, mask_ratio=0.2, reconstruction_loss="cosine")

print(metrics["test_auroc"], metrics["test_auprc"])
scores = pd.DataFrame(A_hat.numpy(), index=rows, columns=cols)       # link logits, source x target
```

For regulator x gene candidates (as in `--candidates tf-targets`), pass the regulators as `rows`, all genes as `cols` and use `split_mode="global"`.

---

## Reproducing the paper

Install the package (`pip install -e .`) and run every script **from its own folder**; data folders and results are relative to that folder. Data are not included in this repository (see [Data](#data) for sources).

### BEELINE benchmark (Fig. 3, Fig. S2, Fig. S3, Table S2)

1. Download the preprocessed BEELINE datasets (as distributed with GENELink / GAEDGRN; raw data: https://doi.org/10.5281/zenodo.3378975) and place them under `experiments/beeline/Dataset/Benchmark_Dataset/`. Each dataset folder must contain `BL--ExpressionData.csv` (genes x cells, gene names in the first column) and `BL--network.csv` (columns `Gene1`, `Gene2`); any nesting works, e.g. `Dataset/Benchmark_Dataset/<network type>/<cell type>/TFs+500/`. Every pair found is run, so keep only the 42 configurations you want.
2. Run:

```bash
cd experiments/beeline
python run_beeline.py                   # Benchmark_TopGAE_Results.csv, Benchmark_TopGAE_Per_Run.csv
python run_beeline_cell_subset.py       # Benchmark_TopGAE_CellFraction_*.csv
python run_beeline_network_subset.py    # Benchmark_TopGAE_NetworkFraction_*.csv
```

`run_beeline.py` performs `N_RUNS` runs per dataset with seeds `42, 43, ...`; Fig. 3 reports 20 runs (seeds 42-61), so set `N_RUNS = 20` to reproduce it. The subsampling scripts use 5 runs (seeds 42-46) per fraction. Edge splits are saved next to the results (`Benchmark_TopGAE_*Splits/`). Results of the seven comparison methods were taken from GAEDGRN and were not re-run.

### Simulations (Fig. 2, Fig. S1, Table S1)

```bash
cd experiments/simulations
# dyngen: one dataset per trajectory backbone -> dyngen_data/dyngen_<name>_{mat,net}.csv
for b in bf branch conv cycle linear tf; do Rscript simulate_dyngen.R $b; done
python run_dyngen.py                    # Dyngen_TopGAE_Results/
```

For SERGIO, run `simulate_sergio.ipynb` three times with `TARGETS_PER_TF = (5, 10)`, `(20, 30)` and `(50, 60)`; it writes `sergio_data/sergio_<a>_<b>_target_Type<1-5>_mat.csv` and `sergio_data/sergio_<a>_<b>_target_net.csv`. Then:

```bash
python run_sergio.py                    # SERGIO_TopGAE_Results/ (structural gates disabled, as in the paper)
```

The last cells of `simulate_sergio.ipynb` plot the dyngen and SERGIO results. Each results folder contains `Summary.csv`, `Per_Run.csv`, the edge splits and, per run, the score matrix (`A_hat_seed_<seed>.csv`) and a BEELINE-style `rankedEdges_seed_<seed>.csv`.

### hESC TF perturbations (Fig. 4, Fig. S4-S6, Table S3)

Put the inputs in `experiments/perturbation/perturb/`:

* `RN203.tsv`: hESC ChIP-seq prior network from SC-MO-GRN-DB (columns `Source`, `Target`, `Relationship`);
* `adata_Training.h5ad`, `adata_Validation.h5ad`, `adata_Test.h5ad`: Virtual Cell Challenge 2025 CRISPRi data;
* `GSE217460_210322_TFAtlas_subsample_raw.h5ad`: TFAtlas data from GEO (gunzip the `.h5ad.gz`).

Then run, in this order, `perturb_knockdown.ipynb`, `perturb_overexpression.ipynb` and `perturb_data_overview.ipynb` (from `experiments/perturbation/`). Each perturbation notebook (i) subsets the data to TFs in the prior, (ii) computes differential expression (Wilcoxon test, Scanpy) against the control cells, (iii) trains TopGAE on the **control cells** with the ChIP-seq prior for 5 seeds (42-46), and (iv) scores each perturbed TF's outgoing edges against its DE genes (|log2FC| >= 0.25, adjusted p < 0.05), both for targets not linked to the TF in the training prior (`unseen_supervision`) and for all targets (`all_targets_descriptive`), with the prior network alone as the baseline. Results are written to `perturb/perturb_kd_validation/`, `perturb/perturb_orf_gfp_validation/` and `perturb/perturb_orf_mCherry_validation/` (`DE_summary.csv`, `DE_metrics_per_seed.csv`, `DE_topk_*.csv`, per-TF predictions and model checkpoints).

---

## Data

All data are public; none are stored in this repository.

| Dataset | Used for | Description | Source |
|---|---|---|---|
| **BEELINE** (curated scRNA-seq) | Fig. 3, S2, S3; Table S2 | 7 cell types: human embryonic stem cells (hESC), human mature hepatocytes (hHEP), mouse dendritic cells (mDC), mouse embryonic stem cells (mESC) and mouse haematopoietic stem cells of the erythroid (mHSC-E), granulocyte-monocyte (mHSC-GM) and lymphoid (mHSC-L) lineages. 3 ground-truth networks per cell type (cell-type-specific ChIP-seq, non-specific ChIP-seq, STRING) and 2 gene sets (all TFs + 500 or + 1,000 most variable genes) = **42 configurations**. | Pratapa et al., *Nat Methods* 2020; https://doi.org/10.5281/zenodo.3378975; preprocessed files as provided with GENELink (Chen & Liu, *Bioinformatics* 2022) |
| **dyngen** (simulated) | Fig. 2A-B, S1; Table S1 | 6 trajectory backbones: bifurcating (`bf`), branching (`branch`), converging (`conv`), cycling (`cycle`), linear (`linear`), trifurcating (`tf`). Each: 1,000 cells, 500 target genes, 500 housekeeping genes; the number of TFs follows from the backbone's regulatory modules (`set.seed(11)`). Ground truth = dyngen feature network. | Generated with `simulate_dyngen.R` (Cannoodt et al., *Nat Commun* 2021) |
| **SERGIO** (simulated) | Fig. 2C-D; Table S1 | 3 network densities with 5-10, 20-30 or 50-60 targets per TF; 60 TFs + 550 target genes (610 genes); K = 2.5, Hill coefficient = 2.0. TFs: 20% housekeeping, 30% random, 50% cell-type-specific. 5 cell types x 1,000 cells (noise 1, decay 0.8, sampling state 15, `dpd` noise) with outlier genes, library-size effects and dropout, converted to UMI counts. Each cell type is evaluated separately against the shared network (15 datasets). | Generated with `simulate_sergio.ipynb` (Dibaeinia & Sinha, *Cell Systems* 2020) |
| **hESC TF knockdown** (CRISPRi) | Fig. 4A, S4, S6A; Table S3 | H1 hESC CRISPRi data from the Arc Institute Virtual Cell Challenge 2025. Training + validation + test sets merged: 491,046 cells x 18,080 genes, 300 perturbed genes + non-targeting control. Kept: the **13 perturbed TFs present in the prior** (ATF4, CTBP2, CTNNB1, EP300, KDM1A, KDM2B, MAX, RFX5, RNF2, SOX2, TCF3, TET1, USF2) and 1,000 random non-targeting control cells, genes shared with the prior: **17,179 cells x 17,430 genes**. | Roohani et al., *Cell* 2025; https://console.cloud.google.com/storage/browser/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025 |
| **hESC TF overexpression** (TFAtlas ORF screen) | Fig. 4B-C, S5, S6B-C; Table S3 | `GSE217460_210322_TFAtlas_subsample_raw.h5ad`: 671,453 cells x 37,528 genes, 1,714 overexpressed TFs + GFP and mCherry controls. Kept: TFs present in the prior with >= 100 cells (**36 TFs**: ATF2, ATF3, BCL11A, CBX2, CTBP2, CTCF, E2F6, EZH2, FOSL1, GTF2F1, JUN, JUND, KLF4, KLF5, LEF1, MAFK, MAX, MXI1, MYC, NANOG, POU5F1, RNF2, RXRA, SIRT6, SMAD1, SMAD2, SMAD3, TAF7, TBP, TCF12, TCF3, TEAD4, USF1, USF2, YAP1, YY1) and both controls, genes shared with the prior: **36,819 cells x 23,413 genes**. Compared against GFP and against mCherry separately. | Joung et al., *Cell* 2023; GEO [GSE217460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217460) |
| **hESC ChIP-seq prior** (`RN203.tsv`) | Prior network for both perturbation datasets | Unperturbed hESC ChIP-seq network: 822,475 TF -> target edges, **96 TFs, 27,386 target genes**. | SC-MO-GRN-DB (Valensi et al., *iScience* 2026) |

**Preprocessing.** Expression matrices are used as genes x cells. Duplicate edges and self-interactions are removed and only edges with both genes in the expression matrix are kept; genes without prior edges stay in the candidate universe. Expression is standardised within each cell across genes before training. dyngen and SERGIO raw counts are library-size normalised (to the median library size) and `log1p`-transformed by `run_dyngen.py` / `run_sergio.py`; perturbation counts are normalised to 10,000 per cell and `log1p`-transformed.

---

## Hyperparameters

Values used by the experiment scripts and notebooks (and the defaults of `run_topgae.py`):

| Setting | Value |
|---|---|
| Hidden width / latent dimension | 64 / 32 |
| Input dropout | 0.2 |
| Masked-expression loss | squared cosine error, weight lambda = 0.5, 20% of genes masked per epoch |
| Optimiser | AdamW, learning rate 5e-4, weight decay 5e-4, gradient-norm clipping 5.0 |
| Training | up to 2,000 full-graph epochs; validation every 5 epochs; early stopping after 50 checks without improvement; checkpoint with best validation AUPRC |
| Edge split | 85% train / 5% validation / 10% test positives, each with an equal number of fixed sampled negatives |
| Expression normalisation | per-cell standardisation across genes |
| Structural gates | on (off for SERGIO) |
| Seeds | 42-46 (BEELINE Fig. 3: 42-61). The seed sets the edge split, negative sampling and initialisation. |

---

## Data Availability

Synthetic and scRNA-seq data are publicly available from the sources listed here as well as the manuscript. The notebooks used for pre-processing data and analysis are included in this repository. No raw data files are committed to this repository.

---

## Citation

```bibtex
@article{priyadarshi2026topgae,
  title={TopGAE: A topology-informed regulatory influence-weighted graph autoencoder for gene regulatory network reconstruction from single-cell transcriptomics},
  author={Priyadarshi, Shreyansh and Natarajan, Kedar},
  journal={Under review},
  year={2026}
}
@software{priyadarshi2026topgae_code,
  title={TopGAE: A topology-informed regulatory influence-weighted graph autoencoder for gene regulatory network reconstruction from single-cell transcriptomics},
  author={Priyadarshi, Shreyansh and Natarajan, Kedar},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.24352290},
  url={https://doi.org/10.5281/zenodo.24352290}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.
