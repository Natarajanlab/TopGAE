"""TopGAE on BEELINE with training-network subsampling (10/25/50/75/100% of edges; Fig. S3).

Vary available TRAINING labels; keep all expression cells and held-out pairs fixed.
Install the package first (pip install -e . in the repository root), then run
from this folder:  python run_beeline_network_subset.py   (or call main()).
Fractions apply to the base training pool (~85% of reference positives), NOT
the complete reference network. Training positives AND negatives are reduced.
"""
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from topgae import (
    build_adjacency, prepare_inputs, seed_everything, make_splits,
    validate_splits, TopGAE, train_topgae,
)

BASE_DIR = Path('./Dataset/Benchmark_Dataset')
NETWORK_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
N_RUNS = 5
BASE_SEED = 42
RESULTS_FILE = Path('Benchmark_TopGAE_NetworkFraction_Results.csv')
RUN_RESULTS_FILE = Path('Benchmark_TopGAE_NetworkFraction_Per_Run.csv')
SPLITS_DIR = Path('Benchmark_TopGAE_NetworkFraction_Splits')

# Preserve the settings from the supplied ablation script.
HIDDEN_DIM = 64
LATENT_DIM = 32
DROPOUT = 0.2
NUM_EPOCHS = 2000
PATIENCE = 50
VAL_CHECK_FREQ = 5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
USE_IMPORTANCE = True
REVERSE_MESSAGES = True
LAMBDA_EXPR = 0.5
MASK_RATIO = 0.2
RECONSTRUCTION_LOSS = 'cosine'
EDGE_DROPOUT = 0.0
NORMALIZATION = 'cell'
SPLIT_MODE = 'gaedgrn'
SELECTION_METRIC = 'auprc'
INCLUDE_ALL_EXPRESSION_TARGETS = False
VAL_FRACTION = 0.05
TEST_FRACTION = 0.10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def checked_fraction(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('A network fraction must be a number in (0, 1].')
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('A network fraction must be finite and in (0, 1].')
    return value


def load_dataset(network_path, expression_path):
    """Prepare expression ONCE, with the full reference-label adjacency tensor."""
    net = pd.read_csv(network_path)
    exp = pd.read_csv(expression_path, index_col=0)
    if not {'Gene1', 'Gene2'}.issubset(net.columns):
        raise ValueError('Network CSV needs Gene1 and Gene2 columns.')
    if net[['Gene1', 'Gene2']].isna().any().any() or exp.index.isna().any():
        raise ValueError('Missing gene identifiers.')
    for column in ('Gene1', 'Gene2'):
        net[column] = net[column].astype(str).str.strip()
        if net[column].eq('').any():
            raise ValueError('Empty gene identifiers in network.')
    exp.index = exp.index.astype(str).str.strip()
    if not exp.index.is_unique or (exp.index == '').any():
        raise ValueError('Expression gene IDs must be unique and nonempty.')
    exp = exp.apply(pd.to_numeric, errors='raise')
    net = net[net.Gene1.ne(net.Gene2)].drop_duplicates(['Gene1', 'Gene2'])
    net = net[net.Gene1.isin(exp.index) & net.Gene2.isin(exp.index)]
    if net.empty:
        raise ValueError('No positive interactions match expression genes.')
    if SPLIT_MODE == 'gaedgrn':
        rows = exp.index.tolist()
        cols = list(rows)
    else:
        rows = sorted(net.Gene1.unique().tolist())
        cols = (exp.index.tolist() if INCLUDE_ALL_EXPRESSION_TARGETS
                else sorted(net.Gene2.unique().tolist()))
    # prepare_inputs expects a labelled DataFrame; make_splits expects a tensor.
    adjacency_df, rows, cols = build_adjacency(net, rows, cols)
    xr, xc, full_labels = prepare_inputs(
        exp, adjacency_df, rows, cols, 'cpu', normalization=NORMALIZATION)
    return xr, xc, full_labels, rows, cols


def create_fixed_full_split(full_labels, row_genes, col_genes, seed):
    """Return the original six tensors, with ordering preserved."""
    if not isinstance(full_labels, torch.Tensor):
        raise TypeError('Pass the full adjacency TENSOR returned by prepare_inputs.')
    return make_splits(
        full_labels, row_genes, col_genes, seed=seed, split_mode=SPLIT_MODE,
        val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION)


def make_fraction_split(fixed_split, network_fraction, seed):
    """Nested subsets of fixed training pairs; retain exact val/test tensors.

    Removed known positives become unused labels, NEVER negatives. A minimum
    of one positive/negative is kept; report the achieved fraction for rounding.
    At 100%, return the original base split, including training-edge order.
    """
    fraction = checked_fraction(network_fraction)
    te, ty, ve, vy, ee, ey = fixed_split
    pos = torch.where(ty == 1)[0].detach().cpu().numpy()
    neg = torch.where(ty == 0)[0].detach().cpu().numpy()
    if len(pos) == 0 or len(pos) != len(neg) or len(pos) + len(neg) != len(ty):
        raise ValueError('The base training pool must contain balanced binary labels.')
    if fraction == 1:
        return fixed_split
    n = max(1, math.floor(len(pos) * fraction))
    pos_stream, neg_stream = np.random.SeedSequence(int(seed)).spawn(2)
    p = np.random.default_rng(pos_stream).permutation(pos)[:n]
    q = np.random.default_rng(neg_stream).permutation(neg)[:n]
    # Keep base ordering; no unnecessary full-batch label shuffling.
    keep = torch.as_tensor(np.sort(np.concatenate([p, q])),
                           dtype=torch.long, device=te.device)
    return te[:, keep], ty[keep], ve, vy, ee, ey


def split_metadata(data, full_split, splits):
    xr, xc, full_labels, rows, cols = data
    count = lambda y: int((y == 1).sum().item())
    nbase = count(full_split[1])
    ntrain = count(splits[1])
    nfull = int(full_labels.sum().item())
    record = dict(NCells=int(xr.shape[1]), NSourceGenes=len(rows), NTargetGenes=len(cols),
                  FullNetworkPositive=nfull, BaseTrainPositive=nbase,
                  ActualTrainFraction=ntrain / nbase,
                  FractionOfFullPositiveNetwork=ntrain / nfull)
    for label, y in zip(('Train', 'Val', 'Test'), splits[1::2]):
        record[label + 'Positive'] = count(y)
        record[label + 'Negative'] = int((y == 0).sum().item())
    return record


def run_training_once(data, splits, seed):
    xr, xc, full_labels, rows, cols = data
    validate_splits(splits, full_labels, rows, cols)
    seed_everything(seed)  # before initialization, paired across fractions
    model = TopGAE(
        in_features=xr.shape[1], out_feature=HIDDEN_DIM, latent_dim=LATENT_DIM,
        dropout=DROPOUT, use_importance=USE_IMPORTANCE,
        reverse_messages=REVERSE_MESSAGES, use_s1=True, use_s2=True)
    # full_labels is for checking split labels, NOT graph aggregation.
    # The trainer derives message edges AND HITS/degree features solely from
    # splits[0][:, splits[1] == 1], i.e. this fraction's positive training edges.
    _, _, _, metrics = train_topgae(
        model, xr, xc, full_labels, rows, cols, splits=splits,
        device=device, seed=seed, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
        patience=PATIENCE, val_check_freq=VAL_CHECK_FREQ, verbose=False,
        lambda_expr=LAMBDA_EXPR, mask_ratio=MASK_RATIO,
        reconstruction_loss=RECONSTRUCTION_LOSS, edge_dropout=EDGE_DROPOUT,
        weight_decay=WEIGHT_DECAY, selection_metric=SELECTION_METRIC)
    expected = splits[0][:, splits[1] == 1].cpu()
    if not torch.equal(metrics['message_edge_index'], expected):
        raise RuntimeError('Trainer message graph differs from selected training positives.')
    result = dict(AUROC=metrics['test_auroc'], AUPRC=metrics['test_auprc'],
                  BestEpoch=metrics['best_epoch'], BestValScore=metrics['best_val_score'])
    if not all(math.isfinite(float(result[k])) for k in ('AUROC', 'AUPRC', 'BestValScore')):
        raise FloatingPointError('Nonfinite evaluation metric.')
    return result


def atomic_csv(frame, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_results(records):
    if not records:
        return
    runs = pd.DataFrame(records)
    summaries = []
    for (dataset, fraction), group in runs.groupby(['Dataset', 'NetworkFraction'], sort=False):
        ok = group[group.Status.eq('ok')]
        row = dict(Dataset=dataset, NetworkFraction=fraction, RequestedRuns=N_RUNS,
                   AttemptedRuns=len(group), Runs=len(ok),
                   FailedRuns=int(group.Status.eq('error').sum()))
        for key in ('AUROC', 'AUPRC', 'BestValScore'):
            row['Mean_' + key] = ok[key].mean() if len(ok) else np.nan
            row['SD_' + key] = ok[key].std(ddof=1) if len(ok) > 1 else np.nan
        # Mean counts remain informative if a different splitter has seed-specific counts.
        for key in ('NCells', 'FullNetworkPositive', 'BaseTrainPositive', 'TrainPositive',
                    'TrainNegative', 'ValPositive', 'ValNegative', 'TestPositive',
                    'TestNegative', 'ActualTrainFraction', 'FractionOfFullPositiveNetwork'):
            row[key] = group[key].mean() if key in group else np.nan
        summaries.append(row)
    atomic_csv(runs, RUN_RESULTS_FILE)
    atomic_csv(pd.DataFrame(summaries), RESULTS_FILE)


def new_record(dataset, fraction, run, seed, split_path):
    return dict(Dataset=dataset, NetworkFraction=fraction, Run=run + 1, Seed=seed,
                Status='error', Error='', ErrorStage='', AUROC=np.nan, AUPRC=np.nan,
                BestEpoch=np.nan, BestValScore=np.nan, Seconds=np.nan,
                SplitFile=str(split_path), HiddenDim=HIDDEN_DIM, LatentDim=LATENT_DIM,
                LambdaExpr=LAMBDA_EXPR, ReconstructionLoss=RECONSTRUCTION_LOSS,
                UseS1=True, UseS2=True, ExpressionFraction=1.0,
                FixedValidationSet=True, FixedTestSet=True, SplitMode=SPLIT_MODE,
                SelectionMetric=SELECTION_METRIC, FractionBasis='base_training_positive_pool')


def main():
    if isinstance(N_RUNS, bool) or not isinstance(N_RUNS, int) or N_RUNS < 1:
        raise ValueError('N_RUNS must be a positive integer.')
    fractions = [checked_fraction(f) for f in NETWORK_FRACTIONS]
    if not fractions or len(fractions) != len(set(fractions)):
        raise ValueError('NETWORK_FRACTIONS must be nonempty and unique.')
    base = Path(BASE_DIR)
    pairs = [(p, p.with_name('BL--ExpressionData.csv'))
             for p in sorted(base.rglob('BL--network.csv'))
             if p.with_name('BL--ExpressionData.csv').is_file()]
    if not pairs:
        raise FileNotFoundError(f'No paired benchmark CSVs under {base.resolve()}')
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in globals().items() if k.isupper()}
    config.update(device=str(device), torch_version=str(torch.__version__),
                  fraction_basis='base_training_pool', negative_sampling='fixed_balanced',
                  reciprocal_pairs_grouped=False, expression_cells='all', use_s1=True, use_s2=True)
    config_path = Path(RESULTS_FILE).with_suffix('.config.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Device: {device}; {len(pairs)} datasets; {N_RUNS * len(fractions)} fits/dataset.')
    print('Fractions apply to TRAINING labels; all cells and held-out pairs stay fixed per seed.')
    records = []
    for net_path, exp_path in pairs:
        relative = net_path.parent.relative_to(base)
        dataset = str(relative)
        try:
            data = load_dataset(net_path, exp_path)
            load_error = None
        except Exception as exc:
            data = None
            load_error = f'{type(exc).__name__}: {exc}'
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            setup_error, stage = load_error, 'load'
            if setup_error is None:
                try:
                    _, _, full_labels, rows, cols = data
                    full_split = create_fixed_full_split(full_labels, rows, cols, seed)
                    validate_splits(full_split, full_labels, rows, cols)
                    base_path = Path(SPLITS_DIR) / relative / f'seed_{seed}' / 'full_split.pt'
                    base_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(splits=tuple(t.cpu() for t in full_split), row_genes=rows,
                                    col_genes=cols, seed=seed, split_mode=SPLIT_MODE), base_path)
                except Exception as exc:
                    setup_error = f'{type(exc).__name__}: {exc}'
                    stage = 'base_split'
            for index, fraction in enumerate(fractions):
                path = (Path(SPLITS_DIR) / relative / f'seed_{seed}' /
                        f'fraction_{index + 1:02d}_{fraction!r}.pt')
                record = new_record(dataset, fraction, run, seed, path)
                start = time.perf_counter()
                if setup_error is not None:
                    record.update(Error=setup_error, ErrorStage=stage)
                else:
                    try:
                        stage = 'fraction_split'
                        splits = make_fraction_split(full_split, fraction, seed)
                        validate_splits(splits, full_labels, rows, cols)
                        assert all(torch.equal(a, b) for a, b in zip(splits[2:], full_split[2:]))
                        record.update(split_metadata(data, full_split, splits))
                        torch.save(dict(splits=tuple(t.cpu() for t in splits), row_genes=rows,
                                        col_genes=cols, seed=seed, network_fraction=fraction,
                                        base_split_file=str(base_path), metadata=record.copy(),
                                        message_edge_index=splits[0][:, splits[1] == 1].cpu()), path)
                        stage = 'training'
                        record.update(run_training_once(data, splits, seed))
                        record['Status'] = 'ok'
                    except Exception as exc:
                        record.update(Error=f'{type(exc).__name__}: {exc}', ErrorStage=stage)
                    finally:
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                record['Seconds'] = time.perf_counter() - start
                records.append(record)
                save_results(records)
                outcome = (f"AUROC={record['AUROC']:.4f}, AP={record['AUPRC']:.4f}"
                           if record['Status'] == 'ok' else record['Error'])
                print(f'{dataset} | seed={seed} | training fraction={fraction:g} | {outcome}')
    print(f'Saved: {RESULTS_FILE}\nPer-run results: {RUN_RESULTS_FILE}')
    return pd.DataFrame(records)


if __name__ == '__main__':
    main()
