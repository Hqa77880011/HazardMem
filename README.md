# HazardMem

Implementation of **HazardMem: Transition–Duration Hazard Memory for Long-Horizon Video Anomaly Detection**.

## Installation

Requires Python `>=3.11,<3.13`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

The finite generator creates switching six-channel sensor trajectories with disjoint training/evaluation seeds. Preprocessing extracts 18 current/past-window features; frame truth stays outside model inputs.

```bash
python finite_data.py generate --output data/generated --seed 7
python finite_data.py prepare --source data/generated --output data/features
```

For external causal video features, export `data/videos.csv` with `video,path,split,weak_label`. Feature paths are absolute or relative to the CSV's directory. Each NPZ contains finite `features` of shape T×D. Training uses only video-level `weak_label` (0 normal, 1 anomalous). Test NPZs additionally contain frame/clip-aligned binary `label`, and optional binary `slow` selecting the protocol-fixed slow-event evaluation support. Never define `slow` from test model scores. Use a separate protocol for each dataset and preserve official video splits. Set `--dt` to seconds per feature; default is 0.5 seconds.

## Training and inference

```bash
hazardmem data/features/videos.csv --output outputs/full --states 16 --budget 128 --minimum 1 --maximum 16 --epochs 50 --seed 0
```

## Evaluation

The command fits the weak-label anomaly head and evaluates held-out feature streams.

## Comparisons and ablations

```bash
hazardmem data/features/videos.csv --variant uniform --output outputs/uniform
hazardmem data/features/videos.csv --variant change_only --output outputs/change
hazardmem data/features/videos.csv --variant transition_only --output outputs/transition
hazardmem data/features/videos.csv --variant no_protection --output outputs/no_protection
```

## Outputs

The run writes `model.pkl`, `history.csv`, `predictions.csv`, `metrics.json` and `scores.png` under `--output`. Inspect or execute the full finite full/uniform pipeline with:

```bash
python reproduce.py --plan
python reproduce.py --output outputs/finite-setting
```
