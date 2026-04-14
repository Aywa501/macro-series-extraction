# Idiom Index: Monetary Value Probing in BERT Representations

## Overview

This project investigates whether pre-trained language models — specifically `bert-base-uncased` and `emanjavacas/MacBERTh` — implicitly encode historical monetary value in their internal representation space, and whether those representations track real economic history.

The approach is **corpus-free**: rather than extracting idioms from text, the analysis constructs short synthetic probe sentences (e.g. *"the coin was worth a farthing"*, *"in 1650, a shilling was used for everyday purchases"*) and passes them through frozen model weights. CLS embeddings at each transformer layer are projected onto hand-built semantic axes and correlated against real historical series from the Bank of England Millennium dataset.

The central question: does the geometry of a language model's embedding space reflect centuries of monetary history it was never explicitly trained to encode?

## Setup

### Prerequisites

- Python 3.11+
- Approximately 4–8 GB of RAM; MPS/CUDA recommended

### Installation

```bash
pip install -r requirements.txt
```

### Input data

One external file is required:

| Path | Contents |
|---|---|
| `a-millennium-of-macroeconomic-data-for-the-uk.xlsx` | Bank of England Millennium of Macroeconomic Data (place in the project root or one level above) |

No corpus, no API keys, no additional downloads beyond the HuggingFace model weights (fetched automatically on first run).

## Running the Analysis

Scripts live in `src/analysis_1/` and must be run in order. All output goes to `data/{model}/value_probe/`.

```bash
# Step 1: Build the value axis and time axis from layer 4 CLS embeddings
python src/analysis_1/01_build_axes.py --model bert

# Step 2: Test implicit historical sequences against the year-probe direction across all 12 layers
python src/analysis_1/02_cross_temporal.py --model bert

# Step 3: Embed coins with explicit year context; correlate with BoE purchasing power data
python src/analysis_1/03_coin_value_probe.py --model bert

# Step 4: Test 5 broader BoE macroeconomic series against custom semantic axes
python src/analysis_1/04_broader_series.py --model bert
```

Swap `--model bert` for `--model macberth` to run with the historical English model. Run `sweep_best_layer.py` first if using MacBERTh to identify the optimal layer:

```bash
python src/analysis_1/sweep_best_layer.py --model macberth
# Then pass the recommended --layer flag to scripts 01–04
```

### Script summary

| Script | What it does | Outputs |
|---|---|---|
| `_common.py` | Shared constants (denominations, templates, year ranges, sequence definitions) | — |
| `01_build_axes.py` | Builds value axis (sovereign − farthing) and time axis (late − early year probes) at a given layer; reports Pearson r and Spearman ρ | `value_direction_L{n}.npy`, `time_direction_L{n}.npy`, `axes_summary.csv`, `plots/axes_quality.png` |
| `02_cross_temporal.py` | Projects four implicit monotonic sequences (dynasties, weapons, ships, fuels) onto the year-probe direction across all 12 layers | `cross_temporal_results.csv`, `plots/cross_temporal.png` |
| `03_coin_value_probe.py` | Embeds coins in year-contextualised sentences, projects onto the value axis, correlates with BoE CPI and real earnings | `coin_value_results.csv`, `correlation_summary.csv`, `plots/coin_value_vs_real.png` |
| `04_broader_series.py` | Builds bespoke semantic axes for wheat prices, coin supply, population, wages, and trade volume; correlates projections with BoE series | `broader_series_results.csv`, `broader_series_correlations.csv`, `plots/broader_series.png` |
| `sweep_best_layer.py` | Sweeps all 12 layers for value-axis Pearson r and time-axis Spearman ρ; saves directions for the best layer | `value_direction_L{n}.npy`, `time_direction_L{n}.npy` |

### Script dependencies

`02_cross_temporal.py` and `03_coin_value_probe.py` require the `.npy` direction files produced by `01_build_axes.py` (or `sweep_best_layer.py`).

## Models

| Key | Model ID | Notes |
|---|---|---|
| `bert` | `bert-base-uncased` | Baseline; modern English pre-training |
| `macberth` | `emanjavacas/MacBERTh` | Pre-trained on historical English (1450–1950) |

Both are loaded from HuggingFace with `output_hidden_states=True`. No fine-tuning is performed.

## Project Structure

```
idiom-index/
├── config/
│   └── config.yaml
├── data/
│   └── {model}/
│       └── value_probe/
│           ├── value_direction_L{n}.npy
│           ├── time_direction_L{n}.npy
│           ├── axes_summary.csv
│           ├── cross_temporal_results.csv
│           ├── coin_value_results.csv
│           ├── correlation_summary.csv
│           ├── broader_series_results.csv
│           ├── broader_series_correlations.csv
│           └── plots/
├── outputs/
├── src/
│   ├── analysis_1/
│   │   ├── _common.py
│   │   ├── 01_build_axes.py
│   │   ├── 02_cross_temporal.py
│   │   ├── 03_coin_value_probe.py
│   │   ├── 04_broader_series.py
│   │   └── sweep_best_layer.py
│   └── exploratory/
├── requirements.txt
└── README.md
```
