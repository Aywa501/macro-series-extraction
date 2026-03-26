# Idiom Index: Monetary Denomination Triviality in Hansard

## Overview

This project builds a computational linguistics pipeline to study how British monetary idioms — phrases like *"not worth a farthing"*, *"a pretty penny"*, or *"penny wise pound foolish"* — changed in semantic weight and connotation across the 19th and 20th centuries. Parliamentary debate transcripts from Hansard provide a dense, datable corpus of English prose; idioms embedded in that prose serve as indirect signals of how contemporaries perceived the real value of denominations. The project extracts, disambiguates, embeds, scores, and regresses these signals against historical price-index data.

The pipeline is fully reproducible and staged (stages 01–06). Each stage writes intermediate artefacts to `data/`, enabling resumption after interruption without recomputing prior work. The key output is a yearly triviality index *S_t* — a scalar measuring how "dismissive" or "insignificant" the connotations of monetary idioms were in a given year — which is then regressed against the log real purchasing power of each denomination. If idioms track monetary salience, we would expect a negative relationship: as a coin's real value fell, references to it should cluster toward the "trivial" end of the semantic axis.

## Setup

### Prerequisites

- Python 3.11+
- AWS credentials with Bedrock access (for stage 02 only)
- Approximately 8 GB of RAM; MPS/CUDA recommended for stage 03

### Installation

```bash
cd project/
pip install -r requirements.txt
```

### NLTK data

Stage 01 requires the `punkt` sentence tokenizer:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### AWS credentials

Stage 02 calls AWS Bedrock. Configure credentials via any standard boto3 method:

```bash
# Option A: environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1          # optional; defaults to us-east-1

# Option B: AWS CLI profile
aws configure
```

No credentials are hardcoded in the codebase.

## Input Data

Populate the following before running:

| Path | Contents |
|---|---|
| `data/raw/hansard/` | Hansard XML files. Filenames should match `YYYY_*.xml` for automatic year extraction, or carry a `date` attribute in the XML. |
| `data/raw/price_index.csv` | Two columns: `year` (int) and `price_index` (float). Any normalisation works; the pipeline normalises to 1800 = 100. |

## Running the Pipeline

Run stages in order from the project root:

```bash
# Stage 01: Extract candidates from Hansard XML
python src/01_extract.py

# Stage 02: Disambiguate via AWS Bedrock (LLM classification)
python src/02_disambiguate.py

# Stage 03: Embed idiomatic observations with transformer models
python src/03_embed.py

# Stage 04: Score embeddings (axis projection, cosine similarity, centroid drift)
python src/04_score.py

# Stage 05: Build triviality index S_t; merge with price data; plot
python src/05_index.py

# Stage 06: Panel and time-series regressions
python src/06_regression.py
```

### Common options

All scripts accept:

| Flag | Meaning |
|---|---|
| `--data-dir PATH` | Override the data root (default: `<project_root>/data`). |
| `--config PATH` | Override the idioms config (stage 01 only). |
| `--force` | Ignore checkpoints / existing outputs and recompute. |
| `--log-level` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. |

Stage 03 additionally accepts `--model macberth` or `--model bge` to embed with a single model.

## Resuming Interrupted Runs

- **Stage 01** tracks processed XML files in `data/interim/.extract_progress.json`. Restarting picks up where it left off. Use `--force` to reprocess all files.
- **Stage 02** checks whether each candidate `id` already appears in `data/interim/observations.parquet`. Restarting is safe. Use `--force` to reclassify everything.
- **Stages 03–06** check for existing output files and skip if present. Use `--force` to recompute.

## Running Tests

```bash
pytest tests/
```

Tests mock the Bedrock API — no AWS credentials required. 20 curated test cases cover clear idiomatic, clear literal, and ambiguous contexts.

## Replication Notes

### Model identifiers

| Role | Model ID |
|---|---|
| Disambiguation LLM | `meta.llama3-1-8b-instruct-v1:0` (AWS Bedrock) |
| Historical embedding | `emanjavacas/MacBERTh` (HuggingFace) |
| Modern English embedding | `BAAI/bge-large-en-v1.5` (HuggingFace) |

### Prompt version

The disambiguation prompt is defined in `src/02_disambiguate.py` as `USER_TEMPLATE` (constant). The system prompt is `SYSTEM_PROMPT`. Together these constitute prompt version 1. Any change to either string constitutes a new prompt version and invalidates prior `bedrock_raw.jsonl` entries for comparison purposes.

### Verifying results without AWS

All raw Bedrock responses are appended line-by-line to `data/logs/bedrock_raw.jsonl` immediately after each API call. To replay classification without calling AWS:

1. Confirm `data/logs/bedrock_raw.jsonl` is present.
2. Parse each line (each is a JSON object with keys `id`, `raw_response`, `timestamp`, `model_id`).
3. Pass `raw_response` through `parse_llm_response()` from `src/02_disambiguate.py`.
4. Cross-reference against `data/interim/observations.parquet` by `id`.

This allows auditing, error analysis, and prompt-version comparisons without re-incurring API costs.

## Project Structure

```
project/
├── config/
│   └── idioms.yaml          # Idiom definitions and metadata
├── data/
│   ├── raw/
│   │   ├── hansard/         # Input: Hansard XML files (user-populated)
│   │   └── price_index.csv  # Input: historical price index (user-populated)
│   ├── interim/             # Intermediate: candidates + observations parquet
│   ├── processed/
│   │   └── embeddings/      # .npy embedding arrays + index.parquet
│   └── logs/
│       └── bedrock_raw.jsonl
├── src/
│   ├── 01_extract.py
│   ├── 02_disambiguate.py
│   ├── 03_embed.py
│   ├── 04_score.py
│   ├── 05_index.py
│   ├── 06_regression.py
│   └── utils/
│       ├── checkpoint.py
│       ├── hansard_parser.py
│       └── bedrock_client.py
├── notebooks/
│   ├── 00_eda.ipynb
│   └── 01_results.ipynb
├── tests/
│   └── test_disambiguation.py
├── outputs/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── README.md
```
