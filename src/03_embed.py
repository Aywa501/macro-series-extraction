"""Stage 03 — Sentence embeddings.

Encodes idiomatic observations using HuggingFace transformer models and
saves embeddings as float32 NumPy arrays alongside a row-index parquet.

Layer selection
---------------
For BERT-family models (MacBERTh, bert-base-uncased) you can extract from a
specific transformer layer rather than the final layer.  Periti & Tahmasebi
(2022) found that layers 8–10 of MacBERTh outperform the last layer for
diachronic change detection; we default to layer 9.  Pass ``--layer -1`` to
fall back to the last hidden state (equivalent to the standard pooler input).

Usage
-----
    python src/03_embed.py
    python src/03_embed.py --model macberth --layer 9
    python src/03_embed.py --model bert --layer -1
    python src/03_embed.py --model bge --force
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "macberth": "emanjavacas/MacBERTh",
    "bert":     "bert-base-uncased",
    "bge":      "BAAI/bge-large-en-v1.5",
}

# Default layer to extract per model.
# -1  → use outputs.last_hidden_state (no hidden_states needed)
# N≥0 → use outputs.hidden_states[N]  (requires output_hidden_states=True)
# MacBERTh: layer 9 per Periti & Tahmasebi (2022) recommendation.
# bert-base-uncased: layer 9 also a reasonable default for English WiC tasks.
# BGE: sentence-level model, always uses last hidden state regardless.
MODEL_DEFAULT_LAYER: dict[str, int] = {
    "macberth": 9,
    "bert":     9,
    "bge":      -1,
}

DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available torch device (MPS → CUDA → CPU).

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon).")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("Using CUDA device.")
        return torch.device("cuda")
    logger.info("Using CPU device.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool a hidden state tensor, respecting the attention mask.

    Parameters
    ----------
    hidden_state:
        Shape ``(batch, seq_len, hidden_dim)``.
    attention_mask:
        Shape ``(batch, seq_len)``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, hidden_dim)``.
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    summed = (hidden_state * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    layer: int = -1,
) -> np.ndarray:
    """Encode a list of texts into float32 embeddings.

    Parameters
    ----------
    texts:
        Input strings to encode.
    tokenizer:
        HuggingFace tokenizer.
    model:
        HuggingFace model.
    device:
        Torch device to run inference on.
    batch_size:
        Number of texts per forward pass.
    layer:
        Transformer layer to extract (0-indexed including embedding layer).
        -1 → use ``outputs.last_hidden_state`` (final layer, default).
        N ≥ 0 → use ``outputs.hidden_states[N]``; requires the model to have
        been loaded with ``output_hidden_states=True``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(texts), embedding_dim)``.
    """
    use_hidden_states = layer >= 0
    all_embeddings: list[np.ndarray] = []
    model.eval()

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
        batch_texts = texts[start: start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=use_hidden_states)

        if use_hidden_states:
            if outputs.hidden_states is None:
                raise RuntimeError(
                    "Model did not return hidden_states.  Ensure the model "
                    "supports output_hidden_states and --layer is valid."
                )
            n_layers = len(outputs.hidden_states)
            if layer >= n_layers:
                raise ValueError(
                    f"Requested layer {layer} but model has only {n_layers} "
                    f"layers (0–{n_layers - 1})."
                )
            hidden = outputs.hidden_states[layer]
        else:
            hidden = outputs.last_hidden_state

        embeddings = mean_pool(hidden, encoded["attention_mask"])
        all_embeddings.append(embeddings.float().cpu().numpy())

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def embed_model(
    model_key: str,
    hf_model_id: str,
    texts: list[str],
    ids: list[str],
    embeddings_dir: Path,
    device: torch.device,
    layer: int = -1,
    force: bool = False,
) -> None:
    """Embed texts with one model and save outputs.

    Parameters
    ----------
    model_key:
        Short model name (e.g. "macberth").
    hf_model_id:
        HuggingFace model identifier.
    texts:
        List of context strings to encode.
    ids:
        Row IDs matching *texts* (for the index parquet).
    embeddings_dir:
        Directory to write ``.npy`` and ``index.parquet``.
    device:
        Torch device.
    layer:
        Transformer layer index to extract.  -1 → last hidden state.
    force:
        If True, overwrite existing ``.npy`` file.
    """
    # Encode layer into the filename so different layer choices don't collide.
    suffix = f"_L{layer}" if layer >= 0 else ""
    npy_path = embeddings_dir / f"{model_key}{suffix}.npy"
    index_path = embeddings_dir / "index.parquet"

    if npy_path.exists() and not force:
        logger.info(
            "Embeddings already exist at %s. Skipping (use --force to override).", npy_path
        )
        return

    layer_desc = f"layer {layer}" if layer >= 0 else "last hidden state"
    logger.info("Loading model %s from %s (extracting %s) …", model_key, hf_model_id, layer_desc)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModel.from_pretrained(hf_model_id).to(device)

    logger.info("Encoding %d texts …", len(texts))
    embeddings = encode_texts(texts, tokenizer, model, device, layer=layer)
    embeddings = embeddings.astype(np.float32)

    np.save(npy_path, embeddings)
    logger.info(
        "Saved embeddings: shape=%s, dtype=%s → %s", embeddings.shape, embeddings.dtype, npy_path
    )

    # Save / update index (same for all models — just write once per run)
    if not index_path.exists() or force:
        index_df = pd.DataFrame({"id": ids})
        index_df.to_parquet(index_path, index=False)
        logger.info("Saved index with %d rows → %s", len(index_df), index_path)

    print(
        f"[{model_key}{suffix}] embedding_dim={embeddings.shape[1]}, "
        f"n_rows={embeddings.shape[0]}, layer={layer_desc}"
    )

    # Free GPU/MPS memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def run_embedding(
    data_dir: Path,
    model_filter: str | None = None,
    layer_override: int | None = None,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Run the embedding stage.

    Parameters
    ----------
    data_dir:
        Project data root.
    model_filter:
        If provided, only embed with this model key.
    layer_override:
        If provided, use this layer for all models (overrides
        ``MODEL_DEFAULT_LAYER``).  Pass ``-1`` to always use the last layer.
    force:
        Overwrite existing ``.npy`` files.
    batch_size:
        Batch size for encoding.
    """
    observations_path = data_dir / "interim" / "observations.parquet"
    embeddings_dir = data_dir / "processed" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if not observations_path.exists():
        logger.error(
            "observations.parquet not found at %s. Run stage 02 first.", observations_path
        )
        sys.exit(1)

    obs_df = pd.read_parquet(observations_path)
    # Filter to confirmed idiomatic rows; exclude needs_review if the column exists
    idiomatic_df = obs_df[obs_df["is_idiomatic"] == True].reset_index(drop=True)  # noqa: E712
    if "needs_review" in idiomatic_df.columns:
        n_before = len(idiomatic_df)
        idiomatic_df = idiomatic_df[~idiomatic_df["needs_review"].fillna(False)].reset_index(
            drop=True
        )
        logger.info(
            "Excluded %d needs_review rows; %d remain.",
            n_before - len(idiomatic_df), len(idiomatic_df),
        )
    logger.info(
        "Loaded %d idiomatic observations (of %d total).", len(idiomatic_df), len(obs_df)
    )

    if idiomatic_df.empty:
        logger.warning("No idiomatic observations found. Exiting.")
        return

    texts = idiomatic_df["context_text"].tolist()
    ids = idiomatic_df["id"].tolist()

    device = get_device()

    models_to_run = {
        k: v for k, v in MODEL_REGISTRY.items()
        if model_filter is None or k == model_filter
    }

    if not models_to_run:
        logger.error(
            "No matching model for key %r. Available: %s", model_filter, list(MODEL_REGISTRY)
        )
        sys.exit(1)

    for key, hf_id in models_to_run.items():
        layer = layer_override if layer_override is not None else MODEL_DEFAULT_LAYER.get(key, -1)
        embed_model(
            model_key=key,
            hf_model_id=hf_id,
            texts=texts,
            ids=ids,
            embeddings_dir=embeddings_dir,
            device=device,
            layer=layer,
            force=force,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 03: Embed idiomatic observations with transformer models."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model to embed with. Default: all.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Transformer layer to extract (0-indexed; -1 = last hidden state). "
            "If omitted, uses the per-model default from MODEL_DEFAULT_LAYER "
            f"({MODEL_DEFAULT_LAYER}).  "
            "For BGE the flag is ignored and the last hidden state is always used."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    model_filter = None if args.model == "all" else args.model
    run_embedding(
        data_dir=args.data_dir,
        model_filter=model_filter,
        layer_override=args.layer,
        force=args.force,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
