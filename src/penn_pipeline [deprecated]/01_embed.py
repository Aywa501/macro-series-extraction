"""Stage 01 — Embed Penn Historical English sentences with BERT.

Reads penn_sentences.parquet (Stage 00 output) and runs every sentence
through bert-base-uncased, saving the CLS token hidden state at each of the
12 transformer layers.

No regression is performed here.  All temporal regression, direction
extraction, and residualisation lives in Stage 01b so that expensive
re-embedding is never repeated.

Outputs
-------
../data/embeddings/penn_embeddings_by_layer.npy  — (n_sentences, 12, 768)

Checkpointing
-------------
Sentences are embedded in chunks of 500.  Each chunk is saved as a separate
.npy file in a staging directory.  If the script is interrupted and restarted,
already-completed chunks are skipped automatically.  On completion the chunks
are concatenated into the final array and the staging directory is removed.

Usage
-----
    python src/penn_pipeline/01_embed.py
    python src/penn_pipeline/01_embed.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils"))

from hooks import embed_sentences
from config_utils import get_model_cfg, resolve_paths

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)


def run(cfg: dict, model_key: str, dry_run: bool) -> None:
    import torch
    from transformers import BertModel, BertTokenizerFast

    mcfg       = get_model_cfg(cfg, model_key)
    paths      = resolve_paths(cfg, model_key)
    model_name = mcfg["name"]
    batch_size = mcfg["batch_size"]

    sentences_path = PROJECT_ROOT / paths["penn_sentences"]
    emb_path       = PROJECT_ROOT / paths["penn_embeddings"]

    # --- Load sentences ---
    logger.info("Loading %s", sentences_path)
    df = pd.read_parquet(sentences_path)
    if dry_run:
        df = df.head(50)
        logger.info("[dry-run] Using %d sentences", len(df))
    else:
        logger.info("Loaded %d sentences (years %d–%d)",
                    len(df), df["year"].min(), df["year"].max())

    sentences = df["sentence"].tolist()
    n = len(sentences)

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s …", model_name, device)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model     = BertModel.from_pretrained(model_name).to(device).eval()

    # --- Embed with checkpointing ---
    CHUNK_SIZE  = 500
    staging_dir = emb_path.parent / "_embed_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info("Embedding %d sentences in %d chunks of %d …", n, n_chunks, CHUNK_SIZE)
    t0 = time.time()

    for chunk_idx in range(n_chunks):
        chunk_path = staging_dir / f"chunk_{chunk_idx:05d}.npy"
        if chunk_path.exists():
            logger.info("  Chunk %d / %d already done — skipping", chunk_idx + 1, n_chunks)
            continue

        start = chunk_idx * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, n)

        chunk_emb = embed_sentences(
            model, tokenizer, sentences[start:end], device, batch_size=batch_size
        )  # (chunk_size, 12, 768)

        np.save(chunk_path, chunk_emb)

        done    = end
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (n - done) / rate if rate > 0 else float("inf")
        logger.info(
            "  Chunk %d / %d  (%d / %d sentences)  %.1fs elapsed  ETA %.0fs",
            chunk_idx + 1, n_chunks, done, n, elapsed, eta,
        )

    # --- Concatenate chunks ---
    logger.info("Concatenating %d chunks …", n_chunks)
    embeddings = np.concatenate(
        [np.load(staging_dir / f"chunk_{i:05d}.npy") for i in range(n_chunks)],
        axis=0,
    )  # (n, 12, 768)
    logger.info("Embedding array shape: %s  (%.1f MB)", embeddings.shape, embeddings.nbytes / 1e6)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    logger.info("Saved → %s", emb_path)

    # Clean up staging files
    for i in range(n_chunks):
        (staging_dir / f"chunk_{i:05d}.npy").unlink(missing_ok=True)
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    print(f"\nEmbeddings saved: {emb_path}")
    print(f"Shape: {embeddings.shape}  ({embeddings.nbytes / 1e9:.2f} GB)")
    print("Run Stage 01b to fit temporal regression on these embeddings.")


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    parser = argparse.ArgumentParser(
        description="Stage 01: Embed Penn sentences with the chosen model (no regression)."
    )
    parser.add_argument("--model", default="bert",
                        choices=list(cfg.get("models", {"bert": None}).keys()),
                        help="Which model to embed with (default: bert).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use only the first 50 sentences.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(cfg, args.model, args.dry_run)


if __name__ == "__main__":
    main()
