"""Stage 02 — Define per-idiom triviality axes in BERT embedding space.

For each treatment idiom, embeds the trivial_pole and significant_pole strings
at all 12 BERT layers and computes the normalised difference vector as the
triviality axis.  Also computes a pooled axis by averaging across idioms.

Outputs
-------
data/triviality/axes.npy     — (n_idioms, 12, 768)
data/triviality/axes_index.csv
data/triviality/axis_pooled.npy — (12, 768)

Usage
-----
    python src/02_triviality_axes.py
    python src/02_triviality_axes.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hooks import embed_sentences

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)


def _load_treatment_idioms(idioms_yaml: Path) -> list[dict]:
    """Return treatment idioms that have both trivial_pole and significant_pole."""
    with open(idioms_yaml) as fh:
        cfg = yaml.safe_load(fh)

    idioms = []
    for entry in cfg.get("idioms", []):
        if not entry.get("include", True):
            continue
        if entry.get("group") != "treatment":
            continue
        if not entry.get("trivial_pole") or not entry.get("significant_pole"):
            continue
        idioms.append(entry)

    return idioms


def run(cfg: dict, dry_run: bool) -> None:
    import torch

    paths      = cfg["paths"]
    model_name = cfg["model"]["name"]
    batch_size = cfg["model"]["batch_size"]
    n_layers   = cfg["model"]["n_layers"]

    idioms_path       = PROJECT_ROOT / paths["idioms_config"]
    axes_path         = PROJECT_ROOT / paths["triviality_axes"]
    axes_index_path   = PROJECT_ROOT / paths["triviality_axes_index"]
    pooled_path       = PROJECT_ROOT / paths["triviality_axis_pooled"]

    # --- Load idioms ---
    idioms = _load_treatment_idioms(idioms_path)
    if dry_run:
        idioms = idioms[:5]
        logger.info("[dry-run] Using first %d idioms", len(idioms))
    else:
        logger.info("Loaded %d treatment idioms with poles", len(idioms))

    if not idioms:
        logger.error("No treatment idioms with poles found in %s", idioms_path)
        sys.exit(1)

    # Collect all pole strings (trivial, then significant) for a single embed pass
    trivial_poles    = [e["trivial_pole"]    for e in idioms]
    significant_poles = [e["significant_pole"] for e in idioms]
    all_poles = trivial_poles + significant_poles   # 2 * n_idioms strings

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s", model_name, device)
    from transformers import BertModel, BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model     = BertModel.from_pretrained(model_name).to(device).eval()

    # --- Embed all poles at once ---
    n_idioms = len(idioms)
    logger.info("Embedding %d pole strings …", len(all_poles))
    pole_emb = embed_sentences(
        model, tokenizer, all_poles, device, batch_size=batch_size
    )  # (2 * n_idioms, n_layers, 768)

    trivial_emb    = pole_emb[:n_idioms]   # (n_idioms, n_layers, 768)
    significant_emb = pole_emb[n_idioms:]  # (n_idioms, n_layers, 768)

    # --- Compute triviality axes ---
    diff = trivial_emb - significant_emb   # (n_idioms, n_layers, 768)

    # Normalise each (idiom, layer) vector to unit length
    norms = np.linalg.norm(diff, axis=-1, keepdims=True)   # (n_idioms, n_layers, 1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    axes = (diff / norms).astype(np.float32)               # (n_idioms, n_layers, 768)

    # --- Pooled axis: mean over idioms, then renormalise ---
    pooled = axes.mean(axis=0)                             # (n_layers, 768)
    pool_norms = np.linalg.norm(pooled, axis=-1, keepdims=True)
    pool_norms = np.where(pool_norms < 1e-10, 1.0, pool_norms)
    pooled = (pooled / pool_norms).astype(np.float32)

    # --- Save ---
    axes_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(axes_path, axes)
    logger.info("Saved triviality axes → %s  shape=%s", axes_path, axes.shape)

    np.save(pooled_path, pooled)
    logger.info("Saved pooled axis → %s  shape=%s", pooled_path, pooled.shape)

    index_df = pd.DataFrame([
        {
            "idiom_index":    i,
            "idiom_name":     e["phrase"],
            "trivial_pole":   e["trivial_pole"],
            "significant_pole": e["significant_pole"],
        }
        for i, e in enumerate(idioms)
    ])
    index_df.to_csv(axes_index_path, index=False)
    logger.info("Saved index → %s", axes_index_path)

    print(f"\n=== Triviality axes ===")
    print(f"  Shape: {axes.shape}  (n_idioms × n_layers × hidden_dim)")
    print(f"  Pooled shape: {pooled.shape}")
    for i, row in index_df.iterrows():
        print(f"  [{i:2d}] {row['idiom_name']}")


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    parser = argparse.ArgumentParser(
        description="Stage 02: Define triviality axes from idiom poles."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use only the first 5 idioms.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(cfg, args.dry_run)


if __name__ == "__main__":
    main()
