"""Substitution test — denomination token MLM prediction shift.

For each idiomatic observation, the denomination word (e.g. "penny",
"shilling") is replaced with the MacBERTh ``[MASK]`` token.  The model's
top-k MLM predictions at the mask position are collected as a discrete
distribution over the vocabulary.  These per-observation distributions are
then aggregated by (idiom, decade) to form decade-level substitute
distributions.  The shift between consecutive decades is measured using the
Jensen-Shannon Divergence (JSD).

Rationale
---------
If a monetary denomination idiom is becoming bleached / trivialised over time,
speakers may begin treating the denomination word as interchangeable with other
(non-monetary or more generic) terms.  This should show up as a shift in which
words the language model predicts as the most natural fill for the masked slot.
The JSD captures how different the predicted substitute vocabularies are across
consecutive decades without committing to a specific direction.

This measure complements the embedding-distribution approach (APD/PRT in
stage 04):
  * Stage 04 measures change in the *sentence-level* embedding.
  * The substitution test measures change in the *local* predicted substitute
    vocabulary at the idiom-token position.

A stable idiom should predict the same denomination word decade after decade.
A bleaching idiom may predict more generic or figurative substitutes.

Output
------
    outputs/tables/substitution_jsd.csv
        Columns: idiom, denomination, group, decade_start, decade_end, jsd,
                 n_start, n_end

    outputs/tables/substitution_top_k.csv
        Columns: idiom, decade, rank, token, prob_mean
        Top-k mean probability over all observations in that (idiom, decade).

    outputs/figures/substitution_jsd_<idiom>.png
        Per-idiom JSD time series.

Usage
-----
    python src/scripts/substitution_test.py
    python src/scripts/substitution_test.py --top-k 20 --min-obs 3
    python src/scripts/substitution_test.py --force
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_MODEL_ID = "emanjavacas/MacBERTh"   # Only MacBERTh; has the right vocab for Hansard
TOP_K_DEFAULT = 20
MIN_OBS_DEFAULT = 3
MAX_GAP = 20  # must match stage 04

# Denomination words to mask.  Plural forms included.
DENOMINATION_TOKENS: set[str] = {
    "penny", "pennies", "pence",
    "farthing", "farthings",
    "shilling", "shillings",
    "pound", "pounds",
    "guinea", "guineas",
    "halfpenny", "halfpennies",
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _mask_denomination(text: str, idiom: str) -> tuple[str, bool]:
    """Replace the first denomination token in *text* with '[MASK]'.

    Strategy
    --------
    1. Try to find any token from DENOMINATION_TOKENS in *text* (word boundary
       match, case-insensitive) and replace the first occurrence.
    2. Fall back to masking any word from the idiom phrase itself.

    Returns
    -------
    (masked_text, success)
        *success* is False when no denomination token was found.
    """
    # Attempt 1: explicit denomination vocabulary
    for token in sorted(DENOMINATION_TOKENS, key=len, reverse=True):
        pat = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        new_text, n_subs = pat.subn("[MASK]", text, count=1)
        if n_subs:
            return new_text, True

    # Attempt 2: any word from the idiom phrase
    for word in idiom.split():
        pat = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        new_text, n_subs = pat.subn("[MASK]", text, count=1)
        if n_subs:
            return new_text, True

    return text, False


def _find_mask_positions(input_ids: torch.Tensor, mask_id: int) -> list[int]:
    """Return token indices of all [MASK] tokens in a 1-D tensor."""
    return (input_ids == mask_id).nonzero(as_tuple=True)[0].tolist()


# ---------------------------------------------------------------------------
# Core: collect top-k substitutes per observation
# ---------------------------------------------------------------------------

def collect_substitutes(
    rows: list[dict],
    tokenizer: AutoTokenizer,
    model: BertForMaskedLM,
    device: torch.device,
    top_k: int = TOP_K_DEFAULT,
    batch_size: int = 16,
) -> list[dict]:
    """Run MLM on masked texts and return top-k substitutes for each row.

    Parameters
    ----------
    rows:
        List of dicts with keys: id, idiom, decade, masked_text.
    tokenizer:
        MacBERTh tokenizer.
    model:
        BertForMaskedLM instance (already on device, in eval mode).
    device:
        Torch device.
    top_k:
        Number of top predicted tokens to retain per mask position.
    batch_size:
        Inference batch size.

    Returns
    -------
    list[dict]
        One dict per (row, rank) pair with keys:
        id, idiom, decade, rank, token, prob.
    """
    mask_token_id = tokenizer.mask_token_id
    results: list[dict] = []

    for start in tqdm(range(0, len(rows), batch_size), desc="MLM inference", unit="batch"):
        batch = rows[start: start + batch_size]
        texts = [r["masked_text"] for r in batch]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits  # (B, seq_len, vocab_size)

        probs = torch.softmax(logits, dim=-1)  # (B, seq_len, vocab_size)

        for b_idx, row in enumerate(batch):
            input_ids_b = encoded["input_ids"][b_idx]
            mask_positions = _find_mask_positions(input_ids_b, mask_token_id)

            if not mask_positions:
                logger.debug("Row %s: no [MASK] token found in encoding.", row["id"])
                continue

            # Use the first mask position
            pos = mask_positions[0]
            token_probs = probs[b_idx, pos]  # (vocab_size,)

            top_probs, top_ids = token_probs.topk(top_k)
            top_probs = top_probs.cpu().float().numpy()
            top_ids = top_ids.cpu().numpy()

            tokens = tokenizer.convert_ids_to_tokens(top_ids)
            for rank, (token, prob) in enumerate(zip(tokens, top_probs)):
                results.append({
                    "id":     row["id"],
                    "idiom":  row["idiom"],
                    "decade": row["decade"],
                    "rank":   rank,
                    "token":  token,
                    "prob":   float(prob),
                })

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _decade_distributions(
    sub_df: pd.DataFrame,
    top_k: int,
) -> dict[tuple[str, int], pd.Series]:
    """Build a probability vector over vocab for each (idiom, decade).

    Returns
    -------
    dict mapping (idiom, decade) → pd.Series indexed by token, values = mean prob.
    """
    dist: dict[tuple[str, int], pd.Series] = {}

    for (idiom, decade), grp in sub_df.groupby(["idiom", "decade"]):
        # Average probability per token across all observations in this decade
        token_probs = (
            grp.groupby("token")["prob"]
            .mean()
            .sort_values(ascending=False)
        )
        dist[(idiom, int(decade))] = token_probs

    return dist


def _jsd_between(
    p_series: pd.Series,
    q_series: pd.Series,
) -> float:
    """Jensen-Shannon divergence between two token probability series.

    Aligns on the union of tokens; unshared tokens contribute 0 probability
    for the missing distribution.
    """
    all_tokens = p_series.index.union(q_series.index)
    p = p_series.reindex(all_tokens, fill_value=0.0).values.astype(float)
    q = q_series.reindex(all_tokens, fill_value=0.0).values.astype(float)

    # Normalise (they may not sum to 1 if top-k is smaller than vocab)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum < 1e-12 or q_sum < 1e-12:
        return np.nan
    p /= p_sum
    q /= q_sum

    jsd = float(jensenshannon(p, q) ** 2)   # scipy returns sqrt(JSD); square for JSD
    return jsd


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_substitution_test(
    data_dir: Path,
    top_k: int = TOP_K_DEFAULT,
    min_obs: int = MIN_OBS_DEFAULT,
    force: bool = False,
    batch_size: int = 16,
) -> None:
    """Run the substitution test and write output tables and figures.

    Parameters
    ----------
    data_dir:
        Project data root.
    top_k:
        Number of top MLM predictions to retain per observation.
    min_obs:
        Minimum observations per decade to include in JSD computation.
    force:
        Overwrite existing outputs.
    batch_size:
        MLM inference batch size.
    """
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    jsd_output  = tables_dir / "substitution_jsd.csv"
    topk_output = tables_dir / "substitution_top_k.csv"

    if jsd_output.exists() and not force:
        logger.info("substitution_jsd.csv already exists. Use --force to recompute.")
        return

    observations_path = data_dir / "interim" / "observations.parquet"
    if not observations_path.exists():
        logger.error("observations.parquet not found: %s. Run stage 02 first.", observations_path)
        sys.exit(1)

    obs_df = pd.read_parquet(observations_path)

    # Filter to idiomatic, non-review observations
    mask = obs_df["is_idiomatic"] == True  # noqa: E712
    if "needs_review" in obs_df.columns:
        mask &= ~obs_df["needs_review"].fillna(False)
    obs_df = obs_df[mask].reset_index(drop=True)
    logger.info("Using %d idiomatic observations.", len(obs_df))

    # Use prompt_context_text if available (wider window from stage 02),
    # else fall back to context_text from stage 01.
    text_col = "prompt_context_text" if "prompt_context_text" in obs_df.columns else "context_text"
    logger.info("Using text column: %s", text_col)

    obs_df["decade"] = (obs_df["year"] // 10) * 10

    # ---- Build masked texts ------------------------------------------------
    rows_to_process: list[dict] = []
    n_failed = 0
    for _, row in obs_df.iterrows():
        text = str(row.get(text_col, "") or row.get("context_text", ""))
        masked, ok = _mask_denomination(text, str(row["idiom"]))
        if not ok:
            n_failed += 1
            logger.debug("Row %s: could not find denomination token to mask.", row["id"])
            continue
        rows_to_process.append({
            "id":          row["id"],
            "idiom":       row["idiom"],
            "denomination": row.get("denomination", []),
            "group":       row.get("group", ""),
            "decade":      int(row["decade"]),
            "masked_text": masked,
        })

    logger.info(
        "Masked %d / %d observations (%d failed — no denomination token found).",
        len(rows_to_process), len(obs_df), n_failed,
    )
    if not rows_to_process:
        logger.error("No rows with maskable denomination tokens. Exiting.")
        sys.exit(1)

    # ---- Load MacBERTh MLM model -------------------------------------------
    device = get_device()
    logger.info("Loading %s for MLM …", HF_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = BertForMaskedLM.from_pretrained(HF_MODEL_ID).to(device)
    model.eval()

    # ---- Collect substitutes -----------------------------------------------
    sub_records = collect_substitutes(
        rows_to_process, tokenizer, model, device,
        top_k=top_k, batch_size=batch_size,
    )

    # Free GPU/MPS memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    if not sub_records:
        logger.error("No substitute records collected. Exiting.")
        sys.exit(1)

    sub_df = pd.DataFrame(sub_records)
    logger.info("Collected %d substitute records.", len(sub_df))

    # ---- Save top-k aggregate per (idiom, decade) --------------------------
    topk_records: list[dict] = []
    for (idiom, decade), grp in sub_df.groupby(["idiom", "decade"]):
        token_mean = (
            grp.groupby("token")["prob"]
            .mean()
            .sort_values(ascending=False)
            .head(top_k)
        )
        for rank, (token, prob) in enumerate(token_mean.items()):
            topk_records.append({
                "idiom": idiom, "decade": decade,
                "rank": rank, "token": token, "prob_mean": prob,
            })
    topk_df = pd.DataFrame(topk_records)
    topk_df.to_csv(topk_output, index=False, float_format="%.6f")
    logger.info("Saved top-k aggregate → %s", topk_output)

    # ---- Build decade distributions and compute JSD ------------------------
    decade_dists = _decade_distributions(sub_df, top_k=top_k)

    # Build idiom metadata lookup
    idiom_meta: dict[str, dict] = {}
    for r in rows_to_process:
        idiom_meta[r["idiom"]] = {
            "denomination": r["denomination"],
            "group": r["group"],
        }

    jsd_rows: list[dict] = []
    for idiom in sub_df["idiom"].unique():
        valid_decades = sorted(
            d for (i, d) in decade_dists if i == idiom
            and sub_df[(sub_df["idiom"] == idiom) & (sub_df["decade"] == d)]["id"].nunique() >= min_obs
        )
        meta = idiom_meta.get(idiom, {})
        for j in range(len(valid_decades) - 1):
            d0, d1 = valid_decades[j], valid_decades[j + 1]
            if (d1 - d0) > MAX_GAP:
                logger.debug(
                    "Substitution: idiom '%s' skipping (%d, %d) — gap %d > %d.",
                    idiom, d0, d1, d1 - d0, MAX_GAP,
                )
                continue

            p = decade_dists.get((idiom, d0))
            q = decade_dists.get((idiom, d1))
            if p is None or q is None:
                continue

            jsd = _jsd_between(p, q)

            # Count unique observations per decade
            n_start = sub_df[
                (sub_df["idiom"] == idiom) & (sub_df["decade"] == d0)
            ]["id"].nunique()
            n_end = sub_df[
                (sub_df["idiom"] == idiom) & (sub_df["decade"] == d1)
            ]["id"].nunique()

            jsd_rows.append({
                "idiom":        idiom,
                "denomination": meta.get("denomination", []),
                "group":        meta.get("group", ""),
                "decade_start": d0,
                "decade_end":   d1,
                "jsd":          jsd,
                "n_start":      n_start,
                "n_end":        n_end,
            })

    if not jsd_rows:
        logger.warning("No JSD rows produced (too few observations per decade?).")
    else:
        jsd_df = pd.DataFrame(jsd_rows)
        jsd_df.to_csv(jsd_output, index=False, float_format="%.6f")
        logger.info("Saved JSD results: %d rows → %s", len(jsd_df), jsd_output)

        # Summary
        logger.info(
            "\n%s",
            jsd_df.groupby("group")
            .agg(n=("jsd", "count"), mean_jsd=("jsd", "mean"), std_jsd=("jsd", "std"))
            .round(5)
            .to_string()
        )

        # Plots
        _plot_jsd_timeseries(jsd_df, figures_dir)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_jsd_timeseries(jsd_df: pd.DataFrame, figures_dir: Path) -> None:
    """Per-idiom JSD time-series line plots."""
    treatment = jsd_df[jsd_df["group"] == "treatment"]
    if treatment.empty:
        return

    idioms = sorted(treatment["idiom"].unique())
    n = len(idioms)
    cols = 3
    rows_grid = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(14, 3 * rows_grid), sharex=False)
    axes = np.array(axes).flatten()

    for ax, idiom in zip(axes, idioms):
        sub = treatment[treatment["idiom"] == idiom].sort_values("decade_start")
        ax.plot(
            sub["decade_start"], sub["jsd"],
            marker="o", linewidth=1.2, markersize=4, color="#1a9850",
        )
        ax.set_title(idiom, fontsize=8, wrap=True)
        ax.set_xlabel("Decade", fontsize=7)
        ax.set_ylabel("JSD", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Substitution test: decade-pair JSD (MacBERTh top-k substitutes)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = figures_dir / "substitution_jsd_timeseries.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved JSD time-series plot → %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Substitution test: track MLM denomination-token substitute shift over time."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_DEFAULT,
        help=f"Top-k MLM predictions to retain per observation (default: {TOP_K_DEFAULT}).",
    )
    parser.add_argument(
        "--min-obs", type=int, default=MIN_OBS_DEFAULT,
        help=f"Min observations per decade to compute JSD (default: {MIN_OBS_DEFAULT}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="MLM inference batch size (default: 16).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--log-level", default="INFO",
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
    run_substitution_test(
        data_dir=args.data_dir,
        top_k=args.top_k,
        min_obs=args.min_obs,
        force=args.force,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
