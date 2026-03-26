"""Context-window comparison experiment.

Tests which sentence context radius produces the strongest first-differenced
panel FE coefficient between the triviality score (S_t) and log real value
of the monetary denomination (log RV).

Windows tested: 0 (target sentence only), 1 (±1), 2 (±2), -1 (full speech).

Reads
-----
- data/interim/observations.parquet  (must contain speech_text column)
- data/raw/price_index.csv

Writes
------
- outputs/tables/context_window_comparison.csv
- outputs/figures/context_window_comparison.png

Usage
-----
    python src/scripts/compare_context_windows.py
    python src/scripts/compare_context_windows.py --model bge --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOWS: list[int] = [0, 1, 2, -1]
WINDOW_LABELS: dict[int, str] = {0: "±0 (sentence)", 1: "±1", 2: "±2", -1: "full speech"}

MODEL_REGISTRY: dict[str, str] = {
    "bge": "BAAI/bge-large-en-v1.5",
    "macberth": "emanjavacas/MacBERTh",
}

# Denomination nominal values in old pence
NOMINAL_VALUES: dict[str, float] = {
    "penny": 1.0,
    "farthing": 0.25,
    "shilling": 12.0,
    "pound": 240.0,
    "guinea": 252.0,
}

# Axis-projection anchor pairs (positive = trivial, negative = significant)
ANCHOR_PAIRS: list[tuple[str, str]] = [
    ("trivial", "significant"),
    ("worthless", "valuable"),
    ("petty", "important"),
    ("trifling", "weighty"),
]

BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# NLTK
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


# ---------------------------------------------------------------------------
# Context window recomputation
# ---------------------------------------------------------------------------

def recompute_context(
    obs_df: pd.DataFrame,
    window: int,
) -> list[str]:
    """Recompute context_text for each row at a given sentence window radius.

    Parameters
    ----------
    obs_df:
        Observations dataframe. Must contain ``speech_text`` and
        ``sentence_text`` columns.
    window:
        Sentence radius. ``-1`` returns the full speech text.

    Returns
    -------
    list of str
        Recomputed context strings, one per row, in the same order as *obs_df*.
    """
    _ensure_nltk()
    contexts: list[str] = []

    for _, row in tqdm(obs_df.iterrows(), total=len(obs_df),
                       desc=f"Window {WINDOW_LABELS.get(window, window)}", unit="row",
                       leave=False):
        speech_text: str = str(row.get("speech_text", row.get("context_text", "")))
        sentence_text: str = str(row.get("sentence_text", ""))

        if window < 0:
            contexts.append(speech_text)
            continue

        sentences = nltk.sent_tokenize(speech_text)
        if not sentences:
            contexts.append(sentence_text)
            continue

        # Find the matching sentence by substring containment
        idx = next(
            (i for i, s in enumerate(sentences)
             if sentence_text[:40] in s or s[:40] in sentence_text),
            None,
        )
        if idx is None:
            # Fallback: use sentence_text ± no context
            contexts.append(sentence_text)
            continue

        if window == 0:
            contexts.append(sentences[idx])
        else:
            start = max(0, idx - window)
            end = min(len(sentences), idx + window + 1)
            contexts.append(" ".join(sentences[start:end]))

    return contexts


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed a list of texts using mean-pooled last hidden state.

    Parameters
    ----------
    texts:
        List of strings to embed.
    model_name:
        HuggingFace model identifier.
    batch_size:
        Inference batch size.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(texts), embedding_dim)``.
    """
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding", unit="batch",
                  leave=False):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**encoded)
        # Mean pool over token dimension
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(emb.cpu().float().numpy())

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Scoring — axis projection
# ---------------------------------------------------------------------------

def _encode_word(word: str, model_name: str) -> np.ndarray:
    """Embed a single word and return its mean-pooled vector."""
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    enc = tokenizer([word], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    return emb.cpu().float().numpy()[0]


def axis_projection_scores(
    embeddings: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Score each embedding by projection onto trivial↔significant axes.

    Parameters
    ----------
    embeddings:
        Shape ``(N, D)``.
    model_name:
        HuggingFace model ID used to embed anchor words.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` — higher means more trivial.
    """
    axes: list[np.ndarray] = []
    for pos_word, neg_word in ANCHOR_PAIRS:
        pos_vec = _encode_word(pos_word, model_name)
        neg_vec = _encode_word(neg_word, model_name)
        axis = pos_vec - neg_vec
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm
        axes.append(axis)

    # Average projection across all anchor pairs
    scores = np.zeros(len(embeddings))
    for axis in axes:
        scores += embeddings @ axis
    return scores / len(axes)


# ---------------------------------------------------------------------------
# Price index helpers
# ---------------------------------------------------------------------------

def load_price_index(price_index_path: Path) -> pd.DataFrame:
    """Load and normalise the price index to 1800 = 100.

    Parameters
    ----------
    price_index_path:
        Path to ``price_index.csv`` (columns: year, price_index).
    """
    df = pd.read_csv(price_index_path)
    df.columns = [c.lower().strip() for c in df.columns]
    base = df.loc[df["year"] == 1800, "price_index"]
    if not base.empty:
        df["price_index"] = df["price_index"] / base.iloc[0] * 100.0
    return df[["year", "price_index"]].dropna()


def compute_log_rv(denomination: str, price_index: float) -> float | None:
    """Compute log real value for a denomination at a given price level."""
    nominal = NOMINAL_VALUES.get(str(denomination).lower())
    if nominal is None or price_index <= 0:
        return None
    rv = nominal / price_index
    return float(np.log(rv)) if rv > 0 else None


# ---------------------------------------------------------------------------
# Regression — first-differenced panel FE
# ---------------------------------------------------------------------------

def first_diff_panel_fe(
    panel_df: pd.DataFrame,
    score_col: str = "score",
    entity_col: str = "idiom",
    time_col: str = "year",
    rv_col: str = "log_RV",
    nw_lags: int = 4,
) -> dict:
    """First-difference panel regression with Newey-West SEs.

    Equation: ΔS_{i,t} = β · Δlog(RV_{u,t}) + ε

    Parameters
    ----------
    panel_df:
        Long-format dataframe with entity, time, score, and log_RV columns.
    score_col, entity_col, time_col, rv_col:
        Column names.
    nw_lags:
        Newey-West lag order.

    Returns
    -------
    dict
        Keys: beta, se_nw, t_stat, p_value, n, r2.
    """
    df = (
        panel_df
        .dropna(subset=[score_col, rv_col])
        .sort_values([entity_col, time_col])
        .copy()
    )
    # Aggregate to annual mean per entity
    annual = (
        df.groupby([entity_col, time_col])[[score_col, rv_col]]
        .mean()
        .reset_index()
        .sort_values([entity_col, time_col])
    )
    # First difference within each entity
    annual["d_score"] = annual.groupby(entity_col)[score_col].diff()
    annual["d_rv"] = annual.groupby(entity_col)[rv_col].diff()
    annual = annual.dropna(subset=["d_score", "d_rv"])

    if len(annual) < 5:
        return {"beta": np.nan, "se_nw": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "n": len(annual), "r2": np.nan}

    X = sm.add_constant(annual["d_rv"].values)
    y = annual["d_score"].values
    try:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        beta = float(res.params[1])
        se = float(res.bse[1])
        t = float(res.tvalues[1])
        p = float(res.pvalues[1])
        r2 = float(res.rsquared)
    except Exception:
        return {"beta": np.nan, "se_nw": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "n": len(annual), "r2": np.nan}

    return {"beta": beta, "se_nw": se, "t_stat": t, "p_value": p,
            "n": len(annual), "r2": r2}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_comparison(
    data_dir: Path,
    model_key: str = "bge",
    output_dir_tables: Path | None = None,
    output_dir_figures: Path | None = None,
) -> pd.DataFrame:
    """Run the full context-window comparison experiment.

    Parameters
    ----------
    data_dir:
        Project data root.
    model_key:
        Which embedding model to use (``"bge"`` or ``"macberth"``).
    output_dir_tables, output_dir_figures:
        Override output directories; defaults to project outputs/.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per window size.
    """
    obs_path = data_dir / "interim" / "observations.parquet"
    price_path = data_dir / "raw" / "price_index.csv"

    if not obs_path.exists():
        logger.error("observations.parquet not found: %s. Run stages 01–02 first.", obs_path)
        sys.exit(1)
    if not price_path.exists():
        logger.error("price_index.csv not found: %s", price_path)
        sys.exit(1)

    if "speech_text" not in pd.read_parquet(obs_path, columns=["speech_text"] if "speech_text" in pd.read_parquet(obs_path).columns else []).columns:
        logger.warning(
            "speech_text column not found in observations.parquet. "
            "Re-run stage 01 with the updated script to populate it. "
            "Falling back to context_text for all windows."
        )

    obs_df = pd.read_parquet(obs_path)
    # Filter to confirmed idiomatic treatment rows only
    if "is_idiomatic" in obs_df.columns:
        obs_df = obs_df[obs_df["is_idiomatic"] == True]  # noqa: E712
    if "group" in obs_df.columns:
        obs_df = obs_df[obs_df["group"] == "treatment"]

    logger.info("Working with %d idiomatic treatment observations.", len(obs_df))
    if obs_df.empty:
        logger.error("No treatment observations found after filtering.")
        sys.exit(1)

    price_df = load_price_index(price_path)

    # Precompute log_RV per observation
    def _log_rv_row(row: pd.Series) -> float | None:
        denom = row.get("denomination")
        if isinstance(denom, (list, np.ndarray)):
            denom = denom[0] if len(denom) > 0 else None
        if denom is None or str(denom) in ("placebo", "none"):
            return None
        price_row = price_df[price_df["year"] == row["year"]]
        if price_row.empty:
            return None
        return compute_log_rv(str(denom), float(price_row["price_index"].iloc[0]))

    obs_df = obs_df.copy()
    obs_df["log_RV"] = obs_df.apply(_log_rv_row, axis=1)

    model_hf_id = MODEL_REGISTRY[model_key]
    logger.info("Embedding model: %s (%s)", model_key, model_hf_id)

    # Load model ONCE, reuse for all windows
    logger.info("Loading embedding model …")
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id)
    hf_model = AutoModel.from_pretrained(model_hf_id).to(device).eval()

    def _embed_batch(texts: list[str]) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = hf_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_embs.append(emb.cpu().float().numpy())
        return np.vstack(all_embs)

    # Precompute anchor axis embeddings (once)
    logger.info("Embedding anchor word pairs …")
    def _embed_word(w: str) -> np.ndarray:
        enc = tokenizer([w], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = hf_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return emb.cpu().float().numpy()[0]

    axes: list[np.ndarray] = []
    for pos_word, neg_word in ANCHOR_PAIRS:
        pos_vec = _embed_word(pos_word)
        neg_vec = _embed_word(neg_word)
        axis = pos_vec - neg_vec
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis /= norm
        axes.append(axis)

    results = []
    for window in WINDOWS:
        label = WINDOW_LABELS[window]
        logger.info("Window %s — recomputing context …", label)

        has_speech_text = "speech_text" in obs_df.columns
        if has_speech_text:
            contexts = recompute_context(obs_df, window)
        else:
            # Fallback when speech_text not stored
            contexts = list(obs_df["context_text"].astype(str))

        logger.info("Window %s — embedding %d texts …", label, len(contexts))
        embeddings = _embed_batch(contexts)

        # Axis projection score
        scores = np.zeros(len(embeddings))
        for axis in axes:
            scores += embeddings @ axis
        scores /= len(axes)

        panel_df = obs_df[["idiom", "year", "log_RV"]].copy()
        panel_df["score"] = scores

        reg = first_diff_panel_fe(panel_df)
        reg["window"] = label
        reg["window_int"] = window
        results.append(reg)
        logger.info(
            "  β=%.4f  SE=%.4f  t=%.2f  p=%.4f  N=%d",
            reg["beta"], reg["se_nw"], reg["t_stat"], reg["p_value"], reg["n"],
        )

    result_df = pd.DataFrame(results)[
        ["window", "beta", "se_nw", "t_stat", "p_value", "n", "r2"]
    ]

    # Save table
    tables_dir = output_dir_tables or (PROJECT_ROOT / "outputs" / "tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "context_window_comparison.csv"
    result_df.to_csv(out_csv, index=False)
    logger.info("Saved comparison table → %s", out_csv)

    # Plot
    figures_dir = output_dir_figures or (PROJECT_ROOT / "outputs" / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 4))
    x = range(len(WINDOWS))
    x_labels = [WINDOW_LABELS[w] for w in WINDOWS]

    valid = result_df.dropna(subset=["beta"])
    axes_plot[0].bar(range(len(valid)), valid["beta"], color="steelblue", alpha=0.8)
    axes_plot[0].errorbar(range(len(valid)), valid["beta"],
                          yerr=1.96 * valid["se_nw"], fmt="none", color="black", capsize=4)
    axes_plot[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes_plot[0].set_xticks(range(len(valid)))
    axes_plot[0].set_xticklabels(valid["window"], rotation=15, ha="right")
    axes_plot[0].set_title(f"β (first-diff panel FE) by context window\nModel: {model_key}")
    axes_plot[0].set_ylabel("β")

    axes_plot[1].bar(range(len(valid)), valid["t_stat"].abs(), color="coral", alpha=0.8)
    axes_plot[1].axhline(1.96, color="gray", linewidth=0.8, linestyle="--", label="t=1.96")
    axes_plot[1].set_xticks(range(len(valid)))
    axes_plot[1].set_xticklabels(valid["window"], rotation=15, ha="right")
    axes_plot[1].set_title("|t-statistic| by context window")
    axes_plot[1].set_ylabel("|t|")
    axes_plot[1].legend()

    plt.tight_layout()
    out_fig = figures_dir / "context_window_comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure → %s", out_fig)

    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare regression strength across sentence context window sizes."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="bge",
        help="Embedding model to use (default: bge).",
    )
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
    result_df = run_comparison(data_dir=args.data_dir, model_key=args.model)
    print("\n=== Context Window Comparison ===")
    print(result_df.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
