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
import yaml
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "config" / "idioms.yaml"

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


# ---------------------------------------------------------------------------
# Span extraction (Strategy 1)
# ---------------------------------------------------------------------------

def _locate_span(
    ctx: str,
    span_str: str,
    sent_str: str,
) -> tuple[int | None, int | None]:
    """Return ``(char_start, char_end)`` of *span_str* within *ctx*.

    Searches within the containing sentence (*sent_str*) first for precision,
    then falls back to the first occurrence in the full context.  Returns
    ``(None, None)`` if the span cannot be located.
    """
    span_lower = span_str.lower()
    ctx_lower  = ctx.lower()
    sent_lower = sent_str.lower()

    sent_start = ctx_lower.find(sent_lower)
    if sent_start >= 0:
        rel_pos = sent_lower.find(span_lower)
        if rel_pos >= 0:
            char_start = sent_start + rel_pos
            return char_start, char_start + len(span_str)

    pos = ctx_lower.find(span_lower)
    if pos >= 0:
        return pos, pos + len(span_str)
    return None, None


def encode_spans(
    texts: list[str],
    span_strings: list[str],
    span_sentences: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    layer: int = -1,
) -> np.ndarray:
    """Encode idiom spans within context windows (Strategy 1).

    Tokenizes each context window with offset mapping, locates the idiom
    span tokens, and mean-pools only those token hidden states.  The
    surrounding context still informs the representation via attention; only
    the pooling target changes.  Falls back to full-context mean-pool when
    the span cannot be located.

    Parameters
    ----------
    texts:
        Full context window strings (``context_text``).
    span_strings:
        Raw matched idiom text (``raw_match``).
    span_sentences:
        The sentence containing the match (``sentence_text``), used to anchor
        the search within the context window.
    tokenizer, model, device:
        Already-loaded HuggingFace tokenizer / model / device.
    batch_size:
        Forward-pass batch size.
    layer:
        Transformer layer to extract.  -1 → ``last_hidden_state``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(texts), D)``.
    """
    use_hidden_states = layer >= 0
    all_span_embs: list[np.ndarray] = []
    n_fallback = 0

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding spans", unit="batch"):
        batch_texts = texts[start: start + batch_size]
        batch_spans = span_strings[start: start + batch_size]
        batch_sents = span_sentences[start: start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping").cpu()  # [B, seq_len, 2]
        encoded = {k: v.to(device) for k, v in encoded.items()}

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
                    f"Requested layer {layer} but model has only {n_layers} layers."
                )
            hidden = outputs.hidden_states[layer]  # [B, seq_len, D]
        else:
            hidden = outputs.last_hidden_state

        for i, (ctx, span_str, sent_str) in enumerate(
            zip(batch_texts, batch_spans, batch_sents)
        ):
            char_start, char_end = _locate_span(ctx, span_str, sent_str)

            if char_start is None:
                vec = mean_pool(
                    hidden[i: i + 1], encoded["attention_mask"][i: i + 1]
                ).squeeze(0)
                n_fallback += 1
            else:
                offsets = offset_mapping[i]  # [seq_len, 2]  CPU tensor
                tok_mask = (offsets[:, 0] < char_end) & (offsets[:, 1] > char_start)
                if tok_mask.sum() == 0:
                    vec = mean_pool(
                        hidden[i: i + 1], encoded["attention_mask"][i: i + 1]
                    ).squeeze(0)
                    n_fallback += 1
                else:
                    span_hidden = hidden[i][tok_mask.to(device)]  # [n_span, D]
                    vec = span_hidden.mean(dim=0)

            all_span_embs.append(vec.float().cpu().numpy())

    if n_fallback:
        logger.warning(
            "  span extraction: %d / %d items fell back to full-context mean-pool.",
            n_fallback, len(texts),
        )
    return np.vstack(all_span_embs).astype(np.float32)


# ---------------------------------------------------------------------------
# Paraphrase scoring
# ---------------------------------------------------------------------------

def _load_idiom_poles(config_path: Path) -> dict[str, tuple[str, str]]:
    """Load per-idiom pole phrases from idioms.yaml.

    Returns
    -------
    dict mapping idiom phrase → (trivial_pole, significant_pole).
    Only treatment idioms with ``include: true`` are included.
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    poles: dict[str, tuple[str, str]] = {}
    for entry in cfg.get("idioms", []):
        if entry.get("group") == "treatment" and entry.get("include", False):
            poles[entry["phrase"]] = (entry["trivial_pole"], entry["significant_pole"])
    if not poles:
        raise ValueError(
            "No treatment idioms with trivial_pole/significant_pole found in "
            "config/idioms.yaml."
        )
    return poles


def _compute_paraphrase_scores(
    embeddings: np.ndarray,
    obs_idioms: list[str],
    idiom_poles: dict[str, tuple[str, str]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    model_key: str,
) -> np.ndarray:
    """Compute per-observation score_paraphrase using idiom-specific poles.

    For each unique idiom present in *obs_idioms*, encodes its
    ``trivial_pole`` and ``significant_pole`` (single phrases) using the
    already-loaded *model*, then computes::

        score = sim(obs_emb, trivial_pole_emb) − sim(obs_emb, signif_pole_emb)

    All unique pole phrases are encoded in a single batch (deduplicating
    across idioms) so the model is only called once regardless of how many
    idioms share a pole phrase.

    Parameters
    ----------
    embeddings:
        Float32 array of shape ``(N, D)`` — one row per observation, aligned
        with *obs_idioms*.
    obs_idioms:
        Length-N list of idiom phrase strings, one per observation row.
    idiom_poles:
        Mapping from idiom phrase → ``(trivial_pole_text, significant_pole_text)``.
    tokenizer, model, device:
        Already-loaded HuggingFace tokenizer/model (same instance as the
        context embeddings — no second model load).
    batch_size:
        Forward-pass batch size.
    model_key:
        Short model name for log messages.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(N,)``.  Rows whose idiom has no poles
        defined are set to ``nan``.
    """
    # Collect and deduplicate all pole texts needed for this batch of idioms.
    present_idioms = set(obs_idioms) & set(idiom_poles)
    if not present_idioms:
        logger.warning("[%s] No observation idioms match the poles config.", model_key)
        return np.full(len(obs_idioms), np.nan)

    unique_pole_texts: list[str] = sorted(
        {t for idiom in present_idioms for t in idiom_poles[idiom]}
    )
    logger.info(
        "  [%s] Encoding %d unique pole phrases for %d idioms …",
        model_key, len(unique_pole_texts), len(present_idioms),
    )
    pole_embs = encode_texts(
        unique_pole_texts, tokenizer, model, device, batch_size=batch_size, layer=-1
    )
    pole_lookup: dict[str, np.ndarray] = dict(zip(unique_pole_texts, pole_embs))

    # L2-normalise observation embeddings once.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = np.where(norms > 1e-9, embeddings / norms, 0.0).astype(np.float32)

    obs_idioms_arr = np.asarray(obs_idioms)
    scores = np.full(len(obs_idioms), np.nan, dtype=np.float64)

    for idiom in present_idioms:
        trivial_text, signif_text = idiom_poles[idiom]

        tc = pole_lookup[trivial_text].astype(np.float32)
        sc = pole_lookup[signif_text].astype(np.float32)

        tc_norm = float(np.linalg.norm(tc))
        sc_norm = float(np.linalg.norm(sc))
        tc = tc / tc_norm if tc_norm > 1e-9 else tc
        sc = sc / sc_norm if sc_norm > 1e-9 else sc

        pole_sep = float(1.0 - float(np.dot(tc, sc)))
        if pole_sep < 0.15:
            logger.warning(
                "[%s] '%s': pole separation=%.4f < 0.15 — "
                "score_paraphrase may be unreliable for this idiom.",
                model_key, idiom, pole_sep,
            )

        mask = obs_idioms_arr == idiom
        sim_t = emb_n[mask] @ tc
        sim_s = emb_n[mask] @ sc
        scores[mask] = (sim_t - sim_s).astype(np.float64)

    valid = scores[~np.isnan(scores)]
    logger.info(
        "  [%s] score_paraphrase: mean=%.4f  std=%.4f  n_valid=%d / %d",
        model_key,
        float(valid.mean()) if len(valid) else float("nan"),
        float(valid.std()) if len(valid) else float("nan"),
        len(valid), len(scores),
    )
    return scores


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
    obs_idioms: list[str] | None = None,
    idiom_poles: dict[str, tuple[str, str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    span_strings: list[str] | None = None,
    span_sentences: list[str] | None = None,
) -> np.ndarray | None:
    """Embed texts with one model, save outputs, and compute score_paraphrase.

    Produces two ``.npy`` files when span extraction is enabled:

    * ``{model_key}{suffix}.npy``       — full-context mean-pool (robustness)
    * ``{model_key}{suffix}_span.npy``  — idiom-span mean-pool (main track,
                                          Strategy 1; context-informed via
                                          attention but pooled over span tokens
                                          only)

    Parameters
    ----------
    model_key:
        Short model name (e.g. "macberth").
    hf_model_id:
        HuggingFace model identifier.
    texts:
        List of context strings to encode (``context_text``).
    ids:
        Row IDs matching *texts* (for the index parquet).
    embeddings_dir:
        Directory to write ``.npy`` and ``index.parquet``.
    device:
        Torch device.
    layer:
        Transformer layer index to extract.  -1 → last hidden state.
    force:
        If True, overwrite existing ``.npy`` files.
    obs_idioms:
        Length-N list of idiom phrase strings, one per observation.  Required
        together with *idiom_poles* to compute ``score_paraphrase``.
    idiom_poles:
        Mapping from idiom phrase → ``(trivial_pole_text, significant_pole_text)``
        as loaded by ``_load_idiom_poles()``.
    batch_size:
        Forward-pass batch size.
    span_strings:
        Raw matched idiom text (``raw_match``), one per observation.  When
        provided together with *span_sentences*, span embeddings are computed
        and saved alongside the full-context embeddings.
    span_sentences:
        Sentence containing the idiom match (``sentence_text``), used to
        locate the span precisely within the context window.

    Returns
    -------
    np.ndarray or None
        Float64 array of ``score_paraphrase`` values (from full-context
        embeddings) aligned with *ids*, or None if poles were not supplied.
    """
    suffix = f"_L{layer}" if layer >= 0 else ""
    npy_path      = embeddings_dir / f"{model_key}{suffix}.npy"
    npy_span_path = embeddings_dir / f"{model_key}{suffix}_span.npy"
    index_path    = embeddings_dir / "index.parquet"

    layer_desc = f"layer {layer}" if layer >= 0 else "last hidden state"

    do_span = span_strings is not None and span_sentences is not None

    # ------------------------------------------------------------------
    # Fast path: both outputs already exist
    # ------------------------------------------------------------------
    if npy_path.exists() and (not do_span or npy_span_path.exists()) and not force:
        logger.info(
            "Embeddings already exist at %s%s. Loading for paraphrase scoring "
            "(use --force to re-embed).",
            npy_path,
            f" and {npy_span_path}" if do_span else "",
        )
        if obs_idioms is not None and idiom_poles is not None:
            embeddings = np.load(npy_path)
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModel.from_pretrained(hf_model_id).to(device)
            paraphrase_scores = _compute_paraphrase_scores(
                embeddings, obs_idioms, idiom_poles,
                tokenizer, model, device, batch_size, model_key,
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            return paraphrase_scores
        return None

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model %s from %s (extracting %s) …", model_key, hf_model_id, layer_desc)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModel.from_pretrained(hf_model_id).to(device)

    # ------------------------------------------------------------------
    # Full-context embeddings (robustness track)
    # ------------------------------------------------------------------
    if not npy_path.exists() or force:
        logger.info("Encoding %d texts (full-context) …", len(texts))
        embeddings = encode_texts(
            texts, tokenizer, model, device, batch_size=batch_size, layer=layer
        ).astype(np.float32)
        np.save(npy_path, embeddings)
        logger.info(
            "Saved full-context embeddings: shape=%s → %s", embeddings.shape, npy_path
        )
    else:
        logger.info("Full-context embeddings exist, loading → %s", npy_path)
        embeddings = np.load(npy_path)

    # Save / update index (same for all models — write once per run)
    if not index_path.exists() or force:
        index_df = pd.DataFrame({"id": ids})
        index_df.to_parquet(index_path, index=False)
        logger.info("Saved index with %d rows → %s", len(index_df), index_path)

    print(
        f"[{model_key}{suffix}] embedding_dim={embeddings.shape[1]}, "
        f"n_rows={embeddings.shape[0]}, layer={layer_desc}"
    )

    # ------------------------------------------------------------------
    # Span embeddings (main track — Strategy 1)
    # ------------------------------------------------------------------
    if do_span and (not npy_span_path.exists() or force):
        logger.info("Encoding %d spans (Strategy 1: idiom-span mean-pool) …", len(texts))
        span_embs = encode_spans(
            texts, span_strings, span_sentences,  # type: ignore[arg-type]
            tokenizer, model, device, batch_size=batch_size, layer=layer,
        )
        np.save(npy_span_path, span_embs)
        logger.info(
            "Saved span embeddings: shape=%s → %s", span_embs.shape, npy_span_path
        )
        print(
            f"[{model_key}{suffix}_span] embedding_dim={span_embs.shape[1]}, "
            f"n_rows={span_embs.shape[0]}, layer={layer_desc}"
        )

    # ------------------------------------------------------------------
    # Paraphrase scoring (while model is still loaded)
    # ------------------------------------------------------------------
    paraphrase_scores: np.ndarray | None = None
    if obs_idioms is not None and idiom_poles is not None:
        paraphrase_scores = _compute_paraphrase_scores(
            embeddings, obs_idioms, idiom_poles,
            tokenizer, model, device, batch_size, model_key,
        )

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return paraphrase_scores


def run_embedding(
    data_dir: Path,
    model_filter: str | None = None,
    layer_override: int | None = None,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    span: bool = True,
) -> None:
    """Run the embedding stage.

    For each model, produces two sets of embeddings:

    * **Full-context** (``{model}.npy``) — mean-pool over the entire ±2
      sentence context window.  Used as the robustness/validation track.
    * **Span** (``{model}_span.npy``) — mean-pool over only the idiom token
      positions within the context window (Strategy 1).  The surrounding
      context still informs the representation via attention; this is the main
      analysis track.  Disable with ``span=False`` / ``--no-span``.

    Also computes ``score_paraphrase`` and writes
    ``data/processed/scores.parquet``.

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
    span:
        If True (default), also produce ``_span.npy`` files using idiom-span
        mean-pooling.
    """
    observations_path = data_dir / "interim" / "observations.parquet"
    embeddings_dir    = data_dir / "processed" / "embeddings"
    scores_path       = data_dir / "processed" / "scores.parquet"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if not observations_path.exists():
        logger.error(
            "observations.parquet not found at %s. Run stage 02 first.", observations_path
        )
        sys.exit(1)

    obs_df = pd.read_parquet(observations_path)
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

    texts      = idiomatic_df["context_text"].tolist()
    ids        = idiomatic_df["id"].tolist()
    obs_idioms = idiomatic_df["idiom"].tolist()

    # Span extraction inputs
    span_strings:   list[str] | None = None
    span_sentences: list[str] | None = None
    if span:
        if "raw_match" not in idiomatic_df.columns or "sentence_text" not in idiomatic_df.columns:
            logger.warning(
                "raw_match / sentence_text columns missing from observations — "
                "skipping span extraction.  Re-run stage 01 to populate them."
            )
        else:
            span_strings   = idiomatic_df["raw_match"].tolist()
            span_sentences = idiomatic_df["sentence_text"].tolist()

    # Load per-idiom poles once — used for every model.
    idiom_poles = _load_idiom_poles(CONFIG_PATH)
    logger.info("Loaded poles for %d treatment idioms from config.", len(idiom_poles))

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

    score_rows: list[dict] = []

    for key, hf_id in models_to_run.items():
        layer = layer_override if layer_override is not None else MODEL_DEFAULT_LAYER.get(key, -1)
        paraphrase_scores = embed_model(
            model_key=key,
            hf_model_id=hf_id,
            texts=texts,
            ids=ids,
            embeddings_dir=embeddings_dir,
            device=device,
            layer=layer,
            force=force,
            obs_idioms=obs_idioms,
            idiom_poles=idiom_poles,
            batch_size=batch_size,
            span_strings=span_strings,
            span_sentences=span_sentences,
        )
        if paraphrase_scores is not None:
            for row_id, score in zip(ids, paraphrase_scores):
                if not np.isnan(score):
                    score_rows.append({"id": row_id, "model": key, "score_paraphrase": float(score)})

    # --- Save scores.parquet ---
    if score_rows:
        scores_df = pd.DataFrame(score_rows)
        scores_df.to_parquet(scores_path, index=False)
        logger.info(
            "Saved %d paraphrase score rows (%d models) → %s",
            len(score_rows), len(scores_df["model"].unique()), scores_path,
        )

        # --- Pearson correlation validation ---
        # Join mean score_paraphrase per (model, idiom, decade) with APD/PRT
        # from drift.parquet if it already exists.
        drift_path = data_dir / "processed" / "drift.parquet"
        if drift_path.exists():
            _log_paraphrase_drift_correlation(scores_df, idiomatic_df, drift_path)


def _log_paraphrase_drift_correlation(
    scores_df: pd.DataFrame,
    idiomatic_df: pd.DataFrame,
    drift_path: Path,
) -> None:
    """Log Pearson r between mean(score_paraphrase) per bin and APD/PRT.

    Joins observation-level paraphrase scores with decade-pair drift, using
    the bin that each observation's year falls in (aligned to drift.parquet's
    ``decade_start`` values).
    """
    try:
        drift_df = pd.read_parquet(drift_path)
        # Infer bin_width from drift: most common decade_end - decade_start gap
        gap = (drift_df["decade_end"] - drift_df["decade_start"])
        bin_width = int(gap.mode().iloc[0]) if not gap.empty else 20

        meta = idiomatic_df[["id", "idiom", "year"]].copy()
        meta["bin"] = (meta["year"] // bin_width) * bin_width

        merged = scores_df.merge(meta, on="id", how="inner")
        bin_means = (
            merged.groupby(["model", "idiom", "bin"])["score_paraphrase"]
            .mean()
            .reset_index()
            .rename(columns={"bin": "decade_start"})
        )

        treat_drift = drift_df[drift_df["group"] == "treatment"]
        joined = bin_means.merge(
            treat_drift[["model", "idiom", "decade_start", "apd", "drift_cosine"]],
            on=["model", "idiom", "decade_start"],
            how="inner",
        ).dropna(subset=["score_paraphrase", "apd", "drift_cosine"])

        if len(joined) < 4:
            logger.info(
                "Too few overlapping (score_paraphrase, drift) pairs for Pearson correlation "
                "(%d rows). Run stage 04 before stage 03 to see this metric.", len(joined),
            )
            return

        for model_key in joined["model"].unique():
            sub = joined[joined["model"] == model_key]
            r_apd, p_apd = pearsonr(sub["score_paraphrase"], sub["apd"])
            r_prt, p_prt = pearsonr(sub["score_paraphrase"], sub["drift_cosine"])
            logger.info(
                "  [%s] Pearson(score_paraphrase, APD)=%.3f (p=%.3f)  "
                "Pearson(score_paraphrase, PRT)=%.3f (p=%.3f)  n=%d",
                model_key, r_apd, p_apd, r_prt, p_prt, len(sub),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute Pearson correlation: %s", exc)


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
        "--no-span",
        action="store_true",
        help=(
            "Skip idiom-span extraction (Strategy 1).  By default, span "
            "embeddings are computed alongside full-context embeddings."
        ),
    )
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
        span=not args.no_span,
    )


if __name__ == "__main__":
    main()
