"""Stage 02 — LLM disambiguation via AWS Bedrock.

Classifies each candidate as idiomatic (figurative) vs literal (actual money)
using a Llama model hosted on AWS Bedrock.

Key design decisions
--------------------
- triviality_hint removed: the field was conflating rhetorical register with
  denomination semantics and produced near-uniform "significant" labels.
  Scoring of triviality now lives entirely in the embedding/axis stage.
- Binary gate only: is_idiomatic + confidence + reasoning.
- Prompt context: recomputed at ±PROMPT_CONTEXT_WIN sentences from speech_text
  (wider than stage-01 context_text, which is used for embeddings).
- needs_review flag: rows where confidence < threshold are flagged rather than
  silently accepted; downstream stages can filter or inspect these separately.
- Model: us.meta.llama3-1-70b-instruct-v1:0 (3.1 70B cross-region profile).

Usage
-----
    python src/02_disambiguate.py
    python src/02_disambiguate.py --force
    python src/02_disambiguate.py --prompt-window 3 --confidence-threshold high
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.bedrock_client import BedrockClient  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "us.meta.llama3-1-70b-instruct-v1:0"
DEFAULT_WORKERS = 5
DEFAULT_PROMPT_WINDOW = 3        # ±N sentences fed to the LLM prompt
DEFAULT_CONFIDENCE_THRESHOLD = "medium"  # "low" confidence → needs_review

# Confidence hierarchy: a row is flagged for review when its confidence
# is strictly below the threshold level.
_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0, "failed": -1}

SYSTEM_PROMPT = (
    "You are a linguistic classifier for historical parliamentary text. "
    "Respond only with a JSON object — no preamble, no explanation outside the JSON."
)

USER_TEMPLATE = """\
Determine whether the highlighted phrase in the excerpt below is used \
idiomatically (figuratively) or literally (referring to actual money or a \
physical quantity).

Phrase: "{idiom}"

Excerpt:
{context_text}

Rules:
- is_idiomatic = true  → the phrase is a figure of speech; the denomination \
word is NOT referring to a real monetary amount in this sentence.
- is_idiomatic = false → the phrase refers to an actual sum of money or a \
literal transaction.
- confidence: "high" = clear-cut; "medium" = some ambiguity; \
"low" = genuinely unclear or insufficient context.
- reasoning: one concise sentence explaining your decision.

Respond with ONLY this JSON, nothing else:
{{
  "is_idiomatic": true or false,
  "confidence": "high", "medium", or "low",
  "reasoning": "one sentence"
}}"""


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
# Context recomputation
# ---------------------------------------------------------------------------

def _build_prompt_context(
    row: dict,
    window: int,
) -> str:
    """Return a wider context string for the LLM prompt.

    Uses ``speech_text`` + ``sentence_text`` when available; falls back to the
    pre-computed ``context_text`` from stage 01.

    Parameters
    ----------
    row:
        Candidate row dict.
    window:
        Sentence radius (≥0). -1 returns the full speech.
    """
    speech_text: str = str(row.get("speech_text") or "").strip()
    sentence_text: str = str(row.get("sentence_text") or "").strip()

    if not speech_text or window < 0:
        return speech_text or str(row.get("context_text", ""))

    sentences = nltk.sent_tokenize(speech_text)
    if not sentences:
        return sentence_text

    # Locate matching sentence by prefix overlap
    prefix = sentence_text[:50]
    idx = next(
        (i for i, s in enumerate(sentences) if prefix in s or s[:50] in sentence_text),
        None,
    )
    if idx is None:
        return str(row.get("context_text", sentence_text))

    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(idiom: str, context_text: str) -> str:
    """Render the classification prompt for one candidate row."""
    return USER_TEMPLATE.format(idiom=idiom, context_text=context_text)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """Extract the first JSON object found in *text*."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response.")
    return text[start:end]


def parse_llm_response(raw: str) -> dict:
    """Parse LLM output into a structured dict with is_idiomatic and confidence.

    Parameters
    ----------
    raw:
        Raw string output from the model.

    Returns
    -------
    dict
        Keys: is_idiomatic (bool|None), confidence (str), reasoning (str).
        On parse failure: is_idiomatic=None, confidence="failed".
    """
    try:
        block = _extract_json_block(raw)
        data = json.loads(block)
        is_idiomatic = data.get("is_idiomatic")
        if isinstance(is_idiomatic, str):
            is_idiomatic = is_idiomatic.strip().lower() == "true"
        confidence = str(data.get("confidence", "low")).lower().strip()
        if confidence not in _CONFIDENCE_RANK:
            confidence = "low"
        return {
            "is_idiomatic": bool(is_idiomatic) if is_idiomatic is not None else None,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")),
        }
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("JSON parse failure: %s | raw=%r", exc, raw[:200])
        return {
            "is_idiomatic": None,
            "confidence": "failed",
            "reasoning": f"parse_error: {exc}",
        }


def _needs_review(confidence: str, threshold: str) -> bool:
    """Return True if *confidence* is strictly below *threshold*."""
    return _CONFIDENCE_RANK.get(confidence, -1) < _CONFIDENCE_RANK.get(threshold, 1)


# ---------------------------------------------------------------------------
# Row classifier (runs in thread pool)
# ---------------------------------------------------------------------------

def _classify_row(
    row_dict: dict,
    client: BedrockClient,
    model_id: str,
    system_prompt: str,
    prompt_window: int,
    confidence_threshold: str,
) -> tuple[dict, dict]:
    """Classify one candidate row via Bedrock. Runs in a thread pool.

    All file I/O is deferred to the calling (main) thread via the returned
    tuple — no locks required.

    Parameters
    ----------
    row_dict:
        Candidate row as a plain dict.
    client:
        Shared BedrockClient (boto3 clients are thread-safe).
    model_id:
        Bedrock inference-profile ID.
    system_prompt:
        System prompt string.
    prompt_window:
        Sentence radius for prompt context recomputation.
    confidence_threshold:
        Minimum confidence to auto-accept; below this → needs_review=True.

    Returns
    -------
    tuple
        ``(result_dict, log_entry_dict)``.
    """
    context_text = _build_prompt_context(row_dict, window=prompt_window)
    prompt = build_prompt(row_dict["idiom"], context_text)
    timestamp = datetime.now(timezone.utc).isoformat()
    raw_response: str = ""

    try:
        raw_response = client.invoke_with_retry(
            prompt=prompt,
            system=system_prompt,
            model_id=model_id,
            max_tokens=128,
        )
    except Exception as exc:
        logger.error("Bedrock error for id=%s: %s", row_dict["id"], exc)
        raw_response = f"ERROR: {exc}"

    log_entry = {
        "id": row_dict["id"],
        "timestamp": timestamp,
        "model_id": model_id,
        "prompt_context_text": context_text,
        "raw_response": raw_response,
    }

    parsed = parse_llm_response(raw_response)
    result = dict(row_dict)
    result.update(parsed)
    result["needs_review"] = _needs_review(parsed["confidence"], confidence_threshold)
    result["model_id"] = model_id
    result["timestamp"] = timestamp
    # Carry forward the wider context used for the LLM prompt (for audit)
    result["prompt_context_text"] = context_text

    return result, log_entry


# ---------------------------------------------------------------------------
# Flush helper
# ---------------------------------------------------------------------------

def _flush_observations(rows: list[dict], output_path: Path) -> None:
    """Write *rows* to *output_path* as parquet."""
    if not rows:
        return
    pd.DataFrame(rows).to_parquet(output_path, index=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_disambiguation(
    data_dir: Path,
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
    prompt_window: int = DEFAULT_PROMPT_WINDOW,
    confidence_threshold: str = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    """Run disambiguation on all candidates in candidates.parquet.

    Parameters
    ----------
    data_dir:
        Project data root.
    force:
        Re-classify all rows, ignoring previous results.
    workers:
        Number of concurrent Bedrock threads.
    prompt_window:
        Sentence radius around the matching sentence for the LLM prompt.
        Recomputed from ``speech_text``; independent of stage-01 context_text.
    confidence_threshold:
        Rows with confidence strictly below this level are flagged
        ``needs_review=True`` and excluded from downstream analysis by default.
        Choices: ``"high"`` (flag medium+low) or ``"medium"`` (flag only low).
    """
    _ensure_nltk()

    interim_dir = data_dir / "interim"
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = interim_dir / "candidates.parquet"
    output_path = interim_dir / "observations.parquet"
    raw_log_path = logs_dir / "bedrock_raw.jsonl"

    if not candidates_path.exists():
        logger.error(
            "candidates.parquet not found at %s. Run stage 01 first.", candidates_path
        )
        sys.exit(1)

    candidates_df = pd.read_parquet(candidates_path)
    logger.info("Loaded %d candidates.", len(candidates_df))

    # Resume: skip already-classified IDs
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if output_path.exists() and not force:
        obs_df = pd.read_parquet(output_path)
        done_ids = set(obs_df["id"].tolist())
        existing_rows = obs_df.to_dict("records")
        logger.info("Resuming: %d rows already classified.", len(done_ids))

    todo_df = (
        candidates_df[~candidates_df["id"].isin(done_ids)]
        if not force
        else candidates_df
    )
    logger.info(
        "%d rows to classify (model=%s, prompt_window=±%d, confidence_threshold=%s).",
        len(todo_df), MODEL_ID, prompt_window, confidence_threshold,
    )

    if todo_df.empty:
        logger.info("Nothing to do. All candidates already classified.")
        return

    client = BedrockClient(model_id=MODEL_ID)
    new_rows: list[dict] = []

    raw_log_fh = raw_log_path.open("a", encoding="utf-8")
    try:
        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _, row in todo_df.iterrows():
                future = executor.submit(
                    _classify_row,
                    row.to_dict(),
                    client,
                    MODEL_ID,
                    SYSTEM_PROMPT,
                    prompt_window,
                    confidence_threshold,
                )
                futures_map[future] = row["id"]

            for future in tqdm(
                as_completed(futures_map),
                total=len(futures_map),
                desc="Classifying",
                unit="row",
            ):
                try:
                    result, log_entry = future.result()
                except Exception as exc:
                    logger.error(
                        "Unexpected error for id=%s: %s", futures_map[future], exc
                    )
                    continue

                # All file I/O on the main thread — no lock needed
                raw_log_fh.write(json.dumps(log_entry) + "\n")
                raw_log_fh.flush()
                new_rows.append(result)

                if len(new_rows) % 100 == 0:
                    _flush_observations(existing_rows + new_rows, output_path)
                    logger.info(
                        "Progress: %d classified, %d flagged for review.",
                        len(new_rows),
                        sum(r.get("needs_review", False) for r in new_rows),
                    )
    finally:
        raw_log_fh.close()

    all_rows = existing_rows + new_rows
    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(output_path, index=False)

    n_idiomatic = int((out_df["is_idiomatic"] == True).sum())   # noqa: E712
    n_literal = int((out_df["is_idiomatic"] == False).sum())    # noqa: E712
    n_review = int(out_df.get("needs_review", pd.Series(dtype=bool)).sum())
    n_failed = int((out_df["confidence"] == "failed").sum())
    logger.info(
        "Done. Total: %d | idiomatic: %d | literal: %d | needs_review: %d | failed: %d",
        len(out_df), n_idiomatic, n_literal, n_review, n_failed,
    )
    logger.info("Saved observations → %s", output_path)

    # Print a quick breakdown table
    if "group" in out_df.columns:
        breakdown = (
            out_df.groupby(["idiom", "group"])
            .agg(
                n=("id", "count"),
                pct_idiomatic=("is_idiomatic", lambda x: round(100 * x.sum() / len(x), 1)),
                pct_review=("needs_review", lambda x: round(100 * x.sum() / len(x), 1)),
            )
            .sort_values(["group", "n"], ascending=[True, False])
        )
        logger.info("\n%s", breakdown.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 02: Disambiguate idiom candidates via AWS Bedrock (binary gate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "confidence-threshold controls which rows are auto-accepted:\n"
            "  medium  accept high+medium; flag low confidence → needs_review (default)\n"
            "  high    accept only high; flag medium+low → needs_review\n"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-classify all rows, ignoring previous results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent Bedrock threads (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--prompt-window",
        type=int,
        default=DEFAULT_PROMPT_WINDOW,
        metavar="N",
        help=(
            "Sentence radius used to build the LLM prompt context "
            "(recomputed from speech_text; default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        choices=["high", "medium"],
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence to auto-accept; below this → needs_review=True.",
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
    run_disambiguation(
        data_dir=args.data_dir,
        force=args.force,
        workers=args.workers,
        prompt_window=args.prompt_window,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
