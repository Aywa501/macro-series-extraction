"""Stage 02 — LLM disambiguation via AWS Bedrock (Amazon Nova Micro).

Classifies each candidate as idiomatic (figurative) vs literal (actual money)
using Amazon Nova Micro on AWS Bedrock.

Two execution modes
-------------------
batch (default)
    Submits all candidates as a single ``CreateModelInvocationJob``.  Requires
    ``--s3-uri`` (e.g. ``s3://my-bucket/bedrock-batch/``) and ``--iam-role``
    (an IAM role ARN that Bedrock can assume to read/write that bucket).
    ~40× cheaper than real-time at $0.0175/1M input tokens.  Typical
    turnaround for ~6 000 rows: 30–90 minutes.

realtime
    Sends requests concurrently via the Converse API.  Use ``--realtime``
    for local testing or small incremental batches.  Throttle-aware with
    exponential backoff.

Key design decisions
--------------------
- Binary gate only: is_idiomatic + confidence + reasoning.
- Prompt context: recomputed at ±PROMPT_CONTEXT_WIN sentences from
  speech_text (wider than stage-01 context_text used for embeddings).
- needs_review flag: rows with confidence below the threshold are flagged
  rather than silently accepted.
- Model: amazon.nova-micro-v1:0

Usage
-----
    # Batch mode (recommended)
    python src/02_disambiguate.py \\
        --s3-uri s3://my-bucket/bedrock-batch/ \\
        --iam-role arn:aws:iam::123456789012:role/BedrockBatchRole

    # Real-time mode (testing / small runs)
    python src/02_disambiguate.py --realtime

    # Resume an interrupted run (skips already-classified IDs automatically)
    python src/02_disambiguate.py --s3-uri ... --iam-role ...

    # Force re-classification of all rows
    python src/02_disambiguate.py --force --s3-uri ... --iam-role ...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.bedrock_client import BedrockBatchClient, BedrockClient  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "amazon.nova-micro-v1:0"

DEFAULT_PROMPT_WINDOW      = 3        # ±N sentences fed to the LLM prompt
DEFAULT_CONFIDENCE_THRESHOLD = "medium"
DEFAULT_REALTIME_WORKERS   = 5        # concurrent threads for real-time mode
DEFAULT_MAX_TOKENS         = 128

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
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt_context(row: dict, window: int) -> str:
    """Return a wider context string for the LLM prompt.

    Uses ``speech_text`` + ``sentence_text`` when available; falls back to
    ``context_text`` from stage 01.
    """
    speech_text   = str(row.get("speech_text")   or "").strip()
    sentence_text = str(row.get("sentence_text") or "").strip()

    if not speech_text or window < 0:
        return speech_text or str(row.get("context_text", ""))

    sentences = nltk.sent_tokenize(speech_text)
    if not sentences:
        return sentence_text

    prefix = sentence_text[:50]
    idx = next(
        (i for i, s in enumerate(sentences) if prefix in s or s[:50] in sentence_text),
        None,
    )
    if idx is None:
        return str(row.get("context_text", sentence_text))

    start = max(0, idx - window)
    end   = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


def build_user_prompt(idiom: str, context_text: str) -> str:
    """Render the classification prompt for one candidate."""
    return USER_TEMPLATE.format(idiom=idiom, context_text=context_text)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object in response.")
    return text[start:end]


def parse_llm_response(raw: str) -> dict:
    """Parse Nova Micro output into is_idiomatic / confidence / reasoning."""
    try:
        block = _extract_json_block(raw)
        data  = json.loads(block)
        is_idiomatic = data.get("is_idiomatic")
        if isinstance(is_idiomatic, str):
            is_idiomatic = is_idiomatic.strip().lower() == "true"
        confidence = str(data.get("confidence", "low")).lower().strip()
        if confidence not in _CONFIDENCE_RANK:
            confidence = "low"
        return {
            "is_idiomatic": bool(is_idiomatic) if is_idiomatic is not None else None,
            "confidence":   confidence,
            "reasoning":    str(data.get("reasoning", "")),
        }
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("JSON parse failure: %s | raw=%r", exc, raw[:200])
        return {"is_idiomatic": None, "confidence": "failed", "reasoning": f"parse_error: {exc}"}


def _needs_review(confidence: str, threshold: str) -> bool:
    return _CONFIDENCE_RANK.get(confidence, -1) < _CONFIDENCE_RANK.get(threshold, 1)


def _apply_parsed(row_dict: dict, raw_text: str, model_id: str, timestamp: str, prompt_context: str, threshold: str) -> dict:
    """Merge parsed LLM result into a copy of *row_dict*."""
    parsed = parse_llm_response(raw_text)
    result = dict(row_dict)
    result.update(parsed)
    result["needs_review"]        = _needs_review(parsed["confidence"], threshold)
    result["model_id"]            = model_id
    result["timestamp"]           = timestamp
    result["prompt_context_text"] = prompt_context
    return result


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def _run_batch(
    todo_df: pd.DataFrame,
    existing_rows: list[dict],
    output_path: Path,
    raw_log_path: Path,
    s3_uri: str,
    iam_role: str,
    prompt_window: int,
    confidence_threshold: str,
) -> None:
    """Submit all todo rows as a single batch job and write results."""
    client = BedrockBatchClient()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job_name  = f"idiom-disambig-{timestamp}"

    # Normalise S3 URI to ensure trailing slash on the base prefix
    base_uri  = s3_uri.rstrip("/") + "/"
    s3_input  = f"{base_uri}{job_name}/input.jsonl"
    s3_output = f"{base_uri}{job_name}/output/"

    # Build records
    logger.info("Building %d batch records …", len(todo_df))
    records: list[dict] = []
    prompt_map: dict[str, str] = {}   # record_id → prompt context (for audit log)

    for _, row in todo_df.iterrows():
        ctx    = _build_prompt_context(row.to_dict(), window=prompt_window)
        prompt = build_user_prompt(row["idiom"], ctx)
        record = client.build_record(
            record_id=row["id"],
            system=SYSTEM_PROMPT,
            user_message=prompt,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        records.append(record)
        prompt_map[row["id"]] = ctx

    # Upload → submit → poll
    client.upload_input(records, s3_input)
    job_arn = client.submit_job(
        job_name=job_name,
        model_id=MODEL_ID,
        s3_input_uri=s3_input,
        s3_output_uri=s3_output,
        role_arn=iam_role,
    )
    logger.info("Batch job ARN: %s", job_arn)
    logger.info("Polling for completion (this may take 30–90 min for ~6 000 rows) …")

    client.wait_for_completion(job_arn)

    # Download and parse results
    batch_results = client.download_results(s3_output)
    result_map: dict[str, str] = {}   # record_id → raw response text
    for item in batch_results:
        rid = item.get("recordId", "")
        if "modelOutput" in item:
            try:
                text = (
                    item["modelOutput"]["output"]["message"]["content"][0]["text"]
                )
            except (KeyError, IndexError, TypeError):
                text = json.dumps(item["modelOutput"])
        else:
            err = item.get("error", {})
            text = f"ERROR: {err.get('errorCode','?')} {err.get('errorMessage','')}"
        result_map[rid] = text

    # Write raw log
    ts_now = datetime.now(timezone.utc).isoformat()
    with raw_log_path.open("a", encoding="utf-8") as fh:
        for rid, raw_text in result_map.items():
            fh.write(json.dumps({
                "id": rid, "timestamp": ts_now, "model_id": MODEL_ID,
                "job_arn": job_arn, "raw_response": raw_text,
                "prompt_context_text": prompt_map.get(rid, ""),
            }) + "\n")

    # Merge results with input rows
    id_to_row = {row["id"]: row.to_dict() for _, row in todo_df.iterrows()}
    new_rows: list[dict] = []
    n_failed = 0

    for rid, raw_text in result_map.items():
        row_dict = id_to_row.get(rid)
        if row_dict is None:
            logger.warning("recordId %s not found in input — skipping.", rid)
            continue
        result = _apply_parsed(
            row_dict=row_dict,
            raw_text=raw_text,
            model_id=MODEL_ID,
            timestamp=ts_now,
            prompt_context=prompt_map.get(rid, ""),
            threshold=confidence_threshold,
        )
        new_rows.append(result)
        if result["confidence"] == "failed":
            n_failed += 1

    # Any input row with no result entry → mark failed
    responded_ids = set(result_map)
    for rid, row_dict in id_to_row.items():
        if rid not in responded_ids:
            logger.warning("No output for id=%s — marking failed.", rid)
            result = dict(row_dict)
            result.update({
                "is_idiomatic": None, "confidence": "failed",
                "reasoning": "no_batch_output",
                "needs_review": True, "model_id": MODEL_ID,
                "timestamp": ts_now, "prompt_context_text": prompt_map.get(rid, ""),
            })
            new_rows.append(result)
            n_failed += 1

    _write_observations(existing_rows + new_rows, output_path, n_failed)


# ---------------------------------------------------------------------------
# Real-time mode
# ---------------------------------------------------------------------------

def _classify_row_realtime(
    row_dict: dict,
    client: BedrockClient,
    prompt_window: int,
    confidence_threshold: str,
) -> tuple[dict, dict]:
    """Classify one row via real-time Converse API (runs in thread pool)."""
    ctx    = _build_prompt_context(row_dict, window=prompt_window)
    prompt = build_user_prompt(row_dict["idiom"], ctx)
    ts     = datetime.now(timezone.utc).isoformat()
    raw    = ""

    try:
        raw = client.invoke_with_retry(
            system=SYSTEM_PROMPT,
            user_message=prompt,
            model_id=MODEL_ID,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    except Exception as exc:
        logger.error("Bedrock error for id=%s: %s", row_dict["id"], exc)
        raw = f"ERROR: {exc}"

    log_entry = {
        "id": row_dict["id"], "timestamp": ts, "model_id": MODEL_ID,
        "prompt_context_text": ctx, "raw_response": raw,
    }
    result = _apply_parsed(row_dict, raw, MODEL_ID, ts, ctx, confidence_threshold)
    return result, log_entry


def _run_realtime(
    todo_df: pd.DataFrame,
    existing_rows: list[dict],
    output_path: Path,
    raw_log_path: Path,
    prompt_window: int,
    confidence_threshold: str,
    workers: int,
) -> None:
    """Classify rows concurrently via real-time Converse API."""
    client    = BedrockClient(model_id=MODEL_ID)
    new_rows: list[dict] = []

    with raw_log_path.open("a", encoding="utf-8") as log_fh:
        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _, row in todo_df.iterrows():
                f = executor.submit(
                    _classify_row_realtime,
                    row.to_dict(), client, prompt_window, confidence_threshold,
                )
                futures_map[f] = row["id"]

            for future in tqdm(as_completed(futures_map), total=len(futures_map),
                               desc="Classifying", unit="row"):
                try:
                    result, log_entry = future.result()
                except Exception as exc:
                    logger.error("Unexpected error for id=%s: %s", futures_map[future], exc)
                    continue
                log_fh.write(json.dumps(log_entry) + "\n")
                log_fh.flush()
                new_rows.append(result)

                if len(new_rows) % 100 == 0:
                    _flush_observations(existing_rows + new_rows, output_path)
                    logger.info(
                        "Progress: %d classified, %d needs_review.",
                        len(new_rows), sum(r.get("needs_review", False) for r in new_rows),
                    )

    n_failed = sum(r.get("confidence") == "failed" for r in new_rows)
    _write_observations(existing_rows + new_rows, output_path, n_failed)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _flush_observations(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def _write_observations(all_rows: list[dict], output_path: Path, n_failed: int) -> None:
    """Write final observations.parquet and log a summary."""
    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(output_path, index=False)

    n_idiomatic = int((out_df["is_idiomatic"] == True).sum())   # noqa: E712
    n_literal   = int((out_df["is_idiomatic"] == False).sum())  # noqa: E712
    n_review    = int(out_df.get("needs_review", pd.Series(dtype=bool)).sum())
    logger.info(
        "Done. Total: %d | idiomatic: %d | literal: %d | needs_review: %d | failed: %d",
        len(out_df), n_idiomatic, n_literal, n_review, n_failed,
    )
    logger.info("Saved observations → %s", output_path)

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
# Main pipeline
# ---------------------------------------------------------------------------

def run_disambiguation(
    data_dir: Path,
    force: bool = False,
    realtime: bool = False,
    s3_uri: str | None = None,
    iam_role: str | None = None,
    workers: int = DEFAULT_REALTIME_WORKERS,
    prompt_window: int = DEFAULT_PROMPT_WINDOW,
    confidence_threshold: str = DEFAULT_CONFIDENCE_THRESHOLD,
    candidates_file: str | None = None,
    output_file: str | None = None,
) -> None:
    """Classify all candidates in candidates.parquet via Nova Micro.

    Parameters
    ----------
    data_dir:
        Project data root.
    force:
        Re-classify all rows, ignoring previous results.
    realtime:
        Use real-time Converse API instead of batch inference.
    s3_uri:
        S3 base URI for batch mode, e.g. ``s3://bucket/prefix/``.
    iam_role:
        IAM role ARN for Bedrock to access S3 (batch mode only).
    workers:
        Concurrent threads for real-time mode.
    prompt_window:
        Sentence radius for prompt context recomputation from speech_text.
    confidence_threshold:
        Minimum confidence to auto-accept; below → needs_review=True.
    candidates_file:
        Override the default candidates.parquet path (e.g. to point at
        gutenberg_candidates.parquet).
    output_file:
        Override the default observations.parquet output path.
    """
    _ensure_nltk()

    interim_dir = data_dir / "interim"
    logs_dir    = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = Path(candidates_file) if candidates_file else interim_dir / "candidates.parquet"
    output_path     = Path(output_file) if output_file else interim_dir / "observations.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path    = logs_dir / "bedrock_raw.jsonl"

    if not candidates_path.exists():
        logger.error("candidates.parquet not found: %s", candidates_path)
        sys.exit(1)

    candidates_df = pd.read_parquet(candidates_path)
    logger.info("Loaded %d candidates from %s.", len(candidates_df), candidates_path.name)

    # Resume: skip already-classified IDs
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if output_path.exists() and not force:
        obs_df = pd.read_parquet(output_path)
        done_ids      = set(obs_df["id"].tolist())
        existing_rows = obs_df.to_dict("records")
        logger.info("Resuming: %d rows already classified.", len(done_ids))

    todo_df = (
        candidates_df[~candidates_df["id"].isin(done_ids)]
        if not force else candidates_df
    )
    logger.info(
        "%d rows to classify (model=%s, prompt_window=±%d, threshold=%s, mode=%s).",
        len(todo_df), MODEL_ID, prompt_window, confidence_threshold,
        "realtime" if realtime else "batch",
    )

    if todo_df.empty:
        logger.info("Nothing to do — all candidates already classified.")
        return

    if realtime:
        _run_realtime(
            todo_df=todo_df,
            existing_rows=existing_rows,
            output_path=output_path,
            raw_log_path=raw_log_path,
            prompt_window=prompt_window,
            confidence_threshold=confidence_threshold,
            workers=workers,
        )
    else:
        if not s3_uri or not iam_role:
            logger.error(
                "Batch mode requires --s3-uri and --iam-role. "
                "Use --realtime for local/testing runs."
            )
            sys.exit(1)
        _run_batch(
            todo_df=todo_df,
            existing_rows=existing_rows,
            output_path=output_path,
            raw_log_path=raw_log_path,
            s3_uri=s3_uri,
            iam_role=iam_role,
            prompt_window=prompt_window,
            confidence_threshold=confidence_threshold,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 02: Disambiguate idiom candidates via Amazon Nova Micro on Bedrock.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Batch mode (default, recommended):\n"
            "  Requires --s3-uri and --iam-role.\n"
            "  Submits one CreateModelInvocationJob; polls until completion.\n"
            "  ~40× cheaper than real-time at $0.0175/1M input tokens.\n\n"
            "Real-time mode (--realtime):\n"
            "  No S3 or IAM role required.  Useful for testing or small\n"
            "  incremental batches.\n\n"
            "confidence-threshold:\n"
            "  medium  accept high+medium; flag low → needs_review (default)\n"
            "  high    accept only high; flag medium+low → needs_review\n"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--force", action="store_true",
                        help="Re-classify all rows, ignoring previous results.")
    parser.add_argument("--realtime", action="store_true",
                        help="Use real-time Converse API instead of batch inference.")
    parser.add_argument("--s3-uri", default=None, metavar="URI",
                        help="S3 base URI for batch I/O, e.g. s3://bucket/prefix/.")
    parser.add_argument("--iam-role", default=None, metavar="ARN",
                        help="IAM role ARN Bedrock assumes to access the S3 bucket.")
    parser.add_argument("--workers", type=int, default=DEFAULT_REALTIME_WORKERS,
                        help=f"Concurrent threads for real-time mode (default: {DEFAULT_REALTIME_WORKERS}).")
    parser.add_argument("--prompt-window", type=int, default=DEFAULT_PROMPT_WINDOW, metavar="N",
                        help="Sentence radius for LLM prompt context (default: %(default)s).")
    parser.add_argument("--confidence-threshold", choices=["high", "medium"],
                        default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Minimum confidence to auto-accept (default: %(default)s).")
    parser.add_argument("--candidates-file", default=None, metavar="PATH",
                        help="Override default candidates.parquet path.")
    parser.add_argument("--output-file", default=None, metavar="PATH",
                        help="Override default observations.parquet output path.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
        realtime=args.realtime,
        s3_uri=args.s3_uri,
        iam_role=args.iam_role,
        workers=args.workers,
        prompt_window=args.prompt_window,
        confidence_threshold=args.confidence_threshold,
        candidates_file=args.candidates_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
