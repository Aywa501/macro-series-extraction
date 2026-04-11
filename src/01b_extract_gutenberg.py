"""Stage 01b — Gutenberg idiom extraction.

Pulls idiom instances from the sedthh/gutenberg_english HuggingFace dataset,
resolves composition dates via the Project Gutenberg catalog and/or the
sedthh metadata subjects field, and writes rows in the Stage 01 schema to:

    data/interim/gutenberg_candidates.parquet

Two extra columns are added to every row:
    source:       always "gutenberg"
    date_method:  "catalog_subjects" | "meta_subjects"

The ``is_idiomatic`` field is absent at this stage; Stage 02 fills it in
exactly as it does for Hansard candidates.

Date resolution (priority order)
---------------------------------
1. pg_catalog.csv.gz  (Method 1 — primary)
   Downloaded once and cached locally.  Join on text_id → Text# column;
   parse the Subjects field for 4-digit years (1600–2004) or century/era
   phrases mapped to midpoint years.  Take the earliest year found.

2. sedthh metadata subjects  (Method 2 — fallback)
   Same parsing logic applied to the ``subjects`` field already present in
   the METADATA JSON.  Less authoritative than the catalog but covers books
   missing from it.

Books where neither method resolves a year in 1803–2004 are discarded.

Resume support
--------------
If ``gutenberg_candidates.parquet`` already exists, any book whose text_id
is already represented is skipped (unless ``--force``).

Usage
-----
    python src/01b_extract_gutenberg.py
    python src/01b_extract_gutenberg.py --force
    python src/01b_extract_gutenberg.py --limit 1000
    python src/01b_extract_gutenberg.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

import nltk
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "config" / "idioms.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
YEAR_MIN = 0
YEAR_MAX = 2026
CONTEXT_WINDOW = 1   # ±1 sentence (Gutenberg prose; narrower than Hansard ±2)

# 4-digit years in the range the spec defines for subject parsing.
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|200[0-4])\b")

# Author birth-death pairs: "Dickens, Charles, 1812-1870"
# Capture groups: (birth, death).  Both must be 4 digits; birth 1300–1950.
_AUTHOR_DATES_RE = re.compile(r"\b(1[3-9]\d{2}|20[0-1]\d)-(\d{4})\b")

# Century / era phrases → representative midpoint year.
_CENTURY_MAP: dict[str, int] = {
    "16th century":  1550,
    "17th century":  1650,
    "18th century":  1750,
    "19th century":  1850,
    "20th century":  1950,
    "16th-century":  1550,
    "17th-century":  1650,
    "18th-century":  1750,
    "19th-century":  1850,
    "20th-century":  1950,
    "victorian":     1870,
    "edwardian":     1906,
    "georgian":      1820,
    "restoration":   1680,
}


# ---------------------------------------------------------------------------
# NLTK
# ---------------------------------------------------------------------------

def _ensure_nltk_punkt() -> None:
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


# ---------------------------------------------------------------------------
# Idiom loading and pattern compilation
# (mirrors 01_extract.py — kept local for self-contained operation)
# ---------------------------------------------------------------------------

def _load_idioms(config_path: Path) -> list[dict]:
    """Load idiom configs from YAML, returning only ``include: true`` entries."""
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return [e for e in cfg.get("idioms", []) if e.get("include", True)]


def _build_pattern(phrase: str) -> re.Pattern:
    """Compile a case-insensitive regex for *phrase* with optional plural suffix."""
    tokens = phrase.strip().split()
    parts = []
    for i, tok in enumerate(tokens):
        escaped = re.escape(tok)
        if i == len(tokens) - 1:
            parts.append(escaped + r"(?:s|ies|es)?")
        else:
            parts.append(escaped)
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


def _context_window(sentences: list[str], idx: int, window: int) -> str:
    """Return sentence at *idx* plus *window* sentences on each side."""
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


# ---------------------------------------------------------------------------
# Date resolution
# ---------------------------------------------------------------------------

def _parse_year(text: str) -> int | None:
    """Extract the earliest plausible composition year from a subject string.

    Searches for 4-digit years (1600–2004) and century/era phrases; returns
    the minimum year found, or None.
    """
    if not text or not isinstance(text, str):
        return None
    years: list[int] = []
    text_lower = text.lower()
    for phrase, year in _CENTURY_MAP.items():
        if phrase in text_lower:
            years.append(year)
    for m in _YEAR_RE.finditer(text):
        years.append(int(m.group(1)))
    return min(years) if years else None


def _parse_year_from_authors(authors_str: str) -> int | None:
    """Estimate composition year from author birth-death dates.

    Parses patterns like ``"Dickens, Charles, 1812-1870"`` and returns the
    career midpoint ``(birth + death) // 2`` for the earliest-born author.
    Returns None if no birth-death pair is found.
    """
    if not authors_str or not isinstance(authors_str, str):
        return None
    midpoints: list[int] = []
    for m in _AUTHOR_DATES_RE.finditer(authors_str):
        birth, death = int(m.group(1)), int(m.group(2))
        if birth < death <= 2005:
            midpoints.append((birth + death) // 2)
    return min(midpoints) if midpoints else None


def _download_catalog(cache_path: Path) -> None:
    """Download pg_catalog.csv.gz once and cache it locally."""
    if cache_path.exists():
        logger.info("Using cached catalog: %s", cache_path)
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    logger.info("Downloading Gutenberg catalog from %s …", CATALOG_URL)
    urlretrieve(CATALOG_URL, str(tmp))
    tmp.rename(cache_path)
    logger.info(
        "Catalog cached (%d KB) → %s",
        cache_path.stat().st_size // 1024, cache_path,
    )


def _load_catalog(cache_path: Path) -> pd.DataFrame:
    """Load pg_catalog.csv.gz and normalise the Text# column name."""
    with gzip.open(cache_path, "rt", encoding="utf-8", errors="replace") as fh:
        df = pd.read_csv(fh, dtype=str)
    # The catalog uses "Text#" — rename for clean joining.
    df = df.rename(columns={"Text#": "text_id"})
    df["text_id"] = df["text_id"].astype(str).str.strip()
    return df


def _resolve_year(
    text_id: str,
    meta_subjects: str,
    catalog_df: pd.DataFrame,
) -> tuple[int | None, str | None]:
    """Return (year, date_method) using Methods 1 → 2 → 3.

    Method 1: catalog Subjects field (explicit date ranges / century phrases).
    Method 2: sedthh metadata subjects field (same parsing).
    Method 3: catalog Authors field — parse birth-death pairs and use the
              career midpoint ``(birth + death) // 2`` as a composition proxy.
              Covers the vast majority of PG books that have no date in Subjects.

    Returns (None, None) if no method succeeds.
    """
    catalog_row = catalog_df[catalog_df["text_id"] == text_id]

    # Method 1 — official catalog Subjects column
    if not catalog_row.empty:
        cat_subjects = str(catalog_row.iloc[0].get("Subjects", "") or "")
        year = _parse_year(cat_subjects)
        if year is not None:
            return year, "catalog_subjects"

    # Method 2 — sedthh metadata subjects field
    if meta_subjects:
        year = _parse_year(meta_subjects)
        if year is not None:
            return year, "meta_subjects"

    # Method 3 — author birth-death dates from catalog Authors field
    if not catalog_row.empty:
        authors_str = str(catalog_row.iloc[0].get("Authors", "") or "")
        year = _parse_year_from_authors(authors_str)
        if year is not None:
            return year, "author_dates"

    return None, None


# ---------------------------------------------------------------------------
# Text field discovery
# ---------------------------------------------------------------------------

def _get_text(record: dict) -> str:
    """Extract the book text from a dataset record, trying multiple field names."""
    for key in ("TEXT", "text", "content", "body"):
        val = record.get(key)
        if val and isinstance(val, str) and len(val) > 100:
            return val.strip()
    return ""


def _get_text_id(meta: dict, record: dict) -> str:
    """Extract the Gutenberg text ID, trying metadata then top-level fields."""
    for key in ("id", "text_id", "Text#", "gutenberg_id"):
        val = meta.get(key) or record.get(key)
        if val:
            return str(val).strip()
    return ""


def _get_subjects(meta: dict) -> str:
    """Extract and normalise the subjects string from metadata."""
    val = meta.get("subjects") or meta.get("subject") or meta.get("Subjects") or ""
    if isinstance(val, list):
        return " ".join(str(s) for s in val)
    return str(val)


# ---------------------------------------------------------------------------
# Per-book extraction
# ---------------------------------------------------------------------------

def _extract_from_book(
    text: str,
    text_id: str,
    year: int,
    date_method: str,
    idiom_configs: list[dict],
    patterns: dict[str, re.Pattern],
    context_window: int,
) -> list[dict]:
    """Extract all idiom candidates from a single book.

    The full text is tokenised into sentences once; each sentence is tested
    against all idiom patterns.  ``speech_text`` is set to ``context_text``
    rather than the full book to keep the output file manageable.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as exc:
        logger.warning("Sentence tokenisation failed for text_id=%s: %s", text_id, exc)
        return []

    speech_id = f"gutenberg_{text_id}"
    candidates: list[dict] = []

    for sent_idx, sentence in enumerate(sentences):
        for config in idiom_configs:
            phrase = config["phrase"]
            m = patterns[phrase].search(sentence)
            if not m:
                continue
            denom = config.get("denomination") or []
            if not denom:
                denom = [config.get("group", "none")]
            ctx = _context_window(sentences, sent_idx, context_window)
            candidates.append({
                "id":            str(uuid.uuid4()),
                "idiom":         phrase,
                "denomination":  denom,
                "group":         config.get("group", "treatment"),
                "year":          year,
                "speech_id":     speech_id,
                "sentence_text": sentence,
                "context_text":  ctx,
                "speech_text":   ctx,   # full book not stored; context used instead
                "raw_match":     m.group(0),
                "source":        "gutenberg",
                "date_method":   date_method,
            })

    return candidates


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _flush(rows: list[dict], output_path: Path) -> None:
    """Write all rows to parquet, normalising the denomination column."""
    df = pd.DataFrame(rows)
    def _to_list(x):
        if isinstance(x, list):
            return x
        try:
            return list(x)   # numpy array, tuple, etc.
        except TypeError:
            return [x] if x is not None else []
    df["denomination"] = df["denomination"].apply(_to_list)
    df.to_parquet(output_path, index=False)


def _write_coverage_report(df: pd.DataFrame, output_path: Path) -> None:
    """Log and save a coverage summary: instances per decade and per idiom."""
    if df.empty:
        logger.info("Coverage report: no instances found.")
        return

    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10

    logger.info("\n=== Gutenberg coverage — instances per idiom ===")
    by_idiom = (
        df.groupby(["group", "idiom"]).size()
        .rename("count").sort_values(ascending=False)
    )
    logger.info("\n%s", by_idiom.to_string())

    logger.info("\n=== Gutenberg coverage — instances per decade ===")
    by_decade = df.groupby("decade").size().rename("count").sort_index()
    logger.info("\n%s", by_decade.to_string())

    logger.info(
        "\n=== Date method breakdown ===\n%s",
        df["date_method"].value_counts().to_string(),
    )

    summary_path = output_path.parent / "gutenberg_coverage.csv"
    by_idiom.reset_index().to_csv(summary_path, index=False)
    logger.info("Coverage CSV → %s", summary_path)

    decade_path = output_path.parent / "gutenberg_coverage_decade.csv"
    by_decade.reset_index().to_csv(decade_path, index=False)
    logger.info("Decade CSV   → %s", decade_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_extraction_gutenberg(
    data_dir: Path,
    config_path: Path,
    catalog_cache: Path,
    force: bool = False,
    limit: int | None = None,
    context_window: int = CONTEXT_WINDOW,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    workers: int | None = None,
) -> None:
    """Extract idiom candidates from the sedthh/gutenberg_english dataset.

    Parameters
    ----------
    data_dir:
        Project data root (``data/interim/`` output goes here).
    config_path:
        Path to ``idioms.yaml``.
    catalog_cache:
        Local cache path for ``pg_catalog.csv.gz``.
    force:
        Re-process all books, ignoring resume state.
    limit:
        If set, stop after processing this many books.
    context_window:
        Sentence context radius around each match.
    year_min / year_max:
        Inclusive year window; books outside are discarded.
    workers:
        Number of threads for parallel per-book extraction.
        Defaults to ``min(8, cpu_count)``.  NLTK's C tokenizer releases the
        GIL so threads give real CPU parallelism here.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required: pip install datasets"
        )
        sys.exit(1)

    _ensure_nltk_punkt()

    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / "gutenberg_candidates.parquet"

    # Catalog
    _download_catalog(catalog_cache)
    catalog_df = _load_catalog(catalog_cache)
    logger.info("Catalog loaded: %d entries.", len(catalog_df))

    # Idiom patterns
    idiom_configs = _load_idioms(config_path)
    patterns = {cfg["phrase"]: _build_pattern(cfg["phrase"]) for cfg in idiom_configs}
    logger.info("Loaded %d idiom patterns from %s.", len(patterns), config_path.name)

    state_path = interim_dir / ".gutenberg_stream_state.json"

    # Resume: collect text_ids already written and saved stream position
    done_text_ids: set[str] = set()
    existing_rows: list[dict] = []
    stream_position = 0
    if output_path.exists() and not force:
        existing_df = pd.read_parquet(output_path)
        existing_rows = existing_df.to_dict("records")
        done_text_ids = {
            str(r.get("speech_id", "")).removeprefix("gutenberg_")
            for r in existing_rows
            if str(r.get("source", "")) == "gutenberg"
        }
        if state_path.exists():
            try:
                stream_position = json.loads(state_path.read_text())["stream_position"]
            except (KeyError, json.JSONDecodeError, OSError):
                stream_position = 0
        logger.info(
            "Resuming: %d existing candidates from %d books (stream position: %d).",
            len(existing_rows), len(done_text_ids), stream_position,
        )
    elif force and state_path.exists():
        state_path.unlink()

    all_rows: list[dict] = list(existing_rows)

    # Stream dataset
    logger.info("Streaming sedthh/gutenberg_english …")
    dataset = load_dataset(
        "sedthh/gutenberg_english",
        split="train",
        streaming=True,
    )
    if stream_position > 0:
        logger.info("Skipping first %d records to resume position …", stream_position)
        dataset = dataset.skip(stream_position)

    n_workers = workers or min(6, os.cpu_count() or 4)
    logger.info("Using %d extraction threads.", n_workers)

    n_seen           = 0   # raw records consumed from stream (for resume position)
    n_processed      = 0   # books submitted for extraction
    n_skipped_resume = 0
    n_skipped_year   = 0
    n_candidates     = 0

    pbar = tqdm(dataset, desc="Gutenberg", unit="book")

    def _update_pbar() -> None:
        pct = 100 * n_skipped_year / n_processed if n_processed else 0
        pbar.set_postfix(candidates=n_candidates, no_year=f"{n_skipped_year} ({pct:.0f}%)")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: dict = {}   # future → text_id

        def _drain(wait: bool = False) -> None:
            """Collect completed futures and update counters."""
            nonlocal n_candidates
            done_set = (
                {f for f in pending if f.done()}
                if not wait
                else set(as_completed(pending))
            )
            for fut in done_set:
                tid = pending.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.warning("Extraction failed for text_id=%s: %s", tid, exc)
                    result = []
                if result:
                    n_candidates += len(result)
                    all_rows.extend(result)
                done_text_ids.add(tid)

        for record in pbar:
            if limit is not None and n_processed >= limit:
                break

            n_seen += 1

            # Parse METADATA JSON
            try:
                raw_meta = record.get("METADATA") or "{}"
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
            except (json.JSONDecodeError, TypeError):
                meta = {}

            text_id = _get_text_id(meta, record)
            if not text_id:
                continue

            if text_id in done_text_ids:
                n_skipped_resume += 1
                continue

            n_processed += 1

            # Date resolution (fast, in-memory — keep on main thread)
            meta_subjects = _get_subjects(meta)
            year, date_method = _resolve_year(text_id, meta_subjects, catalog_df)

            if year is None or not (year_min <= year <= year_max):
                n_skipped_year += 1
                done_text_ids.add(text_id)
                _update_pbar()
                continue

            text = _get_text(record)
            if not text:
                done_text_ids.add(text_id)
                continue

            # Submit CPU-bound work (sentence tokenize + regex) to thread pool
            fut = pool.submit(
                _extract_from_book,
                text, text_id, year, date_method,
                idiom_configs, patterns, context_window,
            )
            pending[fut] = text_id

            # Collect any finished futures without blocking
            _drain(wait=False)
            _update_pbar()

            # Periodic flush every 500 submitted books
            if n_processed % 500 == 0 and all_rows:
                _drain(wait=False)
                _flush(all_rows, output_path)
                state_path.write_text(json.dumps({"stream_position": stream_position + n_seen}))
                logger.info(
                    "Checkpoint: %d books processed, %d candidates so far.",
                    n_processed, len(all_rows),
                )

        # Wait for all in-flight tasks
        _drain(wait=True)
        _update_pbar()

    logger.info(
        "Extraction complete. "
        "Books processed: %d | skipped (resume): %d | skipped (no year): %d",
        n_processed, n_skipped_resume, n_skipped_year,
    )
    logger.info("Total new candidates: %d | Total output rows: %d", n_candidates, len(all_rows))

    if all_rows:
        _flush(all_rows, output_path)
        state_path.write_text(json.dumps({"stream_position": stream_position + n_seen}))
        logger.info("Output → %s", output_path)
        df_out = pd.read_parquet(output_path)
        _write_coverage_report(df_out, output_path)
    else:
        logger.warning("No candidates found. Output not written.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 01b: Extract idiom candidates from Project Gutenberg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output: data/interim/gutenberg_candidates.parquet\n"
            "To feed into Stage 02, merge with candidates.parquet or run\n"
            "Stage 02 separately on this file.\n"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data",
        help="Project data root.",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help="Path to idioms.yaml.",
    )
    parser.add_argument(
        "--catalog-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pg_catalog.csv.gz",
        help="Local cache path for pg_catalog.csv.gz (downloaded once).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process all books, ignoring resume state.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stop after processing N books (default: all).",
    )
    parser.add_argument(
        "--context-window", type=int, default=CONTEXT_WINDOW, metavar="N",
        help=f"Sentence context radius around each match (default: {CONTEXT_WINDOW}).",
    )
    parser.add_argument(
        "--year-min", type=int, default=YEAR_MIN,
        help=f"Earliest composition year to include (default: {YEAR_MIN}).",
    )
    parser.add_argument(
        "--year-max", type=int, default=YEAR_MAX,
        help=f"Latest composition year to include (default: {YEAR_MAX}).",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Extraction threads (default: min(8, cpu_count)).",
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
    run_extraction_gutenberg(
        data_dir=args.data_dir,
        config_path=args.config,
        catalog_cache=args.catalog_cache,
        force=args.force,
        limit=args.limit,
        context_window=args.context_window,
        year_min=args.year_min,
        year_max=args.year_max,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
