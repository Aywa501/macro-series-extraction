"""Stage 01 — Candidate extraction.

Supports two input modes selected via ``--input-format``:

csv (default / trial run)
    Reads two pre-scraped CSV files:
      - pre-1979 sentiment corpus  (1935–1979)
      - v3.10 modern corpus        (1979–2021)
    The 1979 overlap is resolved by taking the pre-1979 file for that year
    and starting the modern file from 1980.

xml (future / full run)
    Parses Hansard XML files from ``data/raw/hansard/``.
    Handles pre-1909 (<speech> text) and post-1909 (<p> within <speech>)
    schemas via ``utils.hansard_parser``.

Both modes produce the same output: ``data/interim/candidates.parquet``.

Usage
-----
    python src/01_extract.py                          # CSV mode
    python src/01_extract.py --input-format xml       # XML mode
    python src/01_extract.py --force                  # ignore checkpoints
    python src/01_extract.py --csv-pre /path/pre.csv --csv-post /path/post.csv
"""

import argparse
import logging
import os
import re
import sys
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import nltk
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.checkpoint import CheckpointStore  # noqa: E402
from utils.hansard_parser import parse_file, parse_zip  # noqa: E402

logger = logging.getLogger(__name__)

# Default CSV locations (relative to project root)
DEFAULT_CSV_PRE = PROJECT_ROOT.parent / "hansard_senti_pre_V21.csv"
DEFAULT_CSV_POST = PROJECT_ROOT.parent / "hansard-speeches-v310.csv"

CSV_CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# NLTK
# ---------------------------------------------------------------------------

def _ensure_nltk_punkt() -> None:
    """Download the punkt tokenizer if not already present."""
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_idioms(config_path: Path) -> list[dict]:
    """Load idiom definitions from YAML, returning only ``include: true`` entries.

    Parameters
    ----------
    config_path:
        Path to ``idioms.yaml``.
    """
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return [e for e in cfg.get("idioms", []) if e.get("include", True)]


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

def _build_pattern(phrase: str) -> re.Pattern:
    """Compile a case-insensitive regex for *phrase* with minor inflection support.

    Allows optional plural suffix (s/es/ies) on the final token so that e.g.
    "penny" also matches "pennies".

    Parameters
    ----------
    phrase:
        Raw idiom phrase string from config.
    """
    tokens = phrase.strip().split()
    parts = []
    for i, tok in enumerate(tokens):
        escaped = re.escape(tok)
        if i == len(tokens) - 1:
            parts.append(escaped + r"(?:s|ies|es)?")
        else:
            parts.append(escaped)
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Matching (shared by both input modes)
# ---------------------------------------------------------------------------

def _context_window(sentences: list[str], idx: int, window: int = 2) -> str:
    """Return sentence at *idx* plus *window* sentences on each side.

    Parameters
    ----------
    sentences:
        All sentences of the speech.
    idx:
        Index of the matching sentence.
    window:
        Number of sentences to include before and after *idx*.
        Pass ``-1`` to return the full speech (all sentences joined).
    """
    if window < 0:
        return " ".join(sentences)
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


def match_idioms_in_speech(
    speech_text: str,
    idiom_configs: list[dict],
    patterns: dict[str, re.Pattern],
    speech_id: str,
    year: int,
    context_window: int = 2,
) -> Iterator[dict]:
    """Yield one candidate dict per idiom match found in *speech_text*.

    Parameters
    ----------
    speech_text:
        Full plain text of a single speech.
    idiom_configs:
        List of idiom config dicts from YAML.
    patterns:
        Precompiled regex patterns keyed by idiom phrase.
    speech_id:
        Source speech identifier.
    year:
        Debate year.
    context_window:
        Sentence context radius. ``-1`` returns the full speech.
    """
    sentences = nltk.sent_tokenize(speech_text)
    for sent_idx, sentence in enumerate(sentences):
        for config in idiom_configs:
            phrase = config["phrase"]
            m = patterns[phrase].search(sentence)
            if m:
                # Denomination: placebo idioms have denomination=[] in YAML;
                # use the group name as a sentinel so rows survive downstream explode.
                denom = config.get("denomination") or []
                if not denom:
                    denom = [config.get("group", "none")]
                yield {
                    "id": str(uuid.uuid4()),
                    "idiom": phrase,
                    "denomination": denom,
                    "group": config.get("group", "treatment"),
                    "year": year,
                    "speech_id": speech_id,
                    "sentence_text": sentence,
                    "context_text": _context_window(sentences, sent_idx, context_window),
                    "speech_text": speech_text,
                    "raw_match": m.group(0),
                }


# ---------------------------------------------------------------------------
# CSV input mode
# ---------------------------------------------------------------------------

def _normalise_pre_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalise a chunk from the pre-1979 sentiment CSV.

    Source columns used: ``pp_id``, ``speech``, ``speaker_name``,
    ``speech_date``, ``year``.

    Parameters
    ----------
    chunk:
        Raw dataframe chunk from ``hansard_senti_pre_V21.csv``.
    """
    out = pd.DataFrame()
    out["speech_id"] = chunk["pp_id"].astype(str)
    out["text"] = chunk["speech"].astype(str)
    out["year"] = pd.to_numeric(chunk["year"], errors="coerce").astype("Int64")
    out["date"] = chunk["speech_date"].astype(str)
    out["speaker"] = chunk["speaker_name"].astype(str)
    # Drop rows with no text or no year
    out = out[out["text"].notna() & (out["text"] != "nan") & out["year"].notna()]
    return out


def _normalise_post_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalise a chunk from the v3.10 modern CSV.

    Filters to ``speech_class == 'Speech'`` rows only. Source columns used:
    ``id``, ``speech``, ``speakername``, ``date``, ``year``.

    Parameters
    ----------
    chunk:
        Raw dataframe chunk from ``hansard-speeches-v310.csv``.
    """
    if "speech_class" in chunk.columns:
        chunk = chunk[chunk["speech_class"] == "Speech"]
    out = pd.DataFrame()
    out["speech_id"] = chunk["id"].astype(str)
    out["text"] = chunk["speech"].astype(str)
    out["year"] = pd.to_numeric(chunk["year"], errors="coerce").astype("Int64")
    out["date"] = chunk["date"].astype(str)
    out["speaker"] = chunk["speakername"].astype(str)
    out = out[out["text"].notna() & (out["text"] != "nan") & out["year"].notna()]
    return out


def _process_csv_file(
    csv_path: Path,
    normalise_fn,
    year_min: int | None,
    year_max: int | None,
    idiom_configs: list[dict],
    patterns: dict[str, re.Pattern],
    checkpoint: CheckpointStore,
    force: bool,
    context_window: int = 2,
) -> list[dict]:
    """Process one CSV file in chunks, returning matched candidate dicts.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    normalise_fn:
        Column-normalisation function for this file's schema.
    year_min:
        If set, skip rows with year < year_min.
    year_max:
        If set, skip rows with year > year_max.
    idiom_configs:
        Idiom config list from YAML.
    patterns:
        Precompiled regex patterns.
    checkpoint:
        CheckpointStore for resumption.
    force:
        If True, ignore checkpoints.
    context_window:
        Sentence context radius for :func:`match_idioms_in_speech`.
    """
    rows: list[dict] = []
    file_key_base = csv_path.name

    # Note: a pre-scan row count is unreliable for these files because
    # Hansard speech text contains embedded newlines inside quoted CSV fields.
    # Both a binary wc-l count and a usecols=[0] fast-parse count physical
    # lines rather than logical CSV records, inflating the total ~2-3×.
    # We therefore run tqdm without a total so it shows elapsed time and
    # throughput rather than a misleading percentage bar.
    logger.info("Processing %s …", csv_path.name)

    reader = pd.read_csv(
        csv_path,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="warn",
    )

    for chunk_idx, raw_chunk in enumerate(
        tqdm(reader, desc=csv_path.name, unit="chunk")
    ):
        chunk_key = f"{file_key_base}:chunk:{chunk_idx}"
        if not force and checkpoint.is_done(chunk_key):
            continue

        chunk = normalise_fn(raw_chunk)

        # Year filtering
        if year_min is not None:
            chunk = chunk[chunk["year"] >= year_min]
        if year_max is not None:
            chunk = chunk[chunk["year"] <= year_max]

        if chunk.empty:
            checkpoint.mark_done(chunk_key)
            continue

        for _, speech_row in chunk.iterrows():
            text = str(speech_row["text"]).strip()
            if not text or text == "nan":
                continue
            year_val = speech_row["year"]
            if pd.isna(year_val):
                continue
            for candidate in match_idioms_in_speech(
                text,
                idiom_configs,
                patterns,
                speech_id=str(speech_row["speech_id"]),
                year=int(year_val),
                context_window=context_window,
            ):
                rows.append(candidate)

        checkpoint.mark_done(chunk_key)

    logger.info("%s: found %d candidates.", csv_path.name, len(rows))
    return rows


def run_extraction_csv(
    data_dir: Path,
    config_path: Path,
    csv_pre: Path,
    csv_post: Path,
    force: bool = False,
    context_window: int = 2,
) -> None:
    """Extract candidates from the two pre-scraped CSV corpora.

    Covers 1935–2021 by combining:
    - Pre-1979 sentiment corpus (1935–1979)
    - v3.10 modern corpus (1980–2021; 1979 taken from pre file to avoid overlap)

    Parameters
    ----------
    data_dir:
        Project data root.
    config_path:
        Path to ``idioms.yaml``.
    csv_pre:
        Path to ``hansard_senti_pre_V21.csv``.
    csv_post:
        Path to ``hansard-speeches-v310.csv``.
    force:
        Ignore checkpoints and reprocess everything.
    """
    _ensure_nltk_punkt()

    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / "candidates.parquet"
    checkpoint_path = interim_dir / ".extract_progress.json"

    for p in (csv_pre, csv_post):
        if not p.exists():
            logger.error("CSV file not found: %s", p)
            sys.exit(1)

    idiom_configs = load_idioms(config_path)
    patterns = {cfg["phrase"]: _build_pattern(cfg["phrase"]) for cfg in idiom_configs}
    logger.info("Loaded %d idiom patterns.", len(patterns))

    checkpoint = CheckpointStore(checkpoint_path)
    if force:
        logger.info("--force: ignoring existing checkpoints.")

    # Load existing candidates for append/resume
    existing_rows: list[dict] = []
    if output_path.exists() and not force:
        existing_rows = pd.read_parquet(output_path).to_dict("records")
        logger.info("Resuming: %d existing candidates loaded.", len(existing_rows))

    all_rows = list(existing_rows)

    # Pre-1979 file: all years up to and including 1979
    logger.info("Processing pre-1979 CSV …")
    all_rows += _process_csv_file(
        csv_path=csv_pre,
        normalise_fn=_normalise_pre_chunk,
        year_min=None,
        year_max=1979,
        idiom_configs=idiom_configs,
        patterns=patterns,
        checkpoint=checkpoint,
        force=force,
        context_window=context_window,
    )

    # Post-1979 file: 1980 onwards (skip 1979 to avoid overlap)
    logger.info("Processing post-1979 CSV …")
    all_rows += _process_csv_file(
        csv_path=csv_post,
        normalise_fn=_normalise_post_chunk,
        year_min=1980,
        year_max=None,
        idiom_configs=idiom_configs,
        patterns=patterns,
        checkpoint=checkpoint,
        force=force,
        context_window=context_window,
    )

    logger.info("Total candidates found: %d", len(all_rows))

    if all_rows:
        df = pd.DataFrame(all_rows)
        df["denomination"] = df["denomination"].apply(
            lambda x: x if isinstance(x, list) else [x]
        )
        df.to_parquet(output_path, index=False)
        logger.info("Wrote %d candidates to %s.", len(df), output_path)
    else:
        logger.warning("No candidates found. Output not written.")


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level so it is picklable)
# ---------------------------------------------------------------------------

_W_IDIOM_CONFIGS: list[dict] = []
_W_PATTERNS: dict[str, re.Pattern] = {}
_W_CONTEXT_WINDOW: int = 2


def _worker_init(idiom_configs: list[dict], patterns: dict, context_window: int) -> None:
    """Pool initialiser: store shared state once per worker process."""
    global _W_IDIOM_CONFIGS, _W_PATTERNS, _W_CONTEXT_WINDOW
    _W_IDIOM_CONFIGS = idiom_configs
    _W_PATTERNS = patterns
    _W_CONTEXT_WINDOW = context_window
    _ensure_nltk_punkt()


def _worker_fn(file_path: Path) -> tuple[str, list[dict] | str]:
    """Parse one ZIP/XML file and return matched candidates.

    Returns
    -------
    tuple
        ``(filename, candidates)`` on success where candidates is a list of
        dicts, or ``(filename, error_message)`` on failure where error_message
        is a str.
    """
    try:
        if file_path.suffix == ".zip":
            speeches = parse_zip(file_path)
        else:
            speeches = parse_file(file_path)
    except Exception as exc:
        return file_path.name, str(exc)

    candidates: list[dict] = []
    for speech in speeches:
        for candidate in match_idioms_in_speech(
            speech["text"],
            _W_IDIOM_CONFIGS,
            _W_PATTERNS,
            speech_id=speech["speech_id"],
            year=speech["year"],
            context_window=_W_CONTEXT_WINDOW,
        ):
            candidates.append(candidate)
    return file_path.name, candidates


# ---------------------------------------------------------------------------
# XML / ZIP input mode  (Draft Hansard v3 bulk data release)
# ---------------------------------------------------------------------------

# Default hansard ZIP directory — one level above project root
DEFAULT_HANSARD_DIR = PROJECT_ROOT.parent / "data" / "raw" / "hansard"


def run_extraction_xml(
    data_dir: Path,
    config_path: Path,
    hansard_dir: Path | None = None,
    force: bool = False,
    workers: int | None = None,
    context_window: int = 2,
) -> None:
    """Extract candidates from Draft Hansard v3 ZIP archives.

    Each ZIP is expected to contain one XML file conforming to
    ``Draft_Hansard_v3.xsd``.  Speech text is extracted from
    ``<membercontribution>`` elements; year from ``<date format="YYYY-MM-DD">``
    context elements.

    Falls back to legacy plain-XML processing (``parse_file``) for any
    ``.xml`` files found alongside the ZIPs.

    Parameters
    ----------
    data_dir:
        Project data root (``interim/`` output goes here).
    config_path:
        Path to ``idioms.yaml``.
    hansard_dir:
        Directory containing the ``.zip`` (or ``.xml``) Hansard files.
        Defaults to ``<project_root>/../data/raw/hansard/``.
    force:
        Ignore checkpoints and reprocess all files.
    workers:
        Number of worker processes for parallel ZIP parsing.
        Defaults to ``os.cpu_count() - 1`` (minimum 1).
    context_window:
        Sentence radius passed to :func:`match_idioms_in_speech`.
        ``-1`` uses the full speech text. Default 2.
    """
    _ensure_nltk_punkt()

    if hansard_dir is None:
        hansard_dir = DEFAULT_HANSARD_DIR

    if not hansard_dir.exists():
        logger.error("Hansard directory not found: %s", hansard_dir)
        import sys; sys.exit(1)

    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / "candidates.parquet"
    checkpoint_path = interim_dir / ".extract_progress.json"

    idiom_configs = load_idioms(config_path)
    patterns = {cfg["phrase"]: _build_pattern(cfg["phrase"]) for cfg in idiom_configs}
    logger.info("Loaded %d idiom patterns.", len(patterns))

    checkpoint = CheckpointStore(checkpoint_path)
    if force:
        logger.info("--force: ignoring existing checkpoints.")

    # Collect all ZIP files (primary) and any loose XML files (legacy)
    zip_files = sorted(hansard_dir.glob("*.zip"))
    xml_files = sorted(hansard_dir.glob("*.xml"))
    all_files = zip_files + xml_files

    if not all_files:
        logger.warning(
            "No .zip or .xml files found in %s. Exiting.", hansard_dir
        )
        return

    logger.info(
        "Found %d ZIP + %d XML files in %s.",
        len(zip_files), len(xml_files), hansard_dir,
    )

    existing_rows: list[dict] = []
    if output_path.exists() and not force:
        existing_rows = pd.read_parquet(output_path).to_dict("records")
        logger.info("Resuming: %d existing candidates loaded.", len(existing_rows))

    all_rows = list(existing_rows)

    # Filter to files not yet processed
    files_to_do = [f for f in all_files if force or not checkpoint.is_done(f.name)]
    files_skipped = len(all_files) - len(files_to_do)
    logger.info(
        "%d files to process, %d already done (cached).",
        len(files_to_do), files_skipped,
    )

    if not files_to_do:
        logger.info("Nothing to do.")
        _flush_candidates(all_rows, output_path)
        return

    n_workers = max(1, (workers or os.cpu_count() or 2) - 1)
    logger.info("Using %d worker processes.", n_workers)

    files_processed = 0
    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(idiom_configs, patterns, context_window),
    ) as pool:
        for filename, result in tqdm(
            pool.imap_unordered(_worker_fn, files_to_do),
            total=len(files_to_do),
            desc="Extracting",
            unit="file",
        ):
            if isinstance(result, str):
                # Worker returned an error message
                logger.error("Failed to parse %s: %s", filename, result)
            else:
                all_rows.extend(result)

            checkpoint.mark_done(filename)
            files_processed += 1

            if files_processed % 100 == 0:
                _flush_candidates(all_rows, output_path)
                logger.info(
                    "Progress: %d / %d files done, %d candidates so far.",
                    files_processed, len(files_to_do), len(all_rows),
                )

    logger.info(
        "Done. Processed: %d, skipped (cached): %d. Total candidates: %d.",
        files_processed, files_skipped, len(all_rows),
    )
    _flush_candidates(all_rows, output_path)


def _flush_candidates(rows: list[dict], output_path: Path) -> None:
    """Write *rows* to *output_path* as parquet, normalising denomination column."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["denomination"] = df["denomination"].apply(
        lambda x: x if isinstance(x, list) else [x]
    )
    df.to_parquet(output_path, index=False)
    logger.info("Flushed %d candidates to %s.", len(df), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 01: Extract idiom candidates from Hansard data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input modes:\n"
            "  csv  Read two pre-scraped CSV files (trial run, default)\n"
            "  xml  Read Hansard XML files from data/raw/hansard/ (full run)\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Project data root (default: <project_root>/data).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "idioms.yaml",
    )
    parser.add_argument(
        "--input-format",
        choices=["csv", "xml"],
        default="csv",
        help="Input format: csv (pre-scraped, default) or xml (ZIP/XML Hansard bulk data).",
    )
    parser.add_argument(
        "--hansard-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Hansard .zip (or .xml) files for xml mode. "
            f"Default: {DEFAULT_HANSARD_DIR}"
        ),
    )
    parser.add_argument(
        "--csv-pre",
        type=Path,
        default=DEFAULT_CSV_PRE,
        help="Path to pre-1979 CSV (hansard_senti_pre_V21.csv).",
    )
    parser.add_argument(
        "--csv-post",
        type=Path,
        default=DEFAULT_CSV_POST,
        help="Path to post-1979 CSV (hansard-speeches-v310.csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process all input, ignoring checkpoints.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Sentence context radius around each match "
            "(0=sentence only, 1=±1, 2=±2, -1=full speech). Default: 2."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes for XML mode (default: cpu_count - 1).",
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

    if args.input_format == "csv":
        run_extraction_csv(
            data_dir=args.data_dir,
            config_path=args.config,
            csv_pre=args.csv_pre,
            csv_post=args.csv_post,
            force=args.force,
            context_window=args.context_window,
        )
    else:
        run_extraction_xml(
            data_dir=args.data_dir,
            config_path=args.config,
            hansard_dir=args.hansard_dir,
            force=args.force,
            workers=args.workers,
            context_window=args.context_window,
        )


if __name__ == "__main__":
    main()
