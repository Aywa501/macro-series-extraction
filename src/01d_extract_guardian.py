"""Stage 01d — Guardian/Observer ProQuest XML idiom extraction.

Reads Guardian/Observer article XML files from the ProQuest Historical
Newspapers archive, stored as a nested zip structure in S3:

    s3://idiom-index/raw/guardian/GuardianObserver.zip
        XML/YYYYMMDD_YYYYMMDD/GO_*.zip   ← inner zips, one per batch
            475043769.xml                 ← one XML file per article

ProQuest XML schema (one file per article):
    <Record>
        <RecordID>         — unique article ID
        <NumericPubDate>   — YYYYMMDD
        <AlphaPubDate>     — "Apr 27, 1910" (human-readable)
        <RecordTitle>      — headline
        <ObjectType>       — "Article", "Feature", etc.
        <Publication>/<Title>  — "The Manchester Guardian (1901-1959)"
        <FullText>         — full article text, already OCR'd by ProQuest

Output schema (mirrors Stage 01 / 01b):
    id, idiom, denomination, group, year, speech_id, sentence_text,
    context_text, speech_text, raw_match, source, date_method,
    article_id, object_type, publication

Writes partitioned parquet to:
    s3://idiom-index/interim/guardian/year=YYYY/part-NNNN.parquet

Checkpointing
-------------
Processed inner zip names are saved to a JSON checkpoint on S3. Crashed
runs resume automatically, skipping already-processed inner zips.

Parallelism
-----------
Uses multiprocessing.Pool. Each worker processes one inner zip (fetches
it from S3, parses all XMLs inside, matches idioms). On a c6i.4xlarge
(16 vCPU) with 16 workers this should process ~300 inner zips/hour.

Usage
-----
    # Dry run — process first 3 inner zips, print summary, no S3 write
    python src/01d_extract_guardian.py --dry-run --limit 3

    # Full run
    python src/01d_extract_guardian.py

    # Resume after crash (automatic — checkpoint is checked on startup)
    python src/01d_extract_guardian.py

    # Force reprocess everything
    python src/01d_extract_guardian.py --force
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing
import re
import struct
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import boto3
import nltk
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH   = PROJECT_ROOT / "config" / "idioms.yaml"
CONTEXT_WIN   = 2
ROWS_PER_PART = 50_000

BUCKET     = "idiom-index"
S3_RAW_KEY = "raw/guardian/GuardianObserver.zip"
S3_OUT_PFX = "interim/guardian"
CHECKPOINT_KEY = "interim/guardian/.checkpoint.json"
TOTAL_SIZE = 231256377493

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    for r in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r.split("/")[-1], quiet=True)


# ---------------------------------------------------------------------------
# Idiom loading
# ---------------------------------------------------------------------------

def _load_idioms(config_path: Path) -> list[dict]:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    return [e for e in cfg.get("idioms", []) if e.get("include", True)]


def _build_pattern(phrase: str) -> re.Pattern:
    parts = [re.escape(w) for w in phrase.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# S3 byte-range helpers
# ---------------------------------------------------------------------------

def _s3_get(s3_client, start: int, end: int) -> bytes:
    resp = s3_client.get_object(
        Bucket=BUCKET, Key=S3_RAW_KEY, Range=f"bytes={start}-{end}"
    )
    return resp["Body"].read()


def _read_outer_central_directory(s3_client) -> list[tuple[str, int, int]]:
    """Read ZIP64 central directory; return (filename, local_offset, comp_size)."""
    tail = _s3_get(s3_client, TOTAL_SIZE - 131072, TOTAL_SIZE - 1)

    # ZIP64 EOCD locator
    loc_idx = tail.rfind(b"PK\x06\x07")
    z64_offset = struct.unpack_from("<Q", tail, loc_idx + 8)[0]
    z64 = _s3_get(s3_client, z64_offset, z64_offset + 4095)
    cd_size   = struct.unpack_from("<Q", z64, 40)[0]
    cd_offset = struct.unpack_from("<Q", z64, 48)[0]

    cd_data = _s3_get(s3_client, cd_offset, cd_offset + cd_size - 1)

    entries: list[tuple[str, int, int]] = []
    pos = 0
    while pos < len(cd_data) - 4:
        if cd_data[pos:pos+4] != b"PK\x01\x02":
            break
        comp_size   = struct.unpack_from("<I", cd_data, pos+20)[0]
        fname_len   = struct.unpack_from("<H", cd_data, pos+28)[0]
        extra_len   = struct.unpack_from("<H", cd_data, pos+30)[0]
        comment_len = struct.unpack_from("<H", cd_data, pos+32)[0]
        loc_offset  = struct.unpack_from("<I", cd_data, pos+42)[0]
        fname = cd_data[pos+46:pos+46+fname_len].decode("utf-8", errors="replace")

        extra = cd_data[pos+46+fname_len:pos+46+fname_len+extra_len]
        ep = 0
        while ep < len(extra) - 4:
            tag = struct.unpack_from("<H", extra, ep)[0]
            sz  = struct.unpack_from("<H", extra, ep+2)[0]
            if tag == 0x0001:
                vals = [struct.unpack_from("<Q", extra, ep+4+i*8)[0]
                        for i in range(sz // 8)]
                if comp_size  == 0xFFFFFFFF and vals: comp_size  = vals.pop(0)
                if loc_offset == 0xFFFFFFFF and vals: loc_offset = vals.pop(0)
            ep += 4 + sz

        entries.append((fname, loc_offset, comp_size))
        pos += 46 + fname_len + extra_len + comment_len

    return entries


def _fetch_inner_zip(s3_client, loc_offset: int, comp_size: int) -> zipfile.ZipFile:
    """Fetch one inner zip from S3 by byte range."""
    lfh = _s3_get(s3_client, loc_offset, loc_offset + 299)
    fname_len = struct.unpack_from("<H", lfh, 26)[0]
    extra_len = struct.unpack_from("<H", lfh, 28)[0]
    data_start = loc_offset + 30 + fname_len + extra_len
    data = _s3_get(s3_client, data_start, data_start + comp_size - 1)
    return zipfile.ZipFile(io.BytesIO(data))


# ---------------------------------------------------------------------------
# ProQuest XML parsing
# ---------------------------------------------------------------------------

def _parse_proquest_xml(xml_bytes: bytes) -> dict | None:
    """Parse one ProQuest XML article record. Returns dict or None if unparseable."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    def text(tag: str) -> str:
        el = root.find(f".//{tag}")
        return (el.text or "").strip() if el is not None else ""

    full_text = text("FullText")
    if not full_text:
        return None

    date_str = text("NumericPubDate")   # YYYYMMDD
    year = int(date_str[:4]) if len(date_str) >= 4 and date_str[:4].isdigit() else None
    if not year:
        return None

    object_types = [el.text.strip() for el in root.findall(".//ObjectType") if el.text]

    return {
        "record_id":    text("RecordID"),
        "year":         year,
        "pub_date":     date_str,
        "headline":     text("RecordTitle"),
        "publication":  text("Title"),
        "object_type":  "; ".join(object_types),
        "full_text":    full_text,
    }


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------

def _context_window(sentences: list[str], idx: int, window: int) -> str:
    lo = max(0, idx - window)
    hi = min(len(sentences), idx + window + 1)
    return " ".join(sentences[lo:hi])


def _extract_from_article(
    article: dict,
    idiom_configs: list[dict],
    patterns: dict[str, re.Pattern],
    quick_filters: list[str],
    context_window: int,
) -> list[dict]:
    full_text = article["full_text"]
    full_lower = full_text.lower()

    # Fast pre-filter: skip sentence tokenization entirely if no phrase
    # appears anywhere in the article (handles ~99% of articles).
    if not any(f in full_lower for f in quick_filters):
        return []

    try:
        sentences = nltk.sent_tokenize(full_text)
    except Exception:
        return []

    candidates: list[dict] = []
    speech_id = f"guardian_{article['record_id']}"

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
                "year":          article["year"],
                "speech_id":     speech_id,
                "sentence_text": sentence,
                "context_text":  ctx,
                "speech_text":   ctx,
                "raw_match":     m.group(0),
                "source":        "guardian",
                "date_method":   "xml_publication_date",
                "article_id":    article["record_id"],
                "object_type":   article["object_type"],
                "publication":   article["publication"],
            })
    return candidates


# ---------------------------------------------------------------------------
# Worker — runs in subprocess, one per inner zip
# ---------------------------------------------------------------------------

def _process_inner_zip(args: tuple) -> tuple[str, list[dict], str | None]:
    """
    Process one inner zip (fetched from S3).
    Returns (inner_zip_name, candidates, error_or_None).
    Top-level for multiprocessing pickling.
    """
    inner_zip_name, loc_offset, comp_size, config_path_str, context_window = args

    _ensure_nltk()
    idiom_configs = _load_idioms(Path(config_path_str))
    patterns = {c["phrase"]: _build_pattern(c["phrase"]) for c in idiom_configs}
    # Lowercase keyword fragments for fast pre-filter (first word of each phrase)
    quick_filters = list({c["phrase"].split()[0].lower() for c in idiom_configs})

    s3 = boto3.client("s3")
    try:
        inner = _fetch_inner_zip(s3, loc_offset, comp_size)
    except Exception as e:
        return inner_zip_name, [], f"Failed to fetch: {e}"

    xml_names = [n for n in inner.namelist() if n.lower().endswith(".xml")]
    candidates: list[dict] = []

    for xml_name in xml_names:
        try:
            xml_bytes = inner.read(xml_name)
        except Exception:
            continue
        article = _parse_proquest_xml(xml_bytes)
        if not article:
            continue
        candidates.extend(
            _extract_from_article(article, idiom_configs, patterns, quick_filters, context_window)
        )

    return inner_zip_name, candidates, None


# ---------------------------------------------------------------------------
# Checkpoint helpers (stored on S3)
# ---------------------------------------------------------------------------

def _load_checkpoint(s3_client) -> set[str]:
    try:
        obj = s3_client.get_object(Bucket=BUCKET, Key=CHECKPOINT_KEY)
        return set(json.loads(obj["Body"].read()).get("done", []))
    except s3_client.exceptions.NoSuchKey:
        return set()
    except Exception:
        return set()


def _save_checkpoint(s3_client, done: set[str]) -> None:
    body = json.dumps({"done": sorted(done)}).encode()
    s3_client.put_object(Bucket=BUCKET, Key=CHECKPOINT_KEY, Body=body)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_partition_s3(s3_client, rows: list[dict], part_idx: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["denomination"] = df["denomination"].apply(
        lambda x: x if isinstance(x, list) else list(x)
    )
    for year, group in df.groupby("year"):
        buf = io.BytesIO()
        group.drop(columns=["year"]).to_parquet(buf, index=False)
        buf.seek(0)
        key = f"{S3_OUT_PFX}/year={year}/part-{part_idx:04d}.parquet"
        s3_client.put_object(Bucket=BUCKET, Key=key, Body=buf.read())
    logger.info("Uploaded part-%04d: %d rows, %d years",
                part_idx, len(df), df["year"].nunique())


def _write_partition_local(rows: list[dict], output_dir: Path, part_idx: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["denomination"] = df["denomination"].apply(
        lambda x: x if isinstance(x, list) else list(x)
    )
    for year, group in df.groupby("year"):
        year_dir = output_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year"]).to_parquet(
            year_dir / f"part-{part_idx:04d}.parquet", index=False
        )
    logger.info("Wrote part-%04d: %d rows, %d years",
                part_idx, len(df), df["year"].nunique())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    config_path:    Path,
    context_window: int,
    workers:        int,
    limit:          int | None,
    force:          bool,
    dry_run:        bool,
    output_dir:     Path | None,
) -> None:
    _ensure_nltk()
    s3 = boto3.client("s3")

    logger.info("Reading outer zip central directory from S3 …")
    all_entries = _read_outer_central_directory(s3)
    xml_zips = [e for e in all_entries if e[0].startswith("XML/") and e[0].endswith(".zip")]
    logger.info("Found %d XML inner zips", len(xml_zips))

    if limit:
        xml_zips = xml_zips[:limit]
        logger.info("Limiting to first %d inner zips", limit)

    # Checkpoint
    done_set: set[str] = set()
    if not force and not dry_run:
        done_set = _load_checkpoint(s3)
        logger.info("Checkpoint: %d already done", len(done_set))

    pending = [(name, off, sz) for name, off, sz in xml_zips if name not in done_set]
    logger.info("%d inner zips to process", len(pending))

    if not pending:
        logger.info("Nothing to do.")
        return

    worker_args = [
        (name, off, sz, str(config_path), context_window)
        for name, off, sz in pending
    ]

    accumulated: list[dict] = []
    part_idx = len(done_set) // 10  # rough starting part number
    total_candidates = 0

    with multiprocessing.Pool(processes=workers) as pool:
        for inner_name, candidates, error in pool.imap_unordered(
            _process_inner_zip, worker_args, chunksize=2
        ):
            if error:
                logger.warning("Error in %s: %s", inner_name, error)
            else:
                total_candidates += len(candidates)
                accumulated.extend(candidates)
                logger.debug("%s → %d candidates (total %d)",
                             inner_name, len(candidates), total_candidates)

            done_set.add(inner_name)

            if len(accumulated) >= ROWS_PER_PART:
                if dry_run:
                    logger.info("[dry-run] Would write %d rows", len(accumulated))
                elif output_dir:
                    _write_partition_local(accumulated, output_dir, part_idx)
                else:
                    _write_partition_s3(s3, accumulated, part_idx)

                accumulated = []
                part_idx += 1

                if not dry_run:
                    _save_checkpoint(s3, done_set)

    # Flush remainder
    if accumulated:
        if dry_run:
            logger.info("[dry-run] Would write final %d rows", len(accumulated))
        elif output_dir:
            _write_partition_local(accumulated, output_dir, part_idx)
        else:
            _write_partition_s3(s3, accumulated, part_idx)

    if not dry_run:
        _save_checkpoint(s3, done_set)

    logger.info("Done. Processed %d inner zips, %d total candidates.",
                len(done_set), total_candidates)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 01d: Extract idiom candidates from Guardian ProQuest XML archive."
    )
    parser.add_argument("--config",  type=Path, default=CONFIG_PATH)
    parser.add_argument("--context-window", type=int, default=CONTEXT_WIN)
    parser.add_argument("--workers", type=int,
                        default=min(multiprocessing.cpu_count(), 16))
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process at most N inner zips (for testing).")
    parser.add_argument("--force",   action="store_true",
                        help="Ignore checkpoint, reprocess all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and match but do not write output to S3.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write parquet locally instead of S3 (for testing).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(
        config_path    = args.config,
        context_window = args.context_window,
        workers        = args.workers,
        limit          = args.limit,
        force          = args.force,
        dry_run        = args.dry_run,
        output_dir     = args.output_dir,
    )


if __name__ == "__main__":
    main()
