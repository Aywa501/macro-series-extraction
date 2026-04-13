"""Stage 00 — Parse Penn Parsed Corpora of Historical English.

Reads .psd bracket-notation files from a corpus directory and outputs a
parquet file of (sentence, year) pairs suitable for Stage 01.

Penn Corpus Structure
---------------------
Files end in .psd and contain sequences of top-level parenthesised parse
trees.  Each tree ends with an (ID ...) node whose value encodes the source
filename and sentence number:

    ( (IP-MAT (NP-SBJ (D The) (N man))
              (VBD saw) (NP-OBJ (D the) (N woman))
              (. .))
      (ID CMANCRI,1.2))

Year is resolved in priority order:
  1. 4-digit year in the ID token after the last underscore, e.g. CMANCRI_1350
  2. Year range in the ID token, e.g. _1350-1400 → midpoint 1375
  3. A JSON file mapping filename stems to years (--year-map)
  4. 4-digit year found anywhere in the .psd filename itself

Any sentence whose year cannot be resolved is silently dropped.

Usage
-----
    python src/00_parse_penn.py
    python src/00_parse_penn.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)

# POS tags whose content should be treated as editorial / structural markup
_SKIP_TAGS = frozenset({
    "ID", "CODE", "META", "METADATA", "REF", "FW-ID",
    "LS", "LATIN", "X",
})

# Words that are traces, empty categories, or disfluency markers
_TRACE_RE = re.compile(r"^\*|^0$|^e$|^\+$|^xxx$", re.IGNORECASE)

# PPCME2  period suffix → (start_year, end_year)
_PPCME2_PERIODS: dict[str, tuple[int, int]] = {
    "m1": (1150, 1250), "m2": (1250, 1350),
    "m3": (1350, 1420), "m4": (1420, 1500),
}

# PPCEME  period suffix → (start_year, end_year)
_PPCEME_PERIODS: dict[str, tuple[int, int]] = {
    "e1": (1500, 1570), "e2": (1570, 1640), "e3": (1640, 1710),
}


def _year_from_period_suffix(stem: str) -> int | None:
    """Resolve PPCHE period codes (m1-m4, e1-e3) embedded in filename stems."""
    stem_lower = stem.lower()
    for prefix, table in [("e", _PPCEME_PERIODS), ("m", _PPCME2_PERIODS)]:
        m = re.search(rf"-({prefix}\d+)(?:-|$)", stem_lower)
        if not m:
            continue
        code  = m.group(1)
        parts = [prefix + d for d in code[1:]]
        ranges = [table.get(p) for p in parts]
        ranges = [r for r in ranges if r is not None]
        if ranges:
            lo = min(r[0] for r in ranges)
            hi = max(r[1] for r in ranges)
            return (lo + hi) // 2
    return None


def _split_toplevel(text: str) -> list[str]:
    """Return a list of top-level parenthesised expressions from raw text."""
    tokens = []
    depth  = 0
    start  = -1
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start != -1:
                tokens.append(text[start : i + 1])
                start = -1
    return tokens


def _extract_words(tree: str) -> str:
    """Return the surface word sequence from a Penn bracket tree."""
    words: list[str] = []
    for m in re.finditer(r"\(([A-Z$@\-\+\^]+)\s+([^\s()]+)\)", tree):
        tag, word = m.group(1), m.group(2)
        if tag in _SKIP_TAGS:
            continue
        if _TRACE_RE.match(word):
            continue
        words.append(word)
    return " ".join(words)


def _extract_id_stem(tree: str) -> str | None:
    """Return the filename stem from the (ID ...) node, or None."""
    m = re.search(r"\(ID\s+([^\s,\)]+)", tree)
    if m:
        return m.group(1).upper()
    return None


def _year_from_stem(stem: str) -> int | None:
    """Extract a year from a Penn corpus filename stem.

    Resolution order:
    1. 4-digit year with one 'x' digit (e.g. 176x → 1765)
    2. Explicit 4-digit year appearing after a hyphen (PPCMBE2 convention)
    3. PPCHE period code (m1-m4, e1-e3)
    4. Any 4-digit number in a plausible historical range (1000-1950)
    """
    stem_upper = stem.upper()

    m = re.search(r"-(\d{3}x)", stem_upper, re.IGNORECASE)
    if m:
        return int(m.group(1)[:3]) * 10 + 5

    for candidate in re.findall(r"(?<=-)\d{4}(?=-|$)", stem_upper):
        y = int(candidate)
        if 1000 <= y <= 1950:
            return y

    year = _year_from_period_suffix(stem)
    if year is not None:
        return year

    for candidate in re.findall(r"\d{4}", stem_upper):
        y = int(candidate)
        if 1000 <= y <= 1950:
            return y

    return None


def parse_psd_file(filepath: Path, year_map: dict[str, int]) -> list[tuple[str, int]]:
    """Parse one .psd file; return (sentence_text, year) pairs."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Cannot read %s: %s", filepath, e)
        return []

    results: list[tuple[str, int]] = []
    file_stem = filepath.stem.upper()

    for tree in _split_toplevel(text):
        words = _extract_words(tree)
        if not words or len(words.split()) < 4:
            continue

        id_stem = _extract_id_stem(tree)
        year    = None
        if id_stem:
            year = year_map.get(id_stem) or _year_from_stem(id_stem)
        if year is None:
            year = year_map.get(file_stem) or _year_from_stem(file_stem)
        if year is None:
            continue

        results.append((words, year))

    return results


def run(corpus_dir: Path, year_map_path: Path | None, output_path: Path, dry_run: bool) -> None:
    year_map: dict[str, int] = {}
    if year_map_path and year_map_path.exists():
        with open(year_map_path) as fh:
            year_map = {k.upper(): int(v) for k, v in json.load(fh).items()}
        logger.info("Loaded %d year-map entries", len(year_map))

    psd_files = sorted(corpus_dir.rglob("*.psd"))
    logger.info("Found %d .psd files in %s", len(psd_files), corpus_dir)
    if not psd_files:
        logger.error("No .psd files found — check --corpus-dir")
        sys.exit(1)

    all_rows: list[tuple[str, int]] = []
    t0 = time.time()

    for i, fp in enumerate(psd_files, 1):
        rows = parse_psd_file(fp, year_map)
        all_rows.extend(rows)
        if i % 20 == 0 or i == len(psd_files):
            logger.info("  Parsed %d / %d files — %d sentences (%.1fs)",
                        i, len(psd_files), len(all_rows), time.time() - t0)
        if dry_run and len(all_rows) >= 50:
            all_rows = all_rows[:50]
            logger.info("[dry-run] Truncated to 50 sentences")
            break

    if not all_rows:
        logger.error("No sentences extracted — check corpus format and year resolution")
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=["sentence", "year"])
    df = df[df["year"].between(1000, 1950)].reset_index(drop=True)
    logger.info("Extracted %d sentences, years %d–%d",
                len(df), df["year"].min(), df["year"].max())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved → %s", output_path)


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    paths = cfg["paths"]
    parser = argparse.ArgumentParser(
        description="Stage 00: Parse Penn Historical English corpora → parquet."
    )
    parser.add_argument("--corpus-dir", type=Path,
                        default=PROJECT_ROOT / paths["penn_corpus_dir"])
    parser.add_argument("--year-map",   type=Path,
                        default=PROJECT_ROOT / paths["penn_year_map"])
    parser.add_argument("--output",     type=Path,
                        default=PROJECT_ROOT / paths["penn_sentences"])
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--model", default="bert",
                        choices=list(cfg.get("models", {"bert": None}).keys()),
                        help="Model key (ignored by this script — corpus is shared).")
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    year_map_path = args.year_map if args.year_map.exists() else None
    run(args.corpus_dir, year_map_path, args.output, args.dry_run)


if __name__ == "__main__":
    main()
