"""Stage 01c — Resolve imprecise Gutenberg publication dates via Perplexity Sonar.

Books whose year was assigned from a century/era phrase (e.g. "20th century fiction"
→ 1950, "Victorian" → 1870) are identified and their exact first-publication year is
looked up via Perplexity Sonar using the book title and author from pg_catalog.csv.gz.

Results are cached locally so re-runs only query new books.
The gutenberg_candidates.parquet is patched in-place; date_method is updated to
"sonar_lookup" for resolved rows.

Usage
-----
    python src/01c_resolve_dates.py
    python src/01c_resolve_dates.py --dry-run    # print queries, don't call API
    python src/01c_resolve_dates.py --force      # re-query all, ignoring cache

Requirements
------------
    pip install openai
    export PERPLEXITY_API_KEY=pplx-...
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Years produced by century/era phrase matching in 01b — these are imprecise
# and worth replacing with exact publication dates.
CENTURY_MIDPOINTS: frozenset[int] = frozenset({
    1550, 1650, 1680, 1750, 1820, 1850, 1870, 1906, 1950,
})

CACHE_PATH   = PROJECT_ROOT / "data" / "interim" / ".sonar_date_cache.json"
PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "gutenberg_candidates.parquet"
CATALOG_PATH = PROJECT_ROOT / "data" / "raw" / "pg_catalog.csv.gz"

SONAR_MODEL      = "sonar"
RATE_LIMIT_PAUSE = 30   # seconds to pause when a 429 is received
MAX_RETRIES      = 6    # per-request retry attempts on rate limit
CACHE_SAVE_EVERY = 10   # persist cache after every N completions
YEAR_MIN, YEAR_MAX = 1400, 2010


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def _load_catalog() -> pd.DataFrame:
    with gzip.open(CATALOG_PATH, "rt", encoding="utf-8", errors="replace") as fh:
        df = pd.read_csv(fh, dtype=str)
    df = df.rename(columns={"Text#": "text_id"})
    df["text_id"] = df["text_id"].str.strip()
    return df


def _clean_title(title: str) -> str:
    """Strip subtitle noise and newlines; keep first 80 chars."""
    title = re.sub(r"\s+", " ", title).strip()
    # Drop everything after the first semicolon or newline-replacement
    title = title.split(";")[0].strip()
    return title[:120]


def _clean_authors(authors: str) -> str:
    """Keep only the primary author name, strip dates and parentheticals."""
    # "Wells, H. G. (Herbert George), 1866-1946" → "H. G. Wells"
    primary = authors.split(";")[0].strip()
    # Remove date ranges
    primary = re.sub(r",?\s*\d{4}\??\s*-\s*\d{4}", "", primary)
    # Remove parenthetical expansions like "(Herbert George)"
    primary = re.sub(r"\([^)]*\)", "", primary)
    # Remove suffixes like "Jr.", "Sr.", "II"
    primary = re.sub(r",?\s*(Jr\.|Sr\.|II+|IV|VI*)", "", primary)
    primary = primary.strip().rstrip(",").strip()
    # "Surname, Firstname" → "Firstname Surname"
    parts = [p.strip() for p in primary.split(",", 1)]
    if len(parts) == 2 and parts[1]:
        return f"{parts[1]} {parts[0]}"
    return parts[0]


# ---------------------------------------------------------------------------
# Sonar query
# ---------------------------------------------------------------------------

def _query_sonar(client, title: str, author: str) -> str | None:
    """Ask Sonar for the first publication year; return raw response text.

    Retries on 429 with an escalating pause so all threads back off together
    when the rate limit is hit.
    """
    from openai import RateLimitError

    prompt = (
        f"What year was '{title}' by {author} first published? "
        "Reply with only the 4-digit year and nothing else. "
        "If genuinely unknown, reply 'unknown'."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise bibliographic assistant. "
                "Reply with only a 4-digit year or the word 'unknown'."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=SONAR_MODEL,
                messages=messages,
                max_tokens=16,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            pause = RATE_LIMIT_PAUSE * (attempt + 1)
            logger.warning("Rate limited (attempt %d/%d) — pausing %ds.", attempt + 1, MAX_RETRIES, pause)
            time.sleep(pause)
        except Exception as exc:
            logger.warning("Sonar error for '%s': %s", title, exc)
            return None

    logger.warning("Gave up on '%s' after %d rate-limit retries.", title, MAX_RETRIES)
    return None


def _parse_year(text: str | None) -> int | None:
    """Extract a plausible 4-digit year from *text*."""
    if not text:
        return None
    m = re.search(r"\b(1[4-9]\d{2}|200\d|201\d)\b", text)
    if m:
        y = int(m.group(1))
        if YEAR_MIN <= y <= YEAR_MAX:
            return y
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False, force: bool = False) -> None:
    # Load parquet
    if not PARQUET_PATH.exists():
        logger.error("Parquet not found: %s — run stage 01b first.", PARQUET_PATH)
        sys.exit(1)
    df = pd.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows from %s", len(df), PARQUET_PATH)

    # Identify century-midpoint rows
    affected_mask = df["year"].isin(CENTURY_MIDPOINTS)
    n_affected = affected_mask.sum()
    logger.info("Century-midpoint rows to resolve: %d", n_affected)
    if n_affected == 0:
        logger.info("Nothing to do.")
        return

    # Build text_id → (title, author) mapping from catalog
    cat = _load_catalog()
    affected_ids = (
        df.loc[affected_mask, "speech_id"]
        .str.removeprefix("gutenberg_")
        .unique()
    )
    cat_sub = cat[cat["text_id"].isin(affected_ids)].set_index("text_id")
    logger.info("Catalog matches: %d / %d books", len(cat_sub), len(affected_ids))

    # Load cache
    cache: dict[str, int | None] = {}
    if CACHE_PATH.exists() and not force:
        cache = json.loads(CACHE_PATH.read_text())
        logger.info("Loaded %d cached results.", len(cache))

    # Build list of (text_id, title, author) to query
    to_query = [
        (tid, cat_sub.loc[tid, "Title"], cat_sub.loc[tid, "Authors"])
        for tid in affected_ids
        if tid in cat_sub.index and (force or tid not in cache)
    ]
    logger.info("Books to query: %d  (already cached: %d)", len(to_query), len(affected_ids) - len(to_query))

    if dry_run:
        print(f"\n--- DRY RUN: would query {len(to_query)} books ---")
        for tid, title, authors in to_query[:20]:
            print(f"  {tid}  '{_clean_title(title)}'  by {_clean_authors(str(authors))}")
        if len(to_query) > 20:
            print(f"  ... and {len(to_query) - 20} more")
        return

    if to_query:
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package required: pip install openai")
            sys.exit(1)

        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            logger.error("PERPLEXITY_API_KEY environment variable not set.")
            sys.exit(1)

        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

        n_resolved = 0

        def _save_cache() -> None:
            CACHE_PATH.write_text(json.dumps(cache, indent=2))

        pbar = tqdm(to_query, desc="Sonar lookups", unit="book")
        for i, (tid, raw_title, raw_authors) in enumerate(pbar):
            title  = _clean_title(str(raw_title))
            author = _clean_authors(str(raw_authors))
            raw_resp = _query_sonar(client, title, author)
            year = _parse_year(raw_resp)
            cache[tid] = year
            if year:
                n_resolved += 1
                logger.debug("  %s → %d", tid, year)
            else:
                logger.debug("  %s → unresolved (response: %r)", tid, raw_resp)
            if (i + 1) % CACHE_SAVE_EVERY == 0:
                _save_cache()
            pbar.set_postfix(resolved=n_resolved, unresolved=(i + 1) - n_resolved)

        _save_cache()
        logger.info("Sonar resolved %d / %d queried books.", n_resolved, len(to_query))

    # Apply updates to parquet
    updated = 0
    text_id_col = df["speech_id"].str.removeprefix("gutenberg_")
    for tid, year in cache.items():
        if year is None:
            continue
        row_mask = affected_mask & (text_id_col == tid)
        if row_mask.sum() == 0:
            continue
        df.loc[row_mask, "year"]        = year
        df.loc[row_mask, "date_method"] = "sonar_lookup"
        updated += row_mask.sum()

    logger.info("Patched %d rows with Sonar-resolved years.", updated)
    remaining = df["year"].isin(CENTURY_MIDPOINTS).sum()
    logger.info("Remaining century-midpoint rows (unresolved): %d", remaining)

    df.to_parquet(PARQUET_PATH, index=False)
    logger.info("Saved patched parquet → %s", PARQUET_PATH)

    # Summary
    print("\n=== Date method breakdown after patching ===")
    print(df["date_method"].value_counts().to_string())
    print("\n=== Year distribution (patched rows) ===")
    patched = df[df["date_method"] == "sonar_lookup"]
    if not patched.empty:
        print(((patched["year"] // 10) * 10).value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 01c: Resolve century-midpoint Gutenberg dates via Perplexity Sonar."
    )
    p.add_argument("--dry-run", action="store_true", help="Print queries without calling API.")
    p.add_argument("--force",   action="store_true", help="Re-query all books, ignoring cache.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
