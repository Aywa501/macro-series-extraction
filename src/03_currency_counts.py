"""Stage 03 — Currency occurrence counts in Penn corpus.

Scans penn_sentences.parquet (Stage 00 output) and counts how often each
currency term (penny/pennies, farthing/farthings, shilling/shillings,
pound/pounds) appears per 25-year time bin.  From the resulting frequency
curves it derives a temporal window — [first_year, dropout_year] — for each
currency.  Stage 04 uses these windows to set the λ range for interventions
so that λ_min and λ_max correspond to historically meaningful boundaries.

Dropout year
------------
The dropout year is the first bin after peak frequency where the count drops
below `currency_dropout_threshold × peak_count` (default 10 %).  This marks
the point at which the currency term has effectively fallen out of regular use
in the corpus.

Outputs
-------
data/corpus/currency_counts.parquet
    columns: currency, bin_start, count, count_per_sentence

data/corpus/currency_windows.csv
    columns: currency, first_year, peak_year, peak_count,
             dropout_year, lambda_min, lambda_max

Usage
-----
    python src/03_currency_counts.py
    python src/03_currency_counts.py --dry-run
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)

BIN_SIZE = 25   # years per time bin — matches DECADE_SIZE in 01c


def _count_terms(sentences: pd.Series, forms: list[str]) -> pd.Series:
    """Return a boolean Series: True if any form appears (whole-word) in sentence."""
    pattern = r"\b(" + "|".join(re.escape(f) for f in forms) + r")\b"
    return sentences.str.contains(pattern, case=False, regex=True, na=False)


def _derive_window(
    counts: pd.Series,        # indexed by bin_start year, values = raw counts
    threshold: float,          # fraction of peak
) -> dict:
    """Return first_year, peak_year, peak_count, dropout_year."""
    nonzero = counts[counts > 0]
    if nonzero.empty:
        return {"first_year": None, "peak_year": None,
                "peak_count": 0,   "dropout_year": None}

    first_year = int(nonzero.index.min())
    peak_year  = int(counts.idxmax())
    peak_count = int(counts.max())
    cutoff     = threshold * peak_count

    # Look for dropout only after the peak
    post_peak = counts.loc[peak_year:]
    below = post_peak[post_peak < cutoff]
    dropout_year = int(below.index.min()) if not below.empty else int(counts.index.max())

    return {
        "first_year":   first_year,
        "peak_year":    peak_year,
        "peak_count":   peak_count,
        "dropout_year": dropout_year,
    }


def run(cfg: dict, dry_run: bool) -> None:
    paths     = cfg["paths"]
    terms_cfg = cfg["currency_terms"]         # list of {name, forms}
    threshold = cfg["currency_dropout_threshold"]

    sent_path    = PROJECT_ROOT / paths["penn_sentences"]
    counts_path  = PROJECT_ROOT / paths["currency_counts"]
    windows_path = PROJECT_ROOT / paths["currency_windows"]
    plots_dir    = PROJECT_ROOT / paths["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load sentences ---
    logger.info("Loading %s …", sent_path)
    df = pd.read_parquet(sent_path)
    if dry_run:
        df = df.sample(min(5000, len(df)), random_state=42)
        logger.info("[dry-run] Subsampled to %d sentences", len(df))
    else:
        logger.info("Loaded %d sentences (years %d–%d)",
                    len(df), df["year"].min(), df["year"].max())

    # Assign time bins
    df["bin_start"] = (df["year"] // BIN_SIZE).astype(int) * BIN_SIZE
    bins = np.arange(df["bin_start"].min(), df["bin_start"].max() + BIN_SIZE, BIN_SIZE)
    sentences_per_bin = df.groupby("bin_start").size().reindex(bins, fill_value=0)

    # --- Count occurrences per currency per bin ---
    count_rows: list[dict] = []
    window_rows: list[dict] = []

    fig, ax = plt.subplots(figsize=(12, 5))

    for term in terms_cfg:
        name  = term["name"]
        forms = term["forms"]

        df[f"_match_{name}"] = _count_terms(df["sentence"], forms)
        hits_per_bin = (
            df[df[f"_match_{name}"]].groupby("bin_start").size()
            .reindex(bins, fill_value=0)
        )
        rate_per_bin = hits_per_bin / sentences_per_bin.replace(0, np.nan)

        for bin_start in bins:
            count_rows.append({
                "currency":           name,
                "bin_start":          int(bin_start),
                "count":              int(hits_per_bin.get(bin_start, 0)),
                "count_per_sentence": float(rate_per_bin.get(bin_start, np.nan)),
            })

        win = _derive_window(hits_per_bin, threshold)
        lambda_min = win["first_year"]   if win["first_year"]   is not None else int(bins.min())
        lambda_max = win["dropout_year"] if win["dropout_year"] is not None else int(bins.max())

        window_rows.append({
            "currency":    name,
            "first_year":  win["first_year"],
            "peak_year":   win["peak_year"],
            "peak_count":  win["peak_count"],
            "dropout_year": win["dropout_year"],
            "lambda_min":  lambda_min,
            "lambda_max":  lambda_max,
        })

        logger.info(
            "  %-10s  first=%s  peak=%s(%d)  dropout=%s  "
            "→ λ window [%s, %s]",
            name,
            win["first_year"], win["peak_year"], win["peak_count"],
            win["dropout_year"], lambda_min, lambda_max,
        )

        ax.plot(bins, rate_per_bin.values, label=name, linewidth=1.8)

    ax.set_xlabel("Year (bin start)", fontsize=12)
    ax.set_ylabel("Occurrences per sentence", fontsize=12)
    ax.set_title("Currency term frequency in Penn corpus (25-year bins)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plot_path = plots_dir / "currency_counts.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot → %s", plot_path)

    # --- Save outputs ---
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts_df  = pd.DataFrame(count_rows)
    windows_df = pd.DataFrame(window_rows)

    counts_df.to_parquet(counts_path, index=False)
    windows_df.to_csv(windows_path, index=False)

    logger.info("Saved counts → %s", counts_path)
    logger.info("Saved windows → %s", windows_path)

    print("\n=== Currency temporal windows ===\n")
    print(windows_df.to_string(index=False))
    print(f"\nDropout threshold: {threshold * 100:.0f}% of peak count")
    print(f"Plot → {plot_path}")


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    parser = argparse.ArgumentParser(
        description="Stage 03: Count currency term occurrences in Penn corpus."
    )
    parser.add_argument("--model", default="bert",
                        choices=list(cfg.get("models", {"bert": None}).keys()),
                        help="Model key (ignored by this script — corpus counts are shared).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Subsample to 5000 sentences for speed.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(cfg, args.dry_run)


if __name__ == "__main__":
    main()
