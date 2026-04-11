"""Stage 05 — Merge drift rates with real denomination values.

Loads the decade-pair drift table from stage 04 and the BOE price index,
computes Δlog(RV) = log_RV(decade_end) − log_RV(decade_start) for each
denomination, and merges into a single analysis table.

For idioms with multiple denominations (e.g. "pennies in the pound"), the
Δlog(RV) values for each denomination are averaged so the regression in
stage 06 sees exactly one Δlog(RV) per (idiom, decade_pair) observation.

Output
------
    data/processed/drift_index.parquet

Columns
-------
    model, idiom, denomination (list), group,
    decade_start, decade_end,
    drift_cosine, n_start, n_end,
    log_rv_start, log_rv_end,   ← mean log_RV within each decade window
    delta_log_rv                ← log_rv_end − log_rv_start

Usage
-----
    python src/05_index.py
    python src/05_index.py --data-dir /path/to/project --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Nominal values in old pence (pre-decimalisation)
NOMINAL_VALUES: dict[str, float] = {
    "penny":    1.0,
    "pennies":  1.0,
    "farthing": 0.25,
    "shilling": 12.0,
    "pound":    240.0,
    "guinea":   252.0,
}

NORMALISE_YEAR = 1800


# ---------------------------------------------------------------------------
# Price index
# ---------------------------------------------------------------------------

def load_price_index(csv_path: Path) -> pd.DataFrame:
    """Load price_index.csv and normalise to 1800 = 100.

    Parameters
    ----------
    csv_path:
        Path to CSV with columns ``year`` and ``price_index``.

    Returns
    -------
    pd.DataFrame
        Columns: year (int), price_index (float).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "year" not in df.columns or "price_index" not in df.columns:
        raise ValueError(
            f"price_index.csv must have 'year' and 'price_index'. Got: {list(df.columns)}"
        )
    df["year"] = df["year"].astype(int)
    df["price_index"] = df["price_index"].astype(float)

    base_rows = df[df["year"] == NORMALISE_YEAR]
    base_value = (
        base_rows.iloc[0]["price_index"]
        if not base_rows.empty
        else df.sort_values("year").iloc[0]["price_index"]
    )
    df["price_index"] = df["price_index"] / base_value * 100.0
    return df[["year", "price_index"]]


def _compute_log_rv(denomination: str, price_index: float) -> float:
    """log(nominal / price_level), where price_level = price_index / 100."""
    nominal = NOMINAL_VALUES.get(denomination.lower(), np.nan)
    if np.isnan(nominal) or price_index <= 0:
        return np.nan
    rv = nominal / (price_index / 100.0)
    return float(np.log(rv)) if rv > 0 else np.nan


def build_decade_log_rv(
    price_df: pd.DataFrame,
    bin_width: int = 20,
    step: int = 10,
) -> pd.DataFrame:
    """Compute mean log_RV for each anchor window for every denomination.

    Produces a lookup table keyed at every ``step``-year anchor from the price
    series' range.  For each anchor ``a``, the price average covers
    ``[a, a + bin_width)``.

    This covers both binning modes from stage 04:
    - Fixed 20-year bins: ``decade_start`` ∈ {1800, 1820, …} — all multiples of
      20 are also multiples of 10, so they appear in the lookup.
    - Rolling windows (step=10): ``decade_start`` ∈ {1800, 1810, …} — every
      anchor is directly in the lookup.

    Parameters
    ----------
    price_df:
        Price index table (year, price_index).
    bin_width:
        Width of the averaging window in years (should match stage 04's
        ``--bin-width`` or ``--window-size``).
    step:
        Granularity of anchor keys.  Default 10 so both fixed-20 and rolling-10
        stage-04 outputs are covered without a second call.

    Returns
    -------
    pd.DataFrame
        Columns: denomination, decade (anchor year), log_rv_decade.
    """
    min_year = int(price_df["year"].min())
    max_year = int(price_df["year"].max())
    anchor_start = (min_year // step) * step

    records = []
    for denom in NOMINAL_VALUES:
        for anchor in range(anchor_start, max_year + 1, step):
            window = price_df[
                (price_df["year"] >= anchor) & (price_df["year"] < anchor + bin_width)
            ]
            log_rvs = window["price_index"].apply(
                lambda pi: _compute_log_rv(denom, pi)  # noqa: B023
            ).dropna()
            if len(log_rvs) == 0:
                continue
            records.append({
                "denomination":  denom,
                "decade":        anchor,
                "log_rv_decade": float(log_rvs.mean()),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Merge drift with Δlog(RV)
# ---------------------------------------------------------------------------

def _normalise_denomination_list(val) -> list[str]:
    """Return a plain Python list of denomination strings from a parquet cell."""
    if isinstance(val, (list, np.ndarray)):
        return [str(d).lower() for d in val]
    if isinstance(val, str):
        return [val.lower()]
    return []


def build_drift_index(
    drift_df: pd.DataFrame,
    decade_rv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge drift rates with Δlog(RV) per decade pair.

    For multi-denomination idioms, Δlog(RV) is the average across constituent
    denominations.  Placebo rows (denomination contains only 'placebo' or
    'none') receive NaN Δlog(RV) and are retained for diagnostics but
    excluded from the main regression in stage 06.

    Parameters
    ----------
    drift_df:
        Output of stage 04.
    decade_rv_df:
        Output of :func:`build_decade_log_rv`.

    Returns
    -------
    pd.DataFrame
        One row per (model, idiom, decade_pair); includes delta_log_rv.
    """
    rv_lookup: dict[tuple[str, int], float] = {
        (row["denomination"], int(row["decade"])): row["log_rv_decade"]
        for _, row in decade_rv_df.iterrows()
    }

    results = []
    for _, row in drift_df.iterrows():
        denoms = _normalise_denomination_list(row["denomination"])
        # Filter out sentinel labels used for placebo / excluded rows
        monetary_denoms = [
            d for d in denoms
            if d in NOMINAL_VALUES
        ]

        log_rv_starts = [
            rv_lookup.get((d, int(row["decade_start"])), np.nan)
            for d in monetary_denoms
        ]
        log_rv_ends = [
            rv_lookup.get((d, int(row["decade_end"])), np.nan)
            for d in monetary_denoms
        ]

        if monetary_denoms:
            log_rv_start = float(np.nanmean(log_rv_starts)) if log_rv_starts else np.nan
            log_rv_end = float(np.nanmean(log_rv_ends)) if log_rv_ends else np.nan
        else:
            log_rv_start = log_rv_end = np.nan

        delta_log_rv = (
            log_rv_end - log_rv_start
            if not (np.isnan(log_rv_start) or np.isnan(log_rv_end))
            else np.nan
        )

        out = dict(row)
        out["log_rv_start"] = log_rv_start
        out["log_rv_end"] = log_rv_end
        out["delta_log_rv"] = delta_log_rv
        results.append(out)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_drift_vs_delta_rv(
    index_df: pd.DataFrame,
    output_path: Path,
    model_key: str,
) -> None:
    """Scatter of drift_cosine vs Δlog(RV), treatment coloured by denomination.

    Parameters
    ----------
    index_df:
        Full drift index table.
    output_path:
        PNG output path.
    model_key:
        Which model's data to plot.
    """
    df = index_df[index_df["model"] == model_key].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Treatment panel ---
    treat = df[df["group"] == "treatment"].dropna(subset=["delta_log_rv", "drift_cosine"])
    denom_colors = {
        "penny": "#2166ac", "farthing": "#d6604d",
        "shilling": "#1a9850", "pound": "#7b2d8b",
    }
    plotted_denoms: set[str] = set()

    for _, row in treat.iterrows():
        denoms = _normalise_denomination_list(row["denomination"])
        monetary = [d for d in denoms if d in NOMINAL_VALUES]
        label_denom = monetary[0] if monetary else "other"
        color = denom_colors.get(label_denom, "grey")
        lbl = label_denom if label_denom not in plotted_denoms else None
        axes[0].scatter(
            row["delta_log_rv"], row["drift_cosine"],
            color=color, alpha=0.6, s=30, label=lbl,
        )
        plotted_denoms.add(label_denom)

    # OLS fit line for treatment
    xy = treat[["delta_log_rv", "drift_cosine"]].dropna()
    if len(xy) > 2:
        m, b = np.polyfit(xy["delta_log_rv"], xy["drift_cosine"], 1)
        x_range = np.linspace(xy["delta_log_rv"].min(), xy["delta_log_rv"].max(), 100)
        axes[0].plot(x_range, m * x_range + b, "k--", linewidth=1, label=f"OLS (β={m:.3f})")

    axes[0].axhline(0, color="grey", linewidth=0.6, linestyle=":")
    axes[0].axvline(0, color="grey", linewidth=0.6, linestyle=":")
    axes[0].set_xlabel("Δlog(RV) [decade change in log real value]")
    axes[0].set_ylabel("Drift (1 − cosine similarity of decade centroids)")
    axes[0].set_title(f"Treatment idioms — model: {model_key}")
    axes[0].legend(fontsize=8)

    # --- Placebo panel ---
    placebo = df[df["group"] == "placebo"].dropna(subset=["drift_cosine"])
    axes[1].scatter(
        range(len(placebo)), placebo["drift_cosine"].sort_values(),
        color="grey", alpha=0.5, s=20,
    )
    axes[1].axhline(
        treat["drift_cosine"].mean() if not treat.empty else 0,
        color="#2166ac", linewidth=1, linestyle="--", label="Treatment mean",
    )
    axes[1].set_xlabel("Placebo observations (sorted by drift)")
    axes[1].set_ylabel("Drift")
    axes[1].set_title("Placebo idioms — drift distribution")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved drift-vs-ΔRV plot → %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_index(
    data_dir: Path,
    force: bool = False,
    bin_width: int = 20,
    step: int = 10,
) -> None:
    """Build drift_index.parquet and diagnostic plots.

    Parameters
    ----------
    data_dir:
        Project data root.
    force:
        Overwrite existing outputs.
    bin_width:
        Averaging window width passed to :func:`build_decade_log_rv`.  Should
        match stage 04's ``--bin-width`` (fixed) or ``--window-size`` (rolling).
    step:
        Anchor granularity passed to :func:`build_decade_log_rv`.  Default 10
        covers both fixed-20 and rolling-10 outputs from stage 04.
    """
    drift_path = data_dir / "processed" / "drift.parquet"
    # price_index.csv is shared reference data — look in data_dir/raw first,
    # then fall back to the project-level data/raw so sub-corpus dirs work.
    _local_price = data_dir / "raw" / "price_index.csv"
    _root_price = Path(__file__).parent.parent / "data" / "raw" / "price_index.csv"
    price_path = _local_price if _local_price.exists() else _root_price
    output_path = data_dir / "processed" / "drift_index.parquet"

    if output_path.exists() and not force:
        logger.info("drift_index.parquet already exists. Use --force to recompute.")
        return

    if not drift_path.exists():
        logger.error("drift.parquet not found: %s. Run stage 04 first.", drift_path)
        sys.exit(1)

    drift_df = pd.read_parquet(drift_path)
    logger.info("Loaded %d drift rows.", len(drift_df))

    if price_path.exists():
        price_df = load_price_index(price_path)
        logger.info("Loaded price index: %d years.", len(price_df))
    else:
        logger.warning(
            "price_index.csv not found. delta_log_rv will be NaN for all rows."
        )
        price_df = pd.DataFrame({"year": [], "price_index": []})

    decade_rv_df = build_decade_log_rv(price_df, bin_width=bin_width, step=step)
    index_df = build_drift_index(drift_df, decade_rv_df)

    index_df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s.", len(index_df), output_path)

    # Summary statistics
    treat = index_df[index_df["group"] == "treatment"]
    placebo = index_df[index_df["group"] == "placebo"]
    logger.info(
        "Treatment: %d decade-pairs | mean drift=%.4f | "
        "mean Δlog(RV)=%.4f | NaN Δlog(RV)=%d",
        len(treat),
        treat["drift_cosine"].mean() if not treat.empty else np.nan,
        treat["delta_log_rv"].mean() if not treat.empty else np.nan,
        treat["delta_log_rv"].isna().sum(),
    )
    logger.info(
        "Placebo: %d decade-pairs | mean drift=%.4f",
        len(placebo),
        placebo["drift_cosine"].mean() if not placebo.empty else np.nan,
    )

    # Plot for each model
    for model_key in index_df["model"].unique():
        plot_path = (
            PROJECT_ROOT / "outputs" / "figures" / f"drift_index_{model_key}.png"
        )
        plot_drift_vs_delta_rv(index_df, plot_path, model_key=str(model_key))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 05: Merge drift rates with denomination real values."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--bin-width", type=int, default=20,
        help=(
            "Averaging window width for log_RV lookup (years). "
            "Should match stage 04's --bin-width or --window-size (default: 20)."
        ),
    )
    parser.add_argument(
        "--step", type=int, default=10,
        help=(
            "Anchor granularity for log_RV lookup (years). "
            "Default 10 covers both fixed-20 and rolling-10 stage-04 outputs."
        ),
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
    run_index(
        data_dir=args.data_dir,
        force=args.force,
        bin_width=args.bin_width,
        step=args.step,
    )


if __name__ == "__main__":
    main()
