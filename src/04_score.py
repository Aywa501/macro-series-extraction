"""Stage 04 — Distributional shift measurement.

For each (idiom, model) pair, computes how the distribution of contextual
embeddings shifts between consecutive time windows.  Two complementary measures
are computed:

APD (primary)
    Average Pairwise Distance — the mean cosine distance across ALL cross-window
    pairs (e_i from window t, e_j from window t+1).  Current state-of-the-art
    for diachronic semantic change (Hamilton et al. 2016, Giulianelli et al.
    2020, Periti & Tahmasebi 2022).

    apd = mean_{i∈t, j∈t+1}(1 − cos(e_i, e_j))

PRT / drift_cosine (robustness check)
    Prototype / centroid drift — cosine distance between the mean (centroid)
    embeddings of consecutive windows.  More stable with small n; kept as a
    robustness check.

Binning modes
-------------
fixed (default)
    Non-overlapping windows of ``--bin-width`` years (default 20).  Each
    observation falls in exactly one bin.  Roughly quadruples per-cell count
    vs 10-year decades at the cost of temporal resolution.

rolling
    Overlapping windows of ``--window-size`` years (default 20) stepped by
    ``--step`` years (default 10).  APD is computed between windows centred at
    t and t+step.  Produces more regression observations than non-overlapping
    bins; serial correlation is handled by clustered / HAC standard errors in
    stage 06.

Method
------
1.  Group observations by (idiom, bin).
2.  For each consecutive bin pair where both have ≥ MIN_OBS observations,
    compute APD and PRT.
3.  Output one row per (model, idiom, bin_start, bin_end).

References
----------
Hamilton et al. (2016) "Diachronic Word Embeddings Reveal Statistical Laws
    of Semantic Change." ACL.
Giulianelli et al. (2020) "Analysing Lexical Semantic Change with
    Contextualised Word Representations." ACL.
Periti & Tahmasebi (2022) "Grammatical Gender's Influence on Semantic Change."

Output
------
    data/processed/drift.parquet

Columns
-------
    model, idiom, denomination, group,
    decade_start, decade_end,  # bin start years (or rolling window centres)
    apd,                       # Average Pairwise Distance (primary)
    drift_cosine,              # centroid-to-centroid cosine distance (PRT)
    n_start, n_end,            # observations per window
    bin_mode,                  # "fixed" or "rolling"
    bin_width                  # window width in years
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

# Minimum embeddings per window to form usable centroid / pairwise distances.
MIN_OBS: int = 3

# --- Fixed-bin defaults ---
# Non-overlapping window width (years).  20-year bins ~4× the per-cell count
# of 10-year decades while still resolving major inflationary episodes.
BIN_WIDTH: int = 10
# Maximum allowed gap between consecutive valid bins before skipping the pair.
# Set to 2×BIN_WIDTH so one empty bin is permitted but two consecutive empty
# bins are not.
MAX_GAP: int = 20

# --- Rolling-window defaults ---
WINDOW_SIZE: int = 20   # total width of each observation window (years)
STEP: int = 10          # step between consecutive window centres (years)


# ---------------------------------------------------------------------------
# Core distance functions
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 − cosine_similarity for two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))


def _average_pairwise_distance(E_t: np.ndarray, E_t1: np.ndarray) -> float:
    """Average pairwise cosine distance across two sets of embeddings.

    Computes mean(1 − cos(e_i, e_j)) for all i ∈ E_t, j ∈ E_t1.

    Parameters
    ----------
    E_t:
        Float array of shape ``(n, D)`` — embeddings for decade t.
    E_t1:
        Float array of shape ``(m, D)`` — embeddings for decade t+1.

    Returns
    -------
    float
        Mean cosine distance, or np.nan if either set is empty or all-zero.
    """
    if E_t.shape[0] == 0 or E_t1.shape[0] == 0:
        return np.nan

    # L2-normalise both sets; rows with zero norm become zero vectors.
    norms_t = np.linalg.norm(E_t, axis=1, keepdims=True)
    norms_t1 = np.linalg.norm(E_t1, axis=1, keepdims=True)

    # Guard against zero-norm rows (map to zero vector — will produce NaN
    # similarities for those pairs, which we filter below).
    with np.errstate(invalid="ignore"):
        E_t_n = np.where(norms_t > 1e-9, E_t / norms_t, 0.0).astype(np.float32)
        E_t1_n = np.where(norms_t1 > 1e-9, E_t1 / norms_t1, 0.0).astype(np.float32)

    # Cosine similarity matrix: (n, m)
    cos_sim = E_t_n @ E_t1_n.T  # values in [-1, 1]
    cos_dist = 1.0 - cos_sim    # values in [0, 2]

    # Mask pairs that involved a zero-norm vector
    zero_t = (norms_t <= 1e-9).flatten()   # shape (n,)
    zero_t1 = (norms_t1 <= 1e-9).flatten() # shape (m,)
    bad_mask = zero_t[:, None] | zero_t1[None, :]  # (n, m)

    cos_dist_clean = cos_dist[~bad_mask]
    if cos_dist_clean.size == 0:
        return np.nan
    return float(cos_dist_clean.mean())


def _representative_denomination(denom_val) -> list:
    """Normalise the denomination value from a parquet cell to a Python list."""
    if isinstance(denom_val, (list, np.ndarray)):
        return list(denom_val)
    if isinstance(denom_val, str):
        return [denom_val]
    return []


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

def compute_drift(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    model_key: str,
    min_obs: int = MIN_OBS,
    max_gap: int = MAX_GAP,
    bin_width: int = BIN_WIDTH,
) -> list[dict]:
    """Compute consecutive fixed-bin APD and centroid drift for one model.

    Parameters
    ----------
    embeddings:
        Float32 array of shape ``(N, D)`` aligned with ``meta``.
    meta:
        DataFrame with columns: id, emb_idx, idiom, denomination, group, year.
    model_key:
        Short model name for the ``model`` column.
    min_obs:
        Minimum embeddings per bin to include that bin.
    max_gap:
        Maximum year gap between consecutive valid bins (skip wider pairs).
    bin_width:
        Non-overlapping window width in years (e.g. 20 → 1800, 1820, 1840 …).

    Returns
    -------
    list of dict
        One dict per consecutive bin-pair per idiom.
    """
    rows: list[dict] = []
    meta = meta.copy()
    meta["bin"] = (meta["year"] // bin_width) * bin_width

    for idiom, grp in meta.groupby("idiom"):
        first = grp.iloc[0]
        denomination = _representative_denomination(first["denomination"])
        group = str(first["group"])

        bin_embeddings: dict[int, np.ndarray] = {}
        bin_centroids:  dict[int, np.ndarray] = {}
        bin_counts:     dict[int, int] = {}

        for bin_val, b_grp in grp.groupby("bin"):
            idxs = b_grp["emb_idx"].tolist()
            if len(idxs) >= min_obs:
                E = embeddings[idxs]
                bin_embeddings[int(bin_val)] = E
                bin_centroids[int(bin_val)]  = E.mean(axis=0)
                bin_counts[int(bin_val)]     = len(idxs)

        valid_bins = sorted(bin_centroids)
        if len(valid_bins) < 2:
            logger.debug(
                "Idiom '%s': fewer than 2 usable bins (%d). Skipping.",
                idiom, len(valid_bins),
            )
            continue

        for i in range(len(valid_bins) - 1):
            b0, b1 = valid_bins[i], valid_bins[i + 1]
            if (b1 - b0) > max_gap:
                logger.debug(
                    "Idiom '%s': skipping pair (%d, %d) — gap %d > max_gap %d.",
                    idiom, b0, b1, b1 - b0, max_gap,
                )
                continue

            apd = _average_pairwise_distance(bin_embeddings[b0], bin_embeddings[b1])
            prt = _cosine_distance(bin_centroids[b0], bin_centroids[b1])

            if np.isnan(apd) and np.isnan(prt):
                continue

            rows.append({
                "model":        model_key,
                "idiom":        idiom,
                "denomination": denomination,
                "group":        group,
                "decade_start": b0,
                "decade_end":   b1,
                "apd":          apd,
                "drift_cosine": prt,
                "n_start":      bin_counts[b0],
                "n_end":        bin_counts[b1],
                "bin_mode":     "fixed",
                "bin_width":    bin_width,
            })

    return rows


def compute_drift_rolling(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    model_key: str,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
    min_obs: int = MIN_OBS,
) -> list[dict]:
    """Compute APD and PRT using overlapping rolling windows.

    For each pair of consecutive anchor years (t, t+step), collects all
    observations within a ``window_size``-year window centred on each anchor
    and computes APD and PRT between the two resulting sets.

    Parameters
    ----------
    embeddings:
        Float32 array of shape ``(N, D)`` aligned with ``meta``.
    meta:
        DataFrame with columns: id, emb_idx, idiom, denomination, group, year.
    model_key:
        Short model name for the ``model`` column.
    window_size:
        Total span of each observation window in years (centred on anchor).
    step:
        Step between consecutive anchor years.  ``decade_start`` values in the
        output are multiples of ``step``.
    min_obs:
        Minimum observations per window to include that window.

    Returns
    -------
    list of dict
        One dict per consecutive anchor-pair per idiom.  ``decade_start`` and
        ``decade_end`` hold the window centre years.
    """
    rows: list[dict] = []
    half = window_size // 2

    for idiom, grp in meta.groupby("idiom"):
        first = grp.iloc[0]
        denomination = _representative_denomination(first["denomination"])
        group = str(first["group"])

        years = grp["year"].values
        if len(years) < 2 * min_obs:
            continue

        min_anchor = (int(years.min()) // step) * step
        max_anchor = (int(years.max()) // step) * step

        win_embeddings: dict[int, np.ndarray] = {}
        win_counts:     dict[int, int] = {}

        for anchor in range(min_anchor, max_anchor + 1, step):
            mask = (grp["year"] >= anchor - half) & (grp["year"] < anchor + half)
            sub  = grp[mask]
            if len(sub) >= min_obs:
                idxs = sub["emb_idx"].tolist()
                E = embeddings[idxs]
                win_embeddings[anchor] = E
                win_counts[anchor]     = len(idxs)

        valid_anchors = sorted(win_embeddings)
        if len(valid_anchors) < 2:
            continue

        for i in range(len(valid_anchors) - 1):
            a0, a1 = valid_anchors[i], valid_anchors[i + 1]
            # Only truly consecutive anchors — any gap means a data hole.
            if a1 - a0 != step:
                logger.debug(
                    "Idiom '%s': rolling gap (%d → %d), skipping.", idiom, a0, a1,
                )
                continue

            apd = _average_pairwise_distance(win_embeddings[a0], win_embeddings[a1])
            prt = _cosine_distance(
                win_embeddings[a0].mean(axis=0), win_embeddings[a1].mean(axis=0)
            )

            if np.isnan(apd) and np.isnan(prt):
                continue

            rows.append({
                "model":        model_key,
                "idiom":        idiom,
                "denomination": denomination,
                "group":        group,
                "decade_start": a0,
                "decade_end":   a1,
                "apd":          apd,
                "drift_cosine": prt,
                "n_start":      win_counts[a0],
                "n_end":        win_counts[a1],
                "bin_mode":     "rolling",
                "bin_width":    window_size,
            })

    return rows


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_drift_timeseries(
    drift_df: pd.DataFrame,
    output_path: Path,
    model_key: str,
) -> None:
    """Line plots of APD and drift_cosine over decade_start for treatment idioms."""
    df = drift_df[
        (drift_df["model"] == model_key) & (drift_df["group"] == "treatment")
    ].copy()
    if df.empty:
        return

    idioms = sorted(df["idiom"].unique())
    n = len(idioms)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), sharex=False)
    axes = np.array(axes).flatten()

    for ax, idiom in zip(axes, idioms):
        sub = df[df["idiom"] == idiom].sort_values("decade_start")
        ax.plot(
            sub["decade_start"], sub["apd"],
            marker="o", linewidth=1.2, markersize=4, label="APD", color="#2166ac",
        )
        if "drift_cosine" in sub.columns and sub["drift_cosine"].notna().any():
            ax.plot(
                sub["decade_start"], sub["drift_cosine"],
                marker="s", linewidth=1.0, markersize=3, linestyle="--",
                label="PRT", color="#d6604d", alpha=0.7,
            )
        ax.set_title(idiom, fontsize=8, wrap=True)
        ax.set_xlabel("Decade", fontsize=7)
        ax.set_ylabel("Distance", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Consecutive-decade distributional drift — model: {model_key}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved drift time-series plot → %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_scoring(
    data_dir: Path,
    force: bool = False,
    min_obs: int = MIN_OBS,
    max_gap: int = MAX_GAP,
    bin_width: int = BIN_WIDTH,
    rolling: bool = False,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
) -> None:
    """Compute distributional drift for all models and write drift.parquet.

    Parameters
    ----------
    data_dir:
        Project data root.
    force:
        Overwrite existing output.
    min_obs:
        Minimum observations per window.
    max_gap:
        Maximum year gap between consecutive valid bins (fixed mode only).
    bin_width:
        Non-overlapping bin width in years (fixed mode).
    rolling:
        If True, use overlapping rolling windows instead of fixed bins.
    window_size:
        Rolling window total span in years.
    step:
        Rolling window step in years.
    """
    embeddings_dir = data_dir / "processed" / "embeddings"
    output_path = data_dir / "processed" / "drift.parquet"

    if output_path.exists() and not force:
        logger.info("drift.parquet already exists. Use --force to recompute.")
        return

    index_path = embeddings_dir / "index.parquet"
    observations_path = data_dir / "interim" / "observations.parquet"

    for p in (index_path, observations_path):
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    index_df = pd.read_parquet(index_path)
    obs_df = pd.read_parquet(observations_path)

    # Keep only confirmed idiomatic, non-review observations
    mask = obs_df["is_idiomatic"] == True  # noqa: E712
    if "needs_review" in obs_df.columns:
        mask &= ~obs_df["needs_review"].fillna(False)
    obs_df = obs_df[mask].reset_index(drop=True)
    logger.info("Using %d confirmed idiomatic observations.", len(obs_df))

    # Build embedding-index → metadata join
    id_to_emb_idx: dict[str, int] = {
        row_id: i for i, row_id in enumerate(index_df["id"])
    }

    meta = obs_df[["id", "idiom", "denomination", "group", "year"]].copy()
    meta = meta[meta["id"].isin(id_to_emb_idx)].copy()
    meta["emb_idx"] = meta["id"].map(id_to_emb_idx)
    meta["decade"] = (meta["year"] // 10) * 10

    npy_files = sorted(embeddings_dir.glob("*.npy"))
    if not npy_files:
        logger.error("No .npy embedding files in %s.", embeddings_dir)
        sys.exit(1)

    all_rows: list[dict] = []

    for npy_path in npy_files:
        model_key = npy_path.stem
        logger.info("Computing drift for model: %s …", model_key)
        embeddings = np.load(npy_path)
        logger.info("  Loaded embeddings: shape=%s", embeddings.shape)

        # Only keep IDs that exist in this model's embedding matrix
        valid_meta = meta[meta["emb_idx"] < len(embeddings)].copy()

        if rolling:
            rows = compute_drift_rolling(
                embeddings, valid_meta, model_key,
                window_size=window_size, step=step, min_obs=min_obs,
            )
        else:
            rows = compute_drift(
                embeddings, valid_meta, model_key,
                min_obs=min_obs, max_gap=max_gap, bin_width=bin_width,
            )
        logger.info("  %d decade-pair observations (model=%s).", len(rows), model_key)
        all_rows.extend(rows)

        # Per-model diagnostic plot
        if rows:
            tmp_df = pd.DataFrame(rows)
            plot_path = (
                PROJECT_ROOT / "outputs" / "figures" / f"drift_timeseries_{model_key}.png"
            )
            plot_drift_timeseries(tmp_df, plot_path, model_key=model_key)

    if not all_rows:
        logger.error(
            "No drift rows produced. Check MIN_OBS (%d) vs observation counts.", min_obs
        )
        sys.exit(1)

    drift_df = pd.DataFrame(all_rows)
    drift_df.to_parquet(output_path, index=False)
    logger.info("Wrote %d drift rows to %s.", len(drift_df), output_path)

    # Summary
    summary = (
        drift_df.groupby(["model", "group"])
        .agg(
            n_pairs=("apd", "count"),
            mean_apd=("apd", "mean"),
            std_apd=("apd", "std"),
            mean_prt=("drift_cosine", "mean"),
        )
        .round(5)
    )
    logger.info("\n%s", summary.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 04: Measure consecutive-decade distributional drift (APD + PRT)."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--min-obs", type=int, default=MIN_OBS,
        help=f"Min embeddings per window (default: {MIN_OBS}).",
    )
    # Fixed-bin options
    parser.add_argument(
        "--bin-width", type=int, default=BIN_WIDTH,
        help=f"Non-overlapping bin width in years (default: {BIN_WIDTH}).",
    )
    parser.add_argument(
        "--max-gap", type=int, default=MAX_GAP,
        help=(
            f"Max year gap between consecutive valid bins, fixed mode only "
            f"(default: {MAX_GAP})."
        ),
    )
    # Rolling-window options
    parser.add_argument(
        "--rolling", action="store_true",
        help="Use overlapping rolling windows instead of fixed bins.",
    )
    parser.add_argument(
        "--window-size", type=int, default=WINDOW_SIZE,
        help=f"Rolling window total span in years (default: {WINDOW_SIZE}).",
    )
    parser.add_argument(
        "--step", type=int, default=STEP,
        help=f"Rolling window step in years (default: {STEP}).",
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
    run_scoring(
        data_dir=args.data_dir,
        force=args.force,
        min_obs=args.min_obs,
        max_gap=args.max_gap,
        bin_width=args.bin_width,
        rolling=args.rolling,
        window_size=args.window_size,
        step=args.step,
    )


if __name__ == "__main__":
    main()
