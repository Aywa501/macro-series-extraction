"""Stage 05 — Analyse intervention results.

Loads the unified intervention results from Stage 04 and produces:

  - Per-idiom output-space intervention slopes
  - Layer sensitivity matrix from activation-space interventions
  - Output intervention plot with:
      · robustness check event markers (vertical lines, colour-coded by severity)
      · pre-active-era shading (grey) for each idiom's currency introduction
      · first-attestation tick marks per idiom
  - Layer sensitivity heatmap

Outputs
-------
data/intervention/intervention_slopes.csv
data/results/layer_sensitivity.npy
data/results/layer_sensitivity_index.csv
data/plots/output_intervention.png
data/plots/layer_heatmap.png

Usage
-----
    python src/05_analyse.py
    python src/05_analyse.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)

# Severity → line style for robustness check markers
_SEVERITY_STYLE = {
    "high":   {"color": "#d62728", "linestyle": "--",  "alpha": 0.55, "linewidth": 1.2},
    "medium": {"color": "#ff7f0e", "linestyle": "-.",  "alpha": 0.45, "linewidth": 0.9},
    "low":    {"color": "#aec7e8", "linestyle": ":",   "alpha": 0.40, "linewidth": 0.8},
}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, slope_se) from simple OLS of y ~ x."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    reg = LinearRegression(fit_intercept=True).fit(x.reshape(-1, 1), y)
    slope = float(reg.coef_[0])
    residuals = y - reg.predict(x.reshape(-1, 1)).ravel()
    var_res = np.sum(residuals ** 2) / (n - 2)
    var_x   = np.sum((x - x.mean()) ** 2)
    se = float(np.sqrt(var_res / var_x)) if var_x > 0 else float("nan")
    return slope, se


def _plot_robustness_markers(
    ax: plt.Axes,
    checks: list[dict],
    x_min: float,
    x_max: float,
    y_top: float,
) -> None:
    """Draw vertical lines + rotated labels for robustness check events."""
    label_y  = y_top
    label_step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04

    # Track label x-positions to stagger overlapping labels
    placed: list[float] = []

    for chk in checks:
        year = chk["year"]
        if not (x_min <= year <= x_max):
            continue
        style = _SEVERITY_STYLE.get(chk.get("severity", "low"), _SEVERITY_STYLE["low"])
        ax.axvline(year, **style)

        # Stagger label vertically if too close to previous
        lbl_y = y_top
        for px in placed:
            if abs(year - px) < (x_max - x_min) * 0.04:
                lbl_y -= label_step * 2
                break
        placed.append(year)

        ax.text(
            year + (x_max - x_min) * 0.004, lbl_y,
            chk["event"],
            rotation=90, va="top", ha="left",
            fontsize=5.5, color=style["color"], alpha=0.85,
        )


def run(cfg: dict, dry_run: bool) -> None:
    paths    = cfg["paths"]
    n_layers = cfg["model"]["n_layers"]
    checks   = cfg.get("robustness_checks", [])

    results_path = PROJECT_ROOT / paths["intervention_results"]
    slopes_path  = PROJECT_ROOT / paths["intervention_slopes"]
    mat_path     = PROJECT_ROOT / paths["layer_sensitivity_npy"]
    mat_idx_path = PROJECT_ROOT / paths["layer_sensitivity_idx"]
    plots_dir    = PROJECT_ROOT / paths["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    logger.info("Loading %s", results_path)
    df = pd.read_parquet(results_path)

    out_df = df[df["intervention_type"] == "output"].copy()
    act_df = df[df["intervention_type"] == "activation"].copy()

    if dry_run:
        idioms_sample = out_df["idiom"].unique()[:3]
        out_df = out_df[out_df["idiom"].isin(idioms_sample)]
        act_df = act_df[act_df["idiom"].isin(idioms_sample)]
        logger.info("[dry-run] Using idioms: %s", list(idioms_sample))

    idioms   = sorted(out_df["idiom"].unique())
    n_idioms = len(idioms)
    logger.info("Analysing %d idioms", n_idioms)

    # Per-idiom historical metadata (take first row per idiom)
    meta_cols = ["idiom", "denominations", "active_from", "first_attested"]
    available_meta = [c for c in meta_cols if c in out_df.columns]
    idiom_meta = (
        out_df[available_meta].drop_duplicates("idiom").set_index("idiom")
        if len(available_meta) > 1 else pd.DataFrame(index=idioms)
    )

    # --- Per-idiom output-space slopes ---
    slope_rows: list[dict] = []
    for idiom in idioms:
        sub = out_df[out_df["idiom"] == idiom]
        lam_arr   = sub["lambda_years"].values
        score_arr = sub["score_paraphrase"].values
        slope, se = _ols_slope(lam_arr, score_arr)
        baseline  = float(sub["score_baseline"].iloc[0])
        row: dict = {
            "idiom":                   idiom,
            "output_slope":            slope,
            "output_slope_se":         se,
            "baseline_score":          baseline,
            "output_slope_normalised": slope / abs(baseline) if abs(baseline) > 1e-10 else float("nan"),
        }
        # Attach metadata if available
        if idiom in idiom_meta.index:
            for col in ["denominations", "active_from", "first_attested"]:
                if col in idiom_meta.columns:
                    row[col] = idiom_meta.at[idiom, col]
        slope_rows.append(row)

    slopes_df = pd.DataFrame(slope_rows).sort_values("output_slope")
    slopes_df.to_csv(slopes_path, index=False)
    logger.info("Saved → %s", slopes_path)

    # --- Layer sensitivity matrix (activation-space) ---
    logger.info("Computing layer sensitivity matrix …")
    layer_mat = np.zeros((n_idioms, n_layers), dtype=np.float32)

    for i, idiom in enumerate(idioms):
        sub = act_df[act_df["idiom"] == idiom]
        for layer_idx in range(n_layers):
            layer_sub = sub[sub["intervention_layer"] == layer_idx + 1]
            if len(layer_sub) < 3:
                continue
            slope, _ = _ols_slope(
                layer_sub["lambda_years"].values,
                layer_sub["score_paraphrase"].values,
            )
            layer_mat[i, layer_idx] = slope

    np.save(mat_path, layer_mat)
    mat_idx_df = pd.DataFrame({"idiom": idioms})
    mat_idx_df.to_csv(mat_idx_path, index=False)
    logger.info("Saved layer sensitivity matrix → %s", mat_path)

    # --- Print results ---
    mean_abs   = np.abs(layer_mat).mean(axis=0)
    best_layer = int(np.argmax(mean_abs)) + 1

    print("\n=== Intervention Analysis Results ===\n")
    print(f"Layer with strongest temporal–semantic effect: Layer {best_layer}")
    print(f"  (mean |slope| across idioms = {mean_abs[best_layer-1]:.2e})\n")

    print("Per-idiom output slopes (Δtriviality per year of temporal shift):")
    display_cols = [c for c in [
        "idiom", "output_slope", "output_slope_se",
        "baseline_score", "output_slope_normalised",
        "active_from", "first_attested",
    ] if c in slopes_df.columns]
    print(slopes_df[display_cols].to_string(index=False))

    n_neg = (slopes_df["output_slope"] < 0).sum()
    n_pos = (slopes_df["output_slope"] > 0).sum()
    print(f"\n  Negative slopes (more modern → more significant): {n_neg}/{n_idioms}")
    print(f"  Positive slopes (more modern → more trivial):     {n_pos}/{n_idioms}")

    # --- Plot 1: Output intervention curves ---
    x_min = float(out_df["lambda_years"].min())
    x_max = float(out_df["lambda_years"].max())
    palette = sns.color_palette("tab20", n_colors=n_idioms)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Shade pre-active era per idiom's earliest denomination introduction
    if "active_from" in idiom_meta.columns:
        global_active_from = int(idiom_meta["active_from"].min())
        if global_active_from > x_min:
            ax.axvspan(x_min, global_active_from, color="lightgrey", alpha=0.35,
                       label=f"Pre-coin era (before {global_active_from})")

    # Draw idiom curves
    for i, idiom in enumerate(idioms):
        sub = out_df[out_df["idiom"] == idiom].sort_values("lambda_years")
        ax.plot(sub["lambda_years"], sub["score_paraphrase"],
                label=idiom, color=palette[i], linewidth=1.4, alpha=0.85)

        # Mark first attestation year with a small tick on the x-axis
        if "first_attested" in idiom_meta.columns:
            fa = idiom_meta.at[idiom, "first_attested"] if idiom in idiom_meta.index else None
            if fa and not pd.isna(fa) and x_min <= fa <= x_max:
                ax.axvline(fa, color=palette[i], linewidth=0.6,
                           linestyle="-", alpha=0.3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Robustness check markers
    y_top = ax.get_ylim()[1] * 0.97
    _plot_robustness_markers(ax, checks, x_min, x_max, y_top)

    # Legend for severity
    legend_patches = [
        mpatches.Patch(color=v["color"], alpha=0.7, label=f"Robustness: {k} severity")
        for k, v in _SEVERITY_STYLE.items()
    ]
    if "active_from" in idiom_meta.columns and global_active_from > x_min:
        legend_patches.append(
            mpatches.Patch(color="lightgrey", alpha=0.5, label="Pre-coin era")
        )
    ax.legend(
        handles=legend_patches,
        loc="upper right", fontsize=7, framealpha=0.6,
        title="Events", title_fontsize=7,
    )

    # Idiom legend on the side
    idiom_handles = [
        plt.Line2D([0], [0], color=palette[i], linewidth=1.5, label=idiom)
        for i, idiom in enumerate(idioms)
    ]
    ax2_legend = ax.legend(
        handles=idiom_handles,
        loc="upper left", fontsize=6.5, ncol=1,
        framealpha=0.55, title="Idioms", title_fontsize=7,
    )
    ax.add_artist(ax2_legend)   # re-add after second legend call

    ax.set_xlabel("λ (year of predicted year shift)", fontsize=12)
    ax.set_ylabel("score_paraphrase  (trivial − significant)", fontsize=12)
    ax.set_title("Effect of temporal shift on triviality score (output space)", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "output_intervention.png", dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", plots_dir / "output_intervention.png")

    # --- Plot 2: Layer sensitivity heatmap ---
    fig, ax = plt.subplots(figsize=(11, max(4, n_idioms * 0.45)))
    sns.heatmap(
        layer_mat,
        ax=ax,
        xticklabels=[f"L{i+1}" for i in range(n_layers)],
        yticklabels=idioms,
        center=0,
        cmap="RdBu_r",
        annot=True,
        fmt=".1e",
        linewidths=0.3,
        cbar_kws={"label": "Δscore_paraphrase / Δyear"},
    )
    ax.set_xlabel("BERT layer of injection", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Layer-wise sensitivity to temporal direction injection", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "layer_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", plots_dir / "layer_heatmap.png")

    print(f"\nPlots → {plots_dir}/")

    # --- Print robustness check reference table ---
    if checks:
        print(f"\n=== Robustness check events within λ range [{x_min:.0f}, {x_max:.0f}] ===\n")
        in_range = [c for c in checks if x_min <= c["year"] <= x_max]
        for c in sorted(in_range, key=lambda x: x["year"]):
            end = f"–{c['end_year']}" if "end_year" in c else ""
            print(f"  {c['year']}{end:6}  [{c['severity']:6}]  {c['event']}")


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    parser = argparse.ArgumentParser(
        description="Stage 05: Analyse intervention results."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    run(cfg, args.dry_run)


if __name__ == "__main__":
    main()
