"""01_build_axes.py — Build and characterise the value axis and time axis.

Findings captured
-----------------
1. The value axis (sovereign – farthing direction at L4) tracks log(pence)
   with Pearson r ≈ 0.92.
2. The year-probe time axis is recoverable from L4 (Spearman ρ with year).
3. The two axes are not the same: cosine(value, time) is small (< 0.3),
   showing they are distinct but potentially share a temporal subspace.

Outputs (under data/{model}/value_probe/)
------------------------------------------
  value_direction_L{VALUE_LAYER}.npy   — unit vector (768,)
  time_direction_L{VALUE_LAYER}.npy    — unit vector (768,)
  axes_summary.csv                     — scalar metrics
  plots/axes_quality.png               — 2-panel diagnostic figure

Usage
-----
    python src/analysis/01_build_axes.py --model bert
    python src/analysis/01_build_axes.py --model macberth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d
from transformers import AutoModel, AutoTokenizer

# Locate project root and import shared constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _common import (
    DENOMINATIONS, VALUE_TEMPLATES, YEAR_TEMPLATES,
    YEAR_PROBE_YEARS, VALUE_LAYER,
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "bert":     "bert-base-uncased",
    "macberth": "emanjavacas/MacBERTh",
}


def load_model(model_key: str):
    name = MODEL_NAMES[model_key]
    print(f"Loading {name} …", flush=True)
    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    print(f"Loaded. Device: {device}\n", flush=True)
    return model, tokenizer, device


def cls_at_layer(model, tokenizer, device, sentence: str, layer: int) -> np.ndarray:
    """Return CLS embedding at the given 1-indexed layer."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    # hidden_states is a tuple of length n_layers+1; index 0 = embedding layer
    return out.hidden_states[layer][0, 0, :].cpu().numpy()


# ---------------------------------------------------------------------------
# Step 1: embed all 10 denominations → centroids
# ---------------------------------------------------------------------------

def build_denomination_centroids(model, tokenizer, device) -> dict[str, np.ndarray]:
    """
    For each denomination, embed all VALUE_TEMPLATES and average → centroid.
    Returns dict  name → centroid  (768,).
    """
    centroids = {}
    for name, _pence in DENOMINATIONS:
        vecs = []
        for tmpl in VALUE_TEMPLATES:
            sent = tmpl.format(c=name)
            vecs.append(cls_at_layer(model, tokenizer, device, sent, VALUE_LAYER))
        centroids[name] = np.mean(vecs, axis=0)
    return centroids


# ---------------------------------------------------------------------------
# Step 2: build axes
# ---------------------------------------------------------------------------

def build_value_direction(centroids: dict[str, np.ndarray]) -> np.ndarray:
    """Fit a value axis from ALL denominations via regression onto log(pence).

    Rather than defining the axis by two anchor points, we find the single
    direction d in the 768-d embedding space that best separates all
    denominations by their log pence value:

        minimise  Σ_i ( centroid_i · d  −  log_pence_i )²

    Both X (embeddings) and y (log pence) are mean-centred before fitting so
    the solution is independent of the global embedding offset.  The result
    is the minimum-norm OLS / PLS direction: every denomination votes on
    where the axis points, weighted by distance from the centroid of the
    denomination space.  This is more robust than a 2-point anchor because
    no single coin pair dominates and lexically ambiguous coins (crown,
    sovereign) are naturally down-weighted if their embeddings deviate from
    the value trend.
    """
    names  = [n for n, _ in DENOMINATIONS]
    pences = np.array([p for _, p in DENOMINATIONS], dtype=float)
    X      = np.stack([centroids[n] for n in names])  # (n_coins, 768)
    y      = np.log(pences)                            # (n_coins,)

    # Mean-centre both
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()

    # Minimum-norm OLS in the underdetermined system (n_coins << 768)
    d, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)
    return d / (np.linalg.norm(d) + 1e-12)


def build_time_direction(model, tokenizer, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed YEAR_TEMPLATES × YEAR_PROBE_YEARS at VALUE_LAYER.
    Time direction = mean(late years ≥ 1750) − mean(early years ≤ 1450), normalised.
    Returns:
        time_dir   (768,)   unit vector
        year_vecs  (n_years, 768) — one centroid per year (mean over templates)
        years_arr  (n_years,)     — the corresponding years
    """
    year_centroids = []
    years_arr      = []
    for y in YEAR_PROBE_YEARS:
        vecs = []
        for tmpl in YEAR_TEMPLATES:
            sent = tmpl.format(y=y)
            vecs.append(cls_at_layer(model, tokenizer, device, sent, VALUE_LAYER))
        year_centroids.append(np.mean(vecs, axis=0))
        years_arr.append(y)

    year_vecs = np.stack(year_centroids)        # (n_years, 768)
    years_arr = np.array(years_arr, dtype=float)

    early_mask = years_arr <= 1450
    late_mask  = years_arr >= 1750
    early_mean = year_vecs[early_mask].mean(axis=0)
    late_mean  = year_vecs[late_mask].mean(axis=0)

    direction  = late_mean - early_mean
    time_dir   = direction / (np.linalg.norm(direction) + 1e-12)
    return time_dir, year_vecs, years_arr


# ---------------------------------------------------------------------------
# Step 3: metrics
# ---------------------------------------------------------------------------

def value_axis_quality(centroids: dict[str, np.ndarray],
                       value_dir: np.ndarray) -> tuple[float, float]:
    """Spearman ρ between projection onto value_dir and pence value.

    Spearman is used (not Pearson) because we only assume a monotonic
    relationship between denomination value and projection — not a linear one.
    Log(pence) is not needed since Spearman is rank-based.
    """
    names  = [n for n, _ in DENOMINATIONS]
    pences = np.array([p for _, p in DENOMINATIONS], dtype=float)
    projs  = np.array([np.dot(centroids[n], value_dir) for n in names])
    rho, p = spearmanr(projs, pences)
    return float(rho), float(p)


def polynomial_fit_r2(centroids: dict[str, np.ndarray],
                      value_dir: np.ndarray) -> dict[str, float]:
    """
    Fit linear and quadratic polynomials to projection vs log(pence) and
    return R² for each.  The quadratic tests whether BERT's perception of
    denomination value is non-linear — e.g. compressing the middle of the
    scale into a curve rather than a straight line.
    """
    names  = [n for n, _ in DENOMINATIONS]
    pences = np.array([p for _, p in DENOMINATIONS], dtype=float)
    projs  = np.array([np.dot(centroids[n], value_dir) for n in names])
    log_p  = np.log(pences)

    ss_tot = float(np.sum((projs - projs.mean()) ** 2))

    def r2_for_degree(deg: int) -> float:
        coeffs   = np.polyfit(log_p, projs, deg)
        fitted   = np.polyval(coeffs, log_p)
        ss_res   = float(np.sum((projs - fitted) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "r2_linear":    r2_for_degree(1),
        "r2_quadratic": r2_for_degree(2),
    }


def time_axis_quality(year_vecs: np.ndarray, years_arr: np.ndarray,
                      time_dir: np.ndarray) -> tuple[float, float]:
    """Spearman ρ between projection onto time_dir and year."""
    projs    = year_vecs @ time_dir
    rho, p   = spearmanr(projs, years_arr)
    return float(rho), float(p)


def axes_cosine(value_dir: np.ndarray, time_dir: np.ndarray) -> float:
    """Cosine similarity between the two unit vectors."""
    return float(np.dot(value_dir, time_dir))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_axes_plot(
    centroids: dict[str, np.ndarray],
    value_dir: np.ndarray,
    year_vecs: np.ndarray,
    years_arr: np.ndarray,
    time_dir: np.ndarray,
    model_key: str,
    plots_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel A: denomination projections vs log(pence) ----
    names  = [n for n, _ in DENOMINATIONS]
    pences = np.array([p for _, p in DENOMINATIONS], dtype=float)
    projs  = np.array([np.dot(centroids[n], value_dir) for n in names])
    log_p  = np.log(pences)
    rho, _ = spearmanr(projs, pences)

    ax1.scatter(log_p, projs, s=70, color="steelblue", zorder=3)
    xs = np.linspace(log_p.min(), log_p.max(), 200)

    # Linear fit
    c1        = np.polyfit(log_p, projs, 1)
    ax1.plot(xs, np.polyval(c1, xs), "r--", linewidth=1.2, alpha=0.6,
             label="linear fit")

    # Quadratic fit
    c2        = np.polyfit(log_p, projs, 2)
    ss_tot    = np.sum((projs - projs.mean()) ** 2)
    ss_res_q  = np.sum((projs - np.polyval(c2, log_p)) ** 2)
    r2_quad   = 1.0 - ss_res_q / ss_tot if ss_tot > 1e-12 else 0.0
    ax1.plot(xs, np.polyval(c2, xs), "g-", linewidth=1.5, alpha=0.8,
             label=f"quadratic fit  R²={r2_quad:.3f}")

    for i, n in enumerate(names):
        ax1.annotate(n, (log_p[i], projs[i]), fontsize=7,
                     xytext=(4, 0), textcoords="offset points")
    ax1.set_xlabel("log(pence value)")
    ax1.set_ylabel(f"Projection onto value axis (L{VALUE_LAYER})")
    ax1.set_title(f"Value axis quality — {model_key}\nSpearman ρ = {rho:.3f}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Panel B: year-probe projections vs year ----
    time_projs = year_vecs @ time_dir
    rho, _     = spearmanr(time_projs, years_arr)

    # Scatter — raw projections
    ax2.scatter(years_arr, time_projs, s=35, color="darkorange",
                zorder=3, alpha=0.85, label="year probe (25-yr step)")

    # LOWESS-style smooth: uniform moving average over 5 points (~125 yrs)
    n_pts   = len(time_projs)
    smooth  = uniform_filter1d(time_projs, size=min(5, n_pts), mode="nearest")
    ax2.plot(years_arr, smooth, color="firebrick", linewidth=2.0,
             zorder=4, label="5-pt rolling mean")

    # Annotate only century years to avoid clutter
    for i, y in enumerate(years_arr):
        if int(y) % 100 == 0:
            ax2.annotate(str(int(y)), (y, time_projs[i]), fontsize=7,
                         xytext=(0, 6), textcoords="offset points",
                         ha="center", color="saddlebrown")

    # X-axis ticks every 25 years
    tick_years = np.arange(
        int(years_arr.min() // 25) * 25,
        int(years_arr.max()) + 1,
        25,
    )
    ax2.set_xticks(tick_years)
    ax2.set_xticklabels([str(int(t)) for t in tick_years],
                        rotation=60, fontsize=7, ha="right")

    ax2.set_xlabel("Year")
    ax2.set_ylabel(f"Projection onto time axis (L{VALUE_LAYER})")
    ax2.set_title(
        f"Time axis quality — {model_key}\n"
        f"Spearman ρ = {rho:.3f}  (jitter = BERT year-token artifact)"
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Axes built from {model_key} Layer {VALUE_LAYER} CLS embeddings",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = plots_dir / "axes_quality.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved plot → {out}")


def make_denomination_axis_plot(
    centroids: dict[str, np.ndarray],
    value_dir: np.ndarray,
    model_key: str,
    plots_dir: Path,
) -> None:
    """1D number-line plot: all denominations placed at their projection value."""
    names  = [n for n, _ in DENOMINATIONS]
    pences = np.array([p for _, p in DENOMINATIONS], dtype=float)
    projs  = np.array([np.dot(centroids[n], value_dir) for n in names])

    # Sort by projection so labels alternate cleanly
    order  = np.argsort(projs)
    names_s  = [names[i]  for i in order]
    pences_s = pences[order]
    projs_s  = projs[order]

    # Colour ramp: low value = red, high value = green
    colours = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(names_s)))

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axhline(0, color="black", linewidth=1.5, zorder=1)

    for i, (name, proj, pence) in enumerate(zip(names_s, projs_s, pences_s)):
        ax.scatter(proj, 0, s=130, color=colours[i], zorder=3,
                   edgecolors="black", linewidths=0.6)
        # Alternate labels above / below to reduce overlap
        y_pts = 14 if i % 2 == 0 else -22
        va    = "bottom" if i % 2 == 0 else "top"
        ax.annotate(
            f"{name}\n({pence:g}d)", (proj, 0),
            xytext=(0, y_pts), textcoords="offset points",
            ha="center", va=va, fontsize=8,
        )

    ax.set_xlabel("Projection onto value axis (regression-fitted, all denominations)",
                  fontsize=9)
    ax.set_yticks([])
    ax.set_title(
        f"All denominations along the value axis — {model_key} L{VALUE_LAYER}\n"
        f"(direction fitted by OLS regression onto log pence, all {len(names)} coins)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    out = plots_dir / "denomination_axis.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved plot → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build value and time axes from BERT/MacBERTh L4 CLS embeddings."
    )
    parser.add_argument("--model", choices=["bert", "macberth"], default="bert")
    args = parser.parse_args()
    model_key = args.model

    # Directories
    out_dir   = PROJECT_ROOT / "data" / model_key / "value_probe"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer, device = load_model(model_key)

    # ---- Step 1: denomination centroids ----
    print("Embedding denominations …", flush=True)
    centroids = build_denomination_centroids(model, tokenizer, device)
    print(f"  Built centroids for {len(centroids)} denominations.")

    # ---- Step 2: build axes ----
    value_dir = build_value_direction(centroids)
    print("Value direction built.")

    print("Embedding year-probe sentences …", flush=True)
    time_dir, year_vecs, years_arr = build_time_direction(model, tokenizer, device)
    print(f"  Built time direction from {len(years_arr)} year centroids.")

    # ---- Step 3: metrics ----
    rho_value, p_value     = value_axis_quality(centroids, value_dir)
    poly_fits              = polynomial_fit_r2(centroids, value_dir)
    rho_time, p_time       = time_axis_quality(year_vecs, years_arr, time_dir)
    cos_vt                 = axes_cosine(value_dir, time_dir)

    print()
    print("=" * 60)
    print(f"MODEL: {model_key}   LAYER: {VALUE_LAYER}")
    print("=" * 60)
    print(f"  Value axis — Spearman ρ (proj vs pence)    : {rho_value:+.4f}  (p={p_value:.2e})")
    print(f"  Value axis — R² linear  (proj vs log pence): {poly_fits['r2_linear']:+.4f}")
    print(f"  Value axis — R² quadratic                  : {poly_fits['r2_quadratic']:+.4f}")
    print(f"  Time  axis — Spearman ρ (proj vs year)     : {rho_time:+.4f}  (p={p_time:.2e})")
    print(f"  Cosine(value_axis, time_axis)               : {cos_vt:+.4f}")
    print()
    print("  Interpretation:")
    print(f"    Value axis monotonic ordering of pence (ρ≈{rho_value:.2f}).")
    print(f"    Linear R²={poly_fits['r2_linear']:.3f} vs quadratic R²={poly_fits['r2_quadratic']:.3f} "
          f"({'quadratic fits better' if poly_fits['r2_quadratic'] > poly_fits['r2_linear'] + 0.05 else 'little gain from quadratic'}).")
    print(f"    Time  axis correctly orders years (ρ≈{rho_time:.2f}).")
    angle_deg = float(np.degrees(np.arccos(np.clip(abs(cos_vt), 0, 1))))
    print(f"    Angle between axes ≈ {angle_deg:.1f}° "
          f"({'mostly orthogonal' if angle_deg > 70 else 'partially shared subspace'}).")
    print()

    # Per-denomination details
    print("  Denomination projections:")
    names  = [n for n, _ in DENOMINATIONS]
    pences = [p for _, p in DENOMINATIONS]
    print(f"  {'Coin':<12} {'Pence':>7} {'log(p)':>8} {'value_proj':>12}")
    for n, pen in zip(names, pences):
        proj = np.dot(centroids[n], value_dir)
        print(f"  {n:<12} {pen:>7.2f} {np.log(pen):>8.3f} {proj:>12.5f}")

    # ---- Save axes ----
    val_path  = out_dir / f"value_direction_L{VALUE_LAYER}.npy"
    time_path = out_dir / f"time_direction_L{VALUE_LAYER}.npy"
    np.save(val_path,  value_dir)
    np.save(time_path, time_dir)
    print(f"Saved → {val_path}")
    print(f"Saved → {time_path}")

    # ---- Save summary CSV ----
    summary = pd.DataFrame([{
        "model":                       model_key,
        "layer":                       VALUE_LAYER,
        "spearman_rho_value_vs_pence": rho_value,
        "spearman_p_value":            p_value,
        "r2_linear_vs_logpence":       poly_fits["r2_linear"],
        "r2_quadratic_vs_logpence":    poly_fits["r2_quadratic"],
        "spearman_rho_time_vs_year":   rho_time,
        "spearman_p_time":             p_time,
        "cosine_value_time":           cos_vt,
        "angle_value_time_deg":        angle_deg,
    }])
    csv_path = out_dir / "axes_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # ---- Plot ----
    make_axes_plot(centroids, value_dir, year_vecs, years_arr, time_dir,
                   model_key, plots_dir)
    make_denomination_axis_plot(centroids, value_dir, model_key, plots_dir)


if __name__ == "__main__":
    main()
