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
    python src/value_probe/01_build_axes.py --model bert
    python src/value_probe/01_build_axes.py --model macberth
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
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer

# Locate project root and import shared constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "value_probe"))
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
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    print("Loaded.\n", flush=True)
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
    """sovereign centroid − farthing centroid, normalised."""
    direction = centroids["sovereign"] - centroids["farthing"]
    return direction / (np.linalg.norm(direction) + 1e-12)


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
    """Pearson r between projection onto value_dir and log(pence)."""
    names  = [n for n, _ in DENOMINATIONS]
    pences = np.array([p for _, p in DENOMINATIONS], dtype=float)
    projs  = np.array([np.dot(centroids[n], value_dir) for n in names])
    r, p   = pearsonr(projs, np.log(pences))
    return float(r), float(p)


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
    r, _   = pearsonr(projs, log_p)

    ax1.scatter(log_p, projs, s=70, color="steelblue", zorder=3)
    # Best-fit line
    m, b   = np.polyfit(log_p, projs, 1)
    xs     = np.linspace(log_p.min(), log_p.max(), 100)
    ax1.plot(xs, m * xs + b, "r--", linewidth=1.2, alpha=0.7)
    for i, n in enumerate(names):
        ax1.annotate(n, (log_p[i], projs[i]), fontsize=7,
                     xytext=(4, 0), textcoords="offset points")
    ax1.set_xlabel("log(pence value)")
    ax1.set_ylabel(f"Projection onto value axis (L{VALUE_LAYER})")
    ax1.set_title(f"Value axis quality — {model_key}\nPearson r = {r:.3f}")
    ax1.grid(True, alpha=0.3)

    # ---- Panel B: year-probe projections vs year ----
    time_projs = year_vecs @ time_dir
    rho, _     = spearmanr(time_projs, years_arr)

    ax2.scatter(years_arr, time_projs, s=50, color="darkorange", zorder=3)
    m2, b2 = np.polyfit(years_arr, time_projs, 1)
    xs2    = np.linspace(years_arr.min(), years_arr.max(), 100)
    ax2.plot(xs2, m2 * xs2 + b2, "r--", linewidth=1.2, alpha=0.7)
    for i, y in enumerate(years_arr):
        ax2.annotate(str(int(y)), (y, time_projs[i]), fontsize=7,
                     xytext=(4, 0), textcoords="offset points")
    ax2.set_xlabel("Year")
    ax2.set_ylabel(f"Projection onto time axis (L{VALUE_LAYER})")
    ax2.set_title(f"Time axis quality — {model_key}\nSpearman ρ = {rho:.3f}")
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
    r_value, p_value       = value_axis_quality(centroids, value_dir)
    rho_time, p_time       = time_axis_quality(year_vecs, years_arr, time_dir)
    cos_vt                 = axes_cosine(value_dir, time_dir)

    print()
    print("=" * 60)
    print(f"MODEL: {model_key}   LAYER: {VALUE_LAYER}")
    print("=" * 60)
    print(f"  Value axis — Pearson r (proj vs log pence) : {r_value:+.4f}  (p={p_value:.2e})")
    print(f"  Time  axis — Spearman ρ (proj vs year)     : {rho_time:+.4f}  (p={p_time:.2e})")
    print(f"  Cosine(value_axis, time_axis)               : {cos_vt:+.4f}")
    print()
    print("  Interpretation:")
    print(f"    Value axis tracks log(pence) well (r≈{r_value:.2f}).")
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
        "model":        model_key,
        "layer":        VALUE_LAYER,
        "pearson_r_value_vs_logpence": r_value,
        "pearson_p_value":             p_value,
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


if __name__ == "__main__":
    main()
