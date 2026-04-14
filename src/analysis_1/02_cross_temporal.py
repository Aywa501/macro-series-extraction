"""02_cross_temporal.py — Cross-axis temporal alignment across all layers.

Finding captured
----------------
At L4-5, projecting implicit sequence embeddings (weapons, dynasties) onto
the year-probe direction gives Spearman ρ ≈ 0.86–0.95 for weapons and ρ ≈ 0.55
for dynasties.  The cosine between the year direction and implicit directions
is ~0.2–0.3, showing partial (not identical) temporal subspace overlap.

This script tests all 4 monotonic sequences across ALL layers (1-12) and
produces:
  - A printed table: layer × sequence × [cosine, ρ(seq→yr_dir), ρ(yr→seq_dir)]
  - cross_temporal_results.csv
  - plots/cross_temporal.png  — 2-panel figure with L4 and L11-12 zones marked

Usage
-----
    python src/analysis/02_cross_temporal.py --model bert
    python src/analysis/02_cross_temporal.py --model macberth

Prerequisites
-------------
    Run 01_build_axes.py first to save the year-probe time direction.
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
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _common import (
    SEQUENCES, YEAR_TEMPLATES, YEAR_PROBE_YEARS, VALUE_LAYER,
)

MODEL_NAMES = {
    "bert":     "bert-base-uncased",
    "macberth": "emanjavacas/MacBERTh",
}

N_LAYERS = 12   # bert-base and MacBERTh both have 12 transformer layers


# ---------------------------------------------------------------------------
# Model / embedding helpers
# ---------------------------------------------------------------------------

def load_model(model_key: str):
    name = MODEL_NAMES[model_key]
    print(f"Loading {name} …", flush=True)
    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    print("Loaded.\n", flush=True)
    return model, tokenizer, device


def embed_all_layers(model, tokenizer, device, sentence: str) -> np.ndarray:
    """Return CLS at all 12 transformer layers: (12, 768).
    hidden_states[0] = embedding, hidden_states[1..12] = transformer blocks.
    We return indices 1-12, i.e. shape (12, 768).
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return np.stack([
        out.hidden_states[layer][0, 0, :].cpu().numpy()
        for layer in range(1, N_LAYERS + 1)
    ])   # (12, 768)


def cls_at_layer(model, tokenizer, device, sentence: str, layer: int) -> np.ndarray:
    """CLS at a single 1-indexed layer."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.hidden_states[layer][0, 0, :].cpu().numpy()


# ---------------------------------------------------------------------------
# Build year-probe embeddings at all layers
# ---------------------------------------------------------------------------

def build_year_probe_all_layers(model, tokenizer, device
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed YEAR_TEMPLATES × YEAR_PROBE_YEARS, average over templates per year.
    Returns:
        year_vecs  (n_years, 12, 768)
        years_arr  (n_years,)
    """
    year_centroids = []
    for y in YEAR_PROBE_YEARS:
        layer_avgs = []   # will be (12, 768)
        # Embed all templates, average per layer
        all_embs = []     # list of (12, 768)
        for tmpl in YEAR_TEMPLATES:
            all_embs.append(embed_all_layers(model, tokenizer, device,
                                              tmpl.format(y=y)))
        layer_avg = np.mean(all_embs, axis=0)   # (12, 768)
        year_centroids.append(layer_avg)

    year_vecs = np.stack(year_centroids)        # (n_years, 12, 768)
    years_arr = np.array(YEAR_PROBE_YEARS, dtype=float)
    return year_vecs, years_arr


# ---------------------------------------------------------------------------
# Build sequence embeddings at all layers
# ---------------------------------------------------------------------------

def _period_mid(start: int, end: int) -> float:
    return (start + end) / 2.0


def build_sequence_all_layers(seq_def: dict, model, tokenizer, device
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed every (template, period) combination.
    Returns:
        embs        (N, 12, 768)
        mid_years   (N,)   — mid-year of the period (repeated for each template)
        period_ids  (N,)   — integer period index
    """
    periods    = seq_def["periods"]   # list of (label, start, end)
    templates  = seq_def["templates"]
    all_embs, all_mids, all_pids = [], [], []
    for p_idx, (label, start, end) in enumerate(periods):
        mid = _period_mid(start, end)
        for tmpl in templates:
            sent = tmpl.format(label=label)
            all_embs.append(embed_all_layers(model, tokenizer, device, sent))
            all_mids.append(mid)
            all_pids.append(p_idx)

    return (
        np.stack(all_embs),           # (N, 12, 768)
        np.array(all_mids, dtype=float),
        np.array(all_pids, dtype=int),
    )


# ---------------------------------------------------------------------------
# Embedding diagnostics
# ---------------------------------------------------------------------------

def diagnose_embeddings(seq_name: str, seq_def: dict,
                        seq_embs: np.ndarray, period_ids: np.ndarray) -> None:
    """
    Print per-sentence embedding stats to trace numerical warnings.
    Reports any sentence whose L1 (layer_idx=0) CLS vector is non-finite
    or has an unusually large norm, and prints the sentence text.
    """
    periods    = seq_def["periods"]
    templates  = seq_def["templates"]
    n_tmpl     = len(templates)
    n_sent     = seq_embs.shape[0]

    has_nonfinite = not np.isfinite(seq_embs).all()
    max_norm      = float(np.linalg.norm(seq_embs[:, 0, :], axis=-1).max())

    print(f"  [diag] {seq_name}: finite={not has_nonfinite}  "
          f"max_norm_L1={max_norm:.2f}")

    # Flag any sentence that is non-finite or suspiciously large at any layer
    for sent_idx in range(n_sent):
        norms       = np.linalg.norm(seq_embs[sent_idx], axis=-1)   # (n_layers,)
        bad_layers  = np.where(~np.isfinite(seq_embs[sent_idx]).all(axis=-1))[0]
        large_layers = np.where(norms > 200)[0]
        flagged      = list(bad_layers) + [l for l in large_layers if l not in bad_layers]
        if flagged:
            p_idx   = period_ids[sent_idx]
            t_idx   = sent_idx % n_tmpl
            label, start, end = periods[p_idx]
            sent    = templates[t_idx].format(label=label)
            print(f"    *** sent {sent_idx:>3}  period={label!r} ({start}–{end})  "
                  f"tmpl={t_idx}")
            print(f"        text: {sent!r}")
            print(f"        norms by layer: {np.round(norms, 1)}")
            if bad_layers.size:
                print(f"        non-finite at layers (1-indexed): {bad_layers + 1}")


# ---------------------------------------------------------------------------
# Per-layer cross-axis metrics
# ---------------------------------------------------------------------------

def compute_layer_metrics(
    layer_idx: int,        # 0-indexed within our (12,768) arrays
    seq_embs:  np.ndarray, # (N_seq, 12, 768)
    seq_mids:  np.ndarray, # (N_seq,)
    period_ids: np.ndarray,# (N_seq,)
    year_vecs:  np.ndarray,# (n_years, 12, 768)
    years_arr:  np.ndarray,# (n_years,)
) -> dict:
    """
    For a single layer, compute:
      1. Sequence's own temporal direction (late_centroid − early_centroid)
      2. cosine(year_direction, seq_direction)
      3. ρ(seq_embs projected onto year_dir, seq_mid_years)
      4. ρ(year_probe projected onto seq_dir, years)
    """
    X_seq  = seq_embs[:, layer_idx, :]     # (N_seq, 768)
    X_year = year_vecs[:, layer_idx, :]    # (n_years, 768)

    # --- Year direction from year-probe data at this layer ---
    early_mask = years_arr <= 1450
    late_mask  = years_arr >= 1750
    year_dir   = X_year[late_mask].mean(0) - X_year[early_mask].mean(0)
    year_dir   = year_dir / (np.linalg.norm(year_dir) + 1e-12)

    # --- Sequence direction: use first period vs last period centroids ---
    n_periods  = int(period_ids.max()) + 1
    first_mask = period_ids == 0
    last_mask  = period_ids == (n_periods - 1)
    seq_dir    = X_seq[last_mask].mean(0) - X_seq[first_mask].mean(0)
    seq_dir    = seq_dir / (np.linalg.norm(seq_dir) + 1e-12)

    # --- Cosine ---
    cos = float(np.dot(year_dir, seq_dir))

    # --- ρ: seq onto year_dir ---
    seq_proj_on_yr  = X_seq @ year_dir        # (N_seq,)
    # Average per period for a cleaner signal
    period_projs = np.array([
        seq_proj_on_yr[period_ids == p].mean()
        for p in range(n_periods)
    ])
    period_mids  = np.array([
        seq_mids[period_ids == p].mean()
        for p in range(n_periods)
    ])
    rho_seq_on_yr, p_seq_on_yr = spearmanr(period_projs, period_mids)

    # --- ρ: year-probe onto seq_dir ---
    yr_proj_on_seq = X_year @ seq_dir         # (n_years,)
    rho_yr_on_seq, p_yr_on_seq = spearmanr(yr_proj_on_seq, years_arr)

    return {
        "cosine":         float(cos),
        "rho_seq_on_yr":  float(rho_seq_on_yr),
        "p_seq_on_yr":    float(p_seq_on_yr),
        "rho_yr_on_seq":  float(rho_yr_on_seq),
        "p_yr_on_seq":    float(p_yr_on_seq),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_cross_temporal_plot(results_df: pd.DataFrame, model_key: str,
                              plots_dir: Path) -> None:
    seq_names = results_df["sequence"].unique()
    layers    = sorted(results_df["layer"].unique())
    cmap      = plt.cm.tab10
    colours   = {s: cmap(i) for i, s in enumerate(seq_names)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for seq in seq_names:
        sub = results_df[results_df["sequence"] == seq].sort_values("layer")
        lbl = seq.replace("_", " ")
        ax1.plot(sub["layer"], sub["cosine"].abs(),
                 marker="o", markersize=4, label=lbl, color=colours[seq])
        ax2.plot(sub["layer"], sub["rho_seq_on_yr"],
                 marker="o", markersize=4, label=lbl, color=colours[seq])

    for ax in (ax1, ax2):
        # Shade L4 zone (our "shared temporal peak")
        ax.axvspan(3.5, 5.5, alpha=0.10, color="blue", label="_L4-5 zone")
        ax.axvspan(3.5, 5.5, alpha=0.00, color="none")
        ax.text(4.5, ax.get_ylim()[1] if ax.get_ylim()[1] < 1 else 0.95,
                "L4-5", fontsize=8, color="steelblue",
                ha="center", va="top", alpha=0.7)
        # Shade L11-12 zone
        ax.axvspan(10.5, 12.5, alpha=0.08, color="green", label="_L11-12 zone")
        ax.text(11.5, ax.get_ylim()[1] if ax.get_ylim()[1] < 1 else 0.95,
                "L11-12", fontsize=8, color="green",
                ha="center", va="top", alpha=0.7)
        ax.set_xlabel("Layer (1-indexed)")
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("|cosine(year_dir, seq_dir)|")
    ax1.set_title(f"Cosine alignment — year vs sequence direction\n{model_key}")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1)

    ax2.set_ylabel("Spearman ρ  (seq projected onto year_dir vs mid-year)")
    ax2.set_title(f"Cross-projection ρ — seq embeddings onto year direction\n{model_key}")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color="black", linewidth=0.7, alpha=0.5)

    fig.suptitle(
        f"Cross-axis temporal alignment by layer — {model_key}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = plots_dir / "cross_temporal.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved plot → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-axis temporal alignment test across all layers."
    )
    parser.add_argument("--model", choices=["bert", "macberth"], default="bert")
    args = parser.parse_args()
    model_key = args.model

    out_dir   = PROJECT_ROOT / "data" / model_key / "value_probe"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Check that 01_build_axes.py has been run
    time_path = out_dir / f"time_direction_L{VALUE_LAYER}.npy"
    if not time_path.exists():
        print(f"ERROR: {time_path} not found. Run 01_build_axes.py first.")
        sys.exit(1)

    # Load model
    model, tokenizer, device = load_model(model_key)

    # Build year-probe embeddings at all layers
    print("Embedding year-probe sentences at all layers …", flush=True)
    year_vecs, years_arr = build_year_probe_all_layers(model, tokenizer, device)
    print(f"  year_vecs shape: {year_vecs.shape}")

    # Process each sequence
    all_rows = []
    for seq_name, seq_def in SEQUENCES.items():
        print(f"\nProcessing sequence: {seq_name} …", flush=True)
        seq_embs, seq_mids, period_ids = build_sequence_all_layers(
            seq_def, model, tokenizer, device)
        print(f"  seq_embs shape: {seq_embs.shape}")
        diagnose_embeddings(seq_name, seq_def, seq_embs, period_ids)

        for layer_idx in range(N_LAYERS):
            layer = layer_idx + 1   # 1-indexed for display
            m = compute_layer_metrics(
                layer_idx, seq_embs, seq_mids, period_ids,
                year_vecs, years_arr,
            )
            all_rows.append({
                "model":        model_key,
                "sequence":     seq_name,
                "layer":        layer,
                **m,
            })

    results_df = pd.DataFrame(all_rows)

    # ---- Print table ----
    print()
    print("=" * 90)
    print(f"CROSS-TEMPORAL ALIGNMENT  —  model={model_key}")
    print("=" * 90)
    print(f"  {'Layer':>5}  {'Sequence':<20}  {'|cosine|':>8}  "
          f"{'ρ(seq→yr)':>10}  {'p':>8}  {'ρ(yr→seq)':>10}  {'p':>8}")
    print("-" * 90)
    for _, row in results_df.sort_values(["layer", "sequence"]).iterrows():
        print(f"  {int(row['layer']):>5}  {row['sequence']:<20}  "
              f"{abs(row['cosine']):>8.4f}  "
              f"{row['rho_seq_on_yr']:>10.4f}  {row['p_seq_on_yr']:>8.2e}  "
              f"{row['rho_yr_on_seq']:>10.4f}  {row['p_yr_on_seq']:>8.2e}")

    # ---- Focus table: L4 summary ----
    print()
    print(f"--- Layer {VALUE_LAYER} summary (the 'shared temporal peak') ---")
    l4 = results_df[results_df["layer"] == VALUE_LAYER]
    print(f"  {'Sequence':<20}  {'|cosine|':>8}  {'ρ(seq→yr)':>10}  {'ρ(yr→seq)':>10}")
    for _, row in l4.iterrows():
        print(f"  {row['sequence']:<20}  "
              f"{abs(row['cosine']):>8.4f}  "
              f"{row['rho_seq_on_yr']:>10.4f}  "
              f"{row['rho_yr_on_seq']:>10.4f}")

    # ---- Save CSV ----
    csv_path = out_dir / "cross_temporal_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # ---- Plot ----
    make_cross_temporal_plot(results_df, model_key, plots_dir)


if __name__ == "__main__":
    main()
