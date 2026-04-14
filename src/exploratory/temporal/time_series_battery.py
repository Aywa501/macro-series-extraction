"""time_series_battery.py — Layer sweep over all categorical time series.

Each series is bounded by its own natural start/end; test years span the
union of all series ranges. Overlap regions (where multiple series are
simultaneously active) are highlighted in all plots.

Outputs (data/appendix/)
------------------------
  battery_coverage_{model}.png      — timeline + overlap density
  battery_layer_sweep_{model}.png   — series × layer heatmap (sorted by start)
  battery_mean_rho_{model}.png      — mean|ρ| per layer with per-series traces
  battery_projection_{model}.png    — PC1 projection at best layer vs coverage
  battery_layer_sweep_{model}.csv   — full (series, layer, rho, p, n)

Usage
-----
    python3 src/exploratory/time_series_battery.py --model bert
    python3 src/exploratory/time_series_battery.py --model macberth
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import torch
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
TIME_SERIES_DIR = PROJECT_ROOT.parent / "time_series"
OUT_DIR         = PROJECT_ROOT / "data" / "outputs" / "battery"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "bert":     "bert-base-uncased",
    "macberth": "emanjavacas/MacBERTh",
}

SKIP = {"time_entities.csv", "time_entities.json", "geologic_time.csv"}

CARRIER_TEMPLATES = [
    "the year is {y} .",
    "it is the year {y} .",
    "the date is {y} .",
    "this was written in {y} .",
]

YEAR_STEP = 25   # resolution of test-year grid


# ---------------------------------------------------------------------------
# Series loading
# ---------------------------------------------------------------------------

def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = set(df.columns)
    if "start_year" in cols and "end_year" in cols and "name" in cols:
        df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
        df["end_year"]   = pd.to_numeric(df["end_year"],   errors="coerce")
        return df.dropna(subset=["start_year"]).sort_values("start_year").reset_index(drop=True)
    raise ValueError(f"Unrecognised format: {list(df.columns)}")


def series_natural_range(df: pd.DataFrame) -> tuple[float, float]:
    s = float(df["start_year"].min())
    e = float(df["end_year"].dropna().max()) if df["end_year"].notna().any() else 2025.0
    return s, e


def get_ordinal(df: pd.DataFrame, year: float) -> float | None:
    """Ordinal position of this year within the sequence; None if outside range."""
    in_period = df[
        (df["start_year"] <= year) &
        (df["end_year"].fillna(9999) >= year)
    ]
    if len(in_period) > 0:
        return float(np.mean(in_period.index.to_numpy()))
    future = df[df["start_year"] > year]
    past   = df[df["start_year"] <= year]
    if len(past) > 0 and len(future) > 0:
        p_idx = int(past.index[-1]);  f_idx = int(future.index[0])
        p_yr  = float(past.loc[p_idx, "start_year"])
        f_yr  = float(future.loc[f_idx, "start_year"])
        t = (year - p_yr) / (f_yr - p_yr + 1e-9)
        return float(p_idx + t * (f_idx - p_idx))
    # Outside coverage on one side — return boundary but mark as covered
    if len(past) > 0:
        return float(past.index[-1])
    if len(future) > 0:
        return float(future.index[0])
    return None


def series_ordinals(df: pd.DataFrame, test_years: np.ndarray,
                    s_start: float, s_end: float) -> np.ndarray:
    """Ordinal at each test year; NaN outside the series' natural range."""
    vals = np.full(len(test_years), np.nan)
    for i, y in enumerate(test_years):
        if y < s_start or y > s_end:
            continue
        v = get_ordinal(df, y)
        if v is not None:
            vals[i] = v
    return vals


# ---------------------------------------------------------------------------
# Model + embedding
# ---------------------------------------------------------------------------

def load_model(model_key: str, device: torch.device):
    name = MODEL_NAMES[model_key]
    print(f"Loading {name} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    n = model.config.num_hidden_layers
    print(f"Loaded. {n} transformer layers + embedding = {n+1} total.\n", flush=True)
    return model, tokenizer


def embed_all_layers(model, tokenizer, device,
                     test_years: np.ndarray) -> dict[int, np.ndarray]:
    """Return {layer: (n_years, hidden_size)} for every layer."""
    n_layers    = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size
    n_years     = len(test_years)
    layer_embs  = {l: np.zeros((n_years, hidden_size)) for l in range(n_layers)}

    print(f"Embedding {n_years} test years × {len(CARRIER_TEMPLATES)} templates "
          f"× {n_layers} layers …", flush=True)

    for i, year in enumerate(test_years):
        print(f"  [{i+1:03d}/{n_years}] year={int(year):>6}", end="\r", flush=True)
        accum = {l: np.zeros(hidden_size) for l in range(n_layers)}
        for tmpl in CARRIER_TEMPLATES:
            sent = tmpl.format(y=int(year))
            enc  = tokenizer(sent, return_tensors="pt",
                             truncation=True, max_length=32).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for l, hidden in enumerate(out.hidden_states):
                accum[l] += hidden[0, 0, :].cpu().numpy()
        for l in range(n_layers):
            layer_embs[l][i] = accum[l] / len(CARRIER_TEMPLATES)

    print()
    return layer_embs


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def temporal_axis(X: np.ndarray) -> np.ndarray:
    Xc  = X - X.mean(axis=0)
    pca = PCA(n_components=1).fit(Xc)
    return pca.components_[0]


def correlate(proj: np.ndarray, vals: np.ndarray) -> tuple[float, float, int]:
    mask = ~np.isnan(vals)
    n    = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    rho, p = spearmanr(proj[mask], vals[mask])
    return float(rho), float(p), n


def sig_label(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return ""


def overlap_count(all_vals: dict[str, np.ndarray]) -> np.ndarray:
    """Number of series with non-NaN coverage at each test year."""
    mat = np.stack(list(all_vals.values()), axis=1)   # (n_years, n_series)
    return (~np.isnan(mat)).sum(axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CMAP_OVERLAP = plt.cm.YlOrRd


def plot_coverage(series_meta: dict[str, tuple[float, float]],
                  all_vals: dict[str, np.ndarray],
                  test_years: np.ndarray, model_key: str) -> plt.Figure:
    """Timeline showing each series' active range + overlap density."""
    names    = list(series_meta.keys())
    n_series = len(names)
    overlap  = overlap_count(all_vals)

    fig, axes = plt.subplots(2, 1, figsize=(14, max(6, n_series * 0.4 + 3)),
                             gridspec_kw={"height_ratios": [1, n_series]})

    # ── Top: overlap density ──────────────────────────────────────────────
    ax0 = axes[0]
    ax0.fill_between(test_years, overlap, alpha=0.5, color="#4575b4")
    ax0.plot(test_years, overlap, lw=1.2, color="#4575b4")
    ax0.set_ylabel("# series active", fontsize=8)
    ax0.set_xlim(test_years[0], test_years[-1])
    ax0.set_ylim(0, n_series + 0.5)
    ax0.grid(True, alpha=0.2)
    ax0.set_title(
        f"Temporal series coverage & overlap — {MODEL_NAMES[model_key]}\n"
        f"Test years {int(test_years[0])} – {int(test_years[-1])}, "
        f"step {YEAR_STEP} yr, n={len(test_years)} points",
        fontsize=9)

    # ── Bottom: per-series bars ───────────────────────────────────────────
    ax1 = axes[1]
    palette = plt.cm.tab10(np.linspace(0, 1, n_series))

    # Sort by start year for visual clarity
    sorted_names = sorted(names, key=lambda n: series_meta[n][0])
    for row_idx, name in enumerate(sorted_names):
        s_start, s_end = series_meta[name]
        color = palette[names.index(name)]
        # Draw bar over covered range
        vals = all_vals[name]
        covered = test_years[~np.isnan(vals)]
        if len(covered) > 0:
            ax1.barh(row_idx, covered[-1] - covered[0],
                     left=covered[0], height=0.65,
                     color=color, alpha=0.75)
        ax1.text(test_years[0] - (test_years[-1] - test_years[0]) * 0.01,
                 row_idx, name, va="center", ha="right", fontsize=7, color=color)

    ax1.set_xlim(test_years[0], test_years[-1])
    ax1.set_ylim(-0.7, n_series - 0.3)
    ax1.set_yticks([])
    ax1.set_xlabel("Year (CE; negative = BCE)", fontsize=8)
    ax1.grid(True, alpha=0.15, axis="x")

    # Shade overlap regions on bottom panel
    for j, yr in enumerate(test_years[:-1]):
        ov = int(overlap[j])
        if ov > 1:
            ax1.axvspan(test_years[j], test_years[j+1],
                        alpha=min(0.04 * (ov - 1), 0.3), color="grey")

    fig.tight_layout()
    return fig


def plot_layer_sweep(pivot_rho: pd.DataFrame, pivot_p: pd.DataFrame,
                     pivot_n: pd.DataFrame, model_key: str) -> plt.Figure:
    """Heatmap: series (rows, sorted by start year) × layer (cols)."""
    series  = pivot_rho.index.tolist()
    layers  = pivot_rho.columns.tolist()
    data    = pivot_rho.values
    p_data  = pivot_p.values
    n_data  = pivot_n.values

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.85),
                                    max(6, len(series) * 0.55 + 1.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Spearman ρ")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series, fontsize=8)
    ax.set_xlabel("Layer (0 = embedding)", fontsize=9)

    for i in range(len(series)):
        for j in range(len(layers)):
            rho = data[i, j];  p = p_data[i, j];  n = n_data[i, j]
            if not np.isnan(rho):
                star = sig_label(p)
                ax.text(j, i, f"{rho:.2f}{star}\n(n={int(n)})",
                        ha="center", va="center", fontsize=4.5,
                        color="white" if abs(rho) > 0.6 else "black")

    ax.set_title(
        f"Temporal encoding by layer — {MODEL_NAMES[model_key]}\n"
        f"PC1(year embeddings) vs categorical sequences  "
        f"(series bounded by own range; Spearman ρ)",
        fontsize=9)
    fig.tight_layout()
    return fig


def plot_mean_rho(df: pd.DataFrame, model_key: str) -> plt.Figure:
    """Mean |ρ| per layer — bold mean, faint per-series traces."""
    layers      = sorted(df["layer"].unique())
    series_list = df["series"].unique()
    palette     = plt.cm.tab10(np.linspace(0, 1, len(series_list)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for color, name in zip(palette, series_list):
        sub  = df[df["series"] == name].sort_values("layer")
        rhos = sub["rho"].abs().values
        ax.plot(layers, rhos, "o-", color=color, alpha=0.45, lw=1.2,
                markersize=3, label=name)

    mean_per = df.groupby("layer")["rho"].apply(lambda x: np.nanmean(np.abs(x)))
    ax.plot(layers, mean_per.values, "o-", color="black", lw=2.5,
            markersize=6, label="mean |ρ|", zorder=10)

    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_ylabel("|Spearman ρ|", fontsize=9)
    ax.set_xlabel("Layer (0 = embedding)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.set_title(f"Mean |temporal ρ| per layer — {MODEL_NAMES[model_key]}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_projection(layer_embs: dict[int, np.ndarray],
                    all_vals: dict[str, np.ndarray],
                    series_meta: dict[str, tuple[float, float]],
                    test_years: np.ndarray, best_layer: int,
                    model_key: str) -> plt.Figure:
    """PC1 projection at best layer + per-series coverage overlay."""
    X      = layer_embs[best_layer]
    t_axis = temporal_axis(X)
    t_proj = X @ t_axis
    if spearmanr(t_proj, test_years)[0] < 0:
        t_proj = -t_proj

    overlap = overlap_count(all_vals)
    names   = list(series_meta.keys())
    palette = plt.cm.tab10(np.linspace(0, 1, len(names)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    # ── Top: projection coloured by overlap count ─────────────────────────
    ax0 = axes[0]
    # Draw projection
    ax0.plot(test_years, t_proj, lw=1.2, color="black", alpha=0.7, zorder=5)
    # Shade background by overlap count
    max_ov = int(overlap.max())
    cmap   = plt.cm.Blues
    for j in range(len(test_years) - 1):
        ov = int(overlap[j])
        if ov > 0:
            ax0.axvspan(test_years[j], test_years[j+1],
                        alpha=0.06 * ov, color=cmap(0.3 + 0.07 * ov))

    # Mark each series' range with a coloured tick at the top
    y_max = t_proj.max()
    for k, name in enumerate(sorted(names, key=lambda n: series_meta[n][0])):
        s_start, s_end = series_meta[name]
        vals = all_vals[name]
        covered = test_years[~np.isnan(vals)]
        if len(covered) == 0:
            continue
        ax0.axvspan(covered[0], covered[-1],
                    ymin=0.96 - k * 0.035, ymax=0.97 - k * 0.035,
                    color=palette[names.index(name)], alpha=0.9,
                    transform=ax0.get_xaxis_transform(), clip_on=False)

    ax0.set_ylabel("PC1 projection (temporal axis)", fontsize=8)
    ax0.set_xlim(test_years[0], test_years[-1])
    ax0.grid(True, alpha=0.2)
    ax0.set_title(
        f"Temporal axis (L{best_layer}) projection — {MODEL_NAMES[model_key]}\n"
        f"Background intensity ∝ number of active series; "
        f"coloured strips = per-series coverage",
        fontsize=9)

    # ── Bottom: overlap count ─────────────────────────────────────────────
    ax1 = axes[1]
    ax1.fill_between(test_years, overlap, alpha=0.55, color="#4575b4")
    ax1.plot(test_years, overlap, lw=1, color="#4575b4")
    ax1.set_ylabel("# series", fontsize=8)
    ax1.set_xlabel("Year (negative = BCE)", fontsize=8)
    ax1.set_xlim(test_years[0], test_years[-1])
    ax1.set_ylim(0, len(names) + 0.5)
    ax1.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "macberth"], default="bert")
    args      = parser.parse_args()
    model_key = args.model

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model(model_key, device)

    # ── Load series, determine combined year range ────────────────────────
    print(f"Loading series from {TIME_SERIES_DIR} …")
    series_dfs:  dict[str, pd.DataFrame]        = {}
    series_meta: dict[str, tuple[float, float]] = {}  # name -> (start, end)

    for path in sorted(TIME_SERIES_DIR.glob("*.csv")):
        if path.name in SKIP:
            continue
        try:
            df = load_series(path)
            s, e = series_natural_range(df)
            series_dfs[path.stem]  = df
            series_meta[path.stem] = (s, e)
            print(f"  {path.stem:<35} {int(s):>7} – {int(e):<7}")
        except Exception as exc:
            print(f"  {path.stem:<35} ERROR: {exc}")

    # Global test year range: cap at 1000 BCE floor to avoid pre-history noise
    global_start = max(-1000.0, min(s for s, _ in series_meta.values()))
    global_end   = min(2025.0,  max(e for _, e in series_meta.values()))
    test_years   = np.arange(
        np.floor(global_start / YEAR_STEP) * YEAR_STEP,
        np.ceil (global_end   / YEAR_STEP) * YEAR_STEP + YEAR_STEP,
        YEAR_STEP, dtype=float
    )
    print(f"\nTest years: {int(test_years[0])} – {int(test_years[-1])}  "
          f"(step={YEAR_STEP}, n={len(test_years)})\n")

    # Build per-series ordinal arrays (NaN outside own range)
    all_vals: dict[str, np.ndarray] = {}
    for name, df in series_dfs.items():
        s, e = series_meta[name]
        vals = series_ordinals(df, test_years, s, e)
        n_valid = int((~np.isnan(vals)).sum())
        if n_valid >= 5:
            all_vals[name] = vals
        else:
            print(f"  {name}: SKIP (n_valid={n_valid})")
    print(f"{len(all_vals)} series retained.\n")

    # ── Embed all years × all layers ──────────────────────────────────────
    layer_embs = embed_all_layers(model, tokenizer, device, test_years)
    n_layers   = len(layer_embs)

    # ── Layer sweep ───────────────────────────────────────────────────────
    print("Sweeping layers …")
    rows = []
    for layer in range(n_layers):
        X      = layer_embs[layer]
        t_axis = temporal_axis(X)
        t_proj = X @ t_axis
        if spearmanr(t_proj, test_years)[0] < 0:
            t_proj = -t_proj

        for name, vals in all_vals.items():
            rho, p, n = correlate(t_proj, vals)
            rows.append({"layer": layer, "series": name,
                         "rho": rho, "p": p, "n": n})

    df_all = pd.DataFrame(rows)

    # ── Console summaries ─────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"BEST LAYER PER SERIES — {MODEL_NAMES[model_key]}")
    print("=" * 65)
    for name in sorted(all_vals):
        sub  = df_all[df_all["series"] == name].dropna(subset=["rho"])
        best = sub.loc[sub["rho"].abs().idxmax()]
        star = sig_label(best["p"])
        print(f"  {name:<35} L{int(best['layer']):<2}  "
              f"ρ={best['rho']:+.3f}{star}  (n={int(best['n'])})")

    print()
    print("=" * 65)
    print(f"MEAN |ρ| PER LAYER — {MODEL_NAMES[model_key]}")
    print("=" * 65)
    mean_rho = (df_all.groupby("layer")["rho"]
                .apply(lambda x: np.nanmean(np.abs(x)))
                .sort_values(ascending=False))
    best_layer = int(mean_rho.idxmax())
    for layer, mrho in mean_rho.items():
        marker = " ←" if layer == best_layer else ""
        print(f"  L{layer:<2}  mean|ρ|={mrho:.4f}{marker}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_csv = OUT_DIR / f"battery_layer_sweep_{model_key}.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Pivot for heatmap (sort rows by series start year) ────────────────
    start_order = sorted(all_vals.keys(),
                         key=lambda n: series_meta[n][0], reverse=True)
    pivot_rho = df_all.pivot(index="series", columns="layer", values="rho").loc[start_order]
    pivot_p   = df_all.pivot(index="series", columns="layer", values="p"  ).loc[start_order]
    pivot_n   = df_all.pivot(index="series", columns="layer", values="n"  ).loc[start_order]

    # ── Plots ─────────────────────────────────────────────────────────────
    fig = plot_coverage(series_meta, all_vals, test_years, model_key)
    p   = OUT_DIR / f"battery_coverage_{model_key}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Coverage → {p}")

    fig = plot_layer_sweep(pivot_rho, pivot_p, pivot_n, model_key)
    p   = OUT_DIR / f"battery_layer_sweep_{model_key}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Heatmap  → {p}")

    fig = plot_mean_rho(df_all, model_key)
    p   = OUT_DIR / f"battery_mean_rho_{model_key}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Mean ρ   → {p}")

    fig = plot_projection(layer_embs, all_vals, series_meta,
                          test_years, best_layer, model_key)
    p   = OUT_DIR / f"battery_projection_{model_key}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Proj     → {p}")


if __name__ == "__main__":
    main()
