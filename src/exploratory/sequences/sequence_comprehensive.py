"""sequence_comprehensive.py — Full multi-figure analysis of categorical
temporal sequences in BERT/MacBERTh embeddings.

Generates 9 figures:
  Fig 1  cross_axis        — cosine alignment + cross-projection ρ per layer
  Fig 2  metric_lines      — 4 metric line plots across layers (all sequences)
  Fig 3  metric_heatmaps   — sequence × layer heatmap per metric
  Fig 4  pca_L04           — PCA scatter at L04, all 8 sequences
  Fig 5  pca_best          — PCA scatter at best layer (by BW), all 8 sequences
  Fig 6  centroid_dist     — embedding dist vs temporal dist per sequence
  Fig 7  confusion         — cross-period confusion matrices at best-BW layer
  Fig 8  template_robust   — within-period template variance per layer
  Fig 9  composite         — per-sequence composite score spider chart

Usage
-----
    python3 src/exploratory/sequence_comprehensive.py --model bert
    python3 src/exploratory/sequence_comprehensive.py --model macberth
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR      = PROJECT_ROOT / "data" / "outputs" / "sequences"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_utils import (  # noqa: E402
    MODEL_NAMES, load_embedder,
    bw_ratio, centroid_tau, knn_purity,
)

# Year-carrier templates for computing the year direction
YEAR_CARRIER = [
    "the year is {y} .",
    "it is the year {y} .",
    "this was written in {y} .",
    "the date is {y} .",
]
# Classical Chinese year-carrier templates for SikuBERT
YEAR_CARRIER_ZH = [
    "时在{y}年。",        # Time is in year {y}.
    "当{y}年之时。",      # At the time of year {y}.
    "此乃{y}年也。",      # This is year {y}.
    "今{y}年是也。",      # Year {y} it is.
]
YEAR_RANGE = np.arange(1200, 1931, 25, dtype=float)   # 30 anchor years


# ---------------------------------------------------------------------------
# Sequence definitions
# ---------------------------------------------------------------------------

class Period(NamedTuple):
    label: str
    start: int
    end:   int


def _sample_years(start: int, end: int, n: int = 6) -> list[int]:
    if end - start < n:
        return list(range(start, end + 1))
    return [int(round(start + i * (end - start) / (n - 1))) for i in range(n)]


SEQUENCES: dict[str, dict] = {

    "country_name": {
        "title": "Country name", "subtitle": "strict: 1707, 1801",
        "boundary_type": "strict", "monotonic": True,
        "transitions": [1707, 1801],
        "periods": [
            Period("Kingdom of England",       1600, 1706),
            Period("Kingdom of Great Britain", 1707, 1800),
            Period("United Kingdom",           1801, 1900),
        ],
        "templates": [
            "the country was known as {label} .",
            "england was then called {label} .",
            "the nation of {label} was ruled by a monarch .",
            "the official name of the realm was {label} .",
        ],
    },

    "state_religion": {
        "title": "State religion", "subtitle": "strict: 1534, 1553, 1558",
        "boundary_type": "strict", "monotonic": False,
        "transitions": [1534, 1553, 1558],
        "periods": [
            Period("Roman Catholic",    1400, 1533),
            Period("Church of England", 1534, 1552),
            Period("Roman Catholic",    1553, 1557),
            Period("Church of England", 1558, 1700),
        ],
        "templates": [
            "the religion of the land was {label} .",
            "the state religion was {label} .",
            "the church followed {label} teachings .",
            "worship in england followed {label} traditions .",
        ],
    },

    "form_of_government": {
        "title": "Form of government", "subtitle": "strict: 1649, 1660",
        "boundary_type": "strict", "monotonic": False,
        "transitions": [1649, 1660],
        "periods": [
            Period("monarchy",     1500, 1648),
            Period("commonwealth", 1649, 1659),
            Period("monarchy",     1660, 1800),
        ],
        "templates": [
            "england was governed as a {label} .",
            "the system of rule was a {label} .",
            "the form of government was {label} .",
            "power was held through a {label} .",
        ],
    },

    "ruling_dynasty": {
        "title": "Ruling dynasty", "subtitle": "strict: accession years",
        "boundary_type": "strict", "monotonic": True,
        "transitions": [1399, 1461, 1485, 1603, 1714, 1837, 1901],
        "periods": [
            Period("Plantagenet", 1200, 1398),
            Period("Lancaster",   1399, 1460),
            Period("York",        1461, 1484),
            Period("Tudor",       1485, 1602),
            Period("Stuart",      1603, 1713),
            Period("Hanover",     1714, 1836),
            Period("Windsor",     1837, 1936),
        ],
        "templates": [
            "the {label} dynasty ruled england .",
            "the house of {label} held the throne .",
            "england was ruled by the {label} family .",
            "the {label} monarch sat on the english throne .",
        ],
    },

    "calendar": {
        "title": "Calendar system", "subtitle": "strict: 1752",
        "boundary_type": "strict", "monotonic": True,
        "transitions": [1752],
        "periods": [
            Period("Julian calendar",    1500, 1751),
            Period("Gregorian calendar", 1752, 1900),
        ],
        "templates": [
            "dates were recorded in the {label} .",
            "the {label} was used to track time .",
            "england used the {label} for official records .",
            "the year was measured by the {label} .",
        ],
    },

    "primary_weapon": {
        "title": "Primary weapon", "subtitle": "approx: ~1500, ~1650, ~1850",
        "boundary_type": "approximate", "monotonic": True,
        "transitions": [1500, 1650, 1850],
        "periods": [
            Period("longbow", 1200, 1499),
            Period("pike",    1500, 1649),
            Period("musket",  1650, 1849),
            Period("rifle",   1850, 1930),
        ],
        "templates": [
            "the soldier carried a {label} into battle .",
            "troops were armed with the {label} .",
            "the primary weapon of the infantry was the {label} .",
            "soldiers fought with the {label} .",
        ],
    },

    "ship_construction": {
        "title": "Ship construction", "subtitle": "approx: ~1820, ~1870",
        "boundary_type": "approximate", "monotonic": True,
        "transitions": [1820, 1870],
        "periods": [
            Period("wooden ship", 1200, 1819),
            Period("iron ship",   1820, 1869),
            Period("steel ship",  1870, 1930),
        ],
        "templates": [
            "the navy sailed in a {label} .",
            "the vessel was a {label} .",
            "the fleet consisted of {label}s .",
            "the warship was a {label} .",
        ],
    },

    "primary_fuel": {
        "title": "Primary fuel", "subtitle": "approx: ~1700, ~1820",
        "boundary_type": "approximate", "monotonic": True,
        "transitions": [1700, 1820],
        "periods": [
            Period("wood and peat", 1200, 1699),
            Period("coal",          1700, 1819),
            Period("steam coal",    1820, 1930),
        ],
        "templates": [
            "homes were heated with {label} .",
            "the furnace burned {label} .",
            "industry relied on {label} for energy .",
            "heat was produced by burning {label} .",
        ],
    },
}

SEQ_ORDER = list(SEQUENCES.keys())


# ---------------------------------------------------------------------------
# Year-direction computation
# ---------------------------------------------------------------------------

def compute_year_directions(embed_fn, n_layers: int,
                            year_templates: list[str]) -> dict[int, np.ndarray]:
    """PC1 of year-carrier embeddings per layer → {layer: unit_vector}."""
    accum = {l: [] for l in range(n_layers)}

    print("  Computing year directions …", flush=True)
    for year in YEAR_RANGE:
        for tmpl in year_templates:
            sent = tmpl.format(y=int(year))
            vecs = embed_fn(sent)
            for l in range(n_layers):
                accum[l].append(vecs[l])

    year_dirs = {}
    for l in range(n_layers):
        X  = np.stack(accum[l])         # (n_year_samples, H)
        Xc = X - X.mean(axis=0)
        pca = PCA(n_components=1).fit(Xc)
        v   = pca.components_[0]
        # Orient: should correlate positively with year
        years_rep = np.repeat(YEAR_RANGE, len(YEAR_CARRIER))
        if np.corrcoef(Xc @ v, years_rep)[0, 1] < 0:
            v = -v
        year_dirs[l] = v
    return year_dirs


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(seq_def: dict, embed_fn, n_per_period: int = 6):
    """Embed all sentences for one sequence.
    Returns:
        embs       — (N, n_layers, H)
        years      — (N,)
        period_ids — (N,)
        tmpl_ids   — (N,)  which template (for robustness analysis)
        periods    — list[Period]
    """
    periods   = seq_def["periods"]
    templates = seq_def["templates"]
    all_embs, all_years, all_pids, all_tids = [], [], [], []

    for p_idx, p in enumerate(periods):
        mid = (p.start + p.end) / 2
        for t_idx, tmpl in enumerate(templates):
            sent = tmpl.format(label=p.label)
            vecs = embed_fn(sent)
            all_embs.append(vecs)
            all_years.append(mid)
            all_pids.append(p_idx)
            all_tids.append(t_idx)

    return (
        np.stack(all_embs),          # (N, L, H)
        np.array(all_years),
        np.array(all_pids, dtype=int),
        np.array(all_tids, dtype=int),
        periods,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(Xs: np.ndarray, pids: np.ndarray) -> np.ndarray:
    n_p  = int(pids.max()) + 1
    conf = np.zeros((n_p, n_p))
    for i in range(len(Xs)):
        own  = pids[i]
        mask = pids != own
        if not mask.any():
            continue
        nn = pids[mask][np.argmin(np.linalg.norm(Xs[mask] - Xs[i], axis=1))]
        conf[own, nn] += 1
    s = conf.sum(1, keepdims=True); s[s == 0] = 1
    return conf / s


def template_robustness(embs: np.ndarray, pids: np.ndarray,
                        tids: np.ndarray, layer: int) -> float:
    """Mean intra-period, across-template std (averaged across periods)."""
    X = embs[:, layer, :]
    stds = []
    for p in np.unique(pids):
        pm = pids == p
        if pm.sum() < 2:
            continue
        stds.append(np.linalg.norm(X[pm].std(axis=0)))
    return float(np.mean(stds)) if stds else 0.0


def cross_axis_metrics(embs: np.ndarray, years: np.ndarray,
                       pids: np.ndarray, layer: int,
                       year_dir: np.ndarray) -> tuple[float, float]:
    """
    Returns:
        cosine_align — |cosine(seq_PC1, year_dir)|
        cross_rho    — Spearman ρ of (X @ year_dir) vs mid-year
    """
    X   = embs[:, layer, :]
    Xs  = StandardScaler().fit_transform(X)
    # Sequence direction: PC1 of sequence embeddings
    pca = PCA(n_components=1).fit(Xs)
    seq_dir = pca.components_[0]
    # Cosine alignment
    cos = abs(float(np.dot(seq_dir, year_dir) /
                    (np.linalg.norm(seq_dir) * np.linalg.norm(year_dir) + 1e-12)))
    # Cross-projection ρ
    proj = X @ year_dir
    rho, _ = stats.spearmanr(proj, years)
    return cos, float(rho)


# ---------------------------------------------------------------------------
# Run all sequences
# ---------------------------------------------------------------------------

def run_all(embed_fn, n_layers: int,
            year_templates: list[str]) -> tuple[pd.DataFrame, dict, dict[int, np.ndarray]]:
    year_dirs = compute_year_directions(embed_fn, n_layers, year_templates)

    all_rows = []
    aux      = {}   # seq_name -> {embs, years, pids, tids, periods, confusion}

    for seq_name, seq_def in SEQUENCES.items():
        print(f"  [{seq_name}] {seq_def['title']} …", flush=True)
        embs, years, pids, tids, periods = build_dataset(seq_def, embed_fn)

        seq_rows = []
        for layer in range(n_layers):
            X  = embs[:, layer, :]
            Xs = StandardScaler().fit_transform(X)
            pca  = PCA(n_components=1).fit(Xs)
            pc1  = pca.fit_transform(Xs)[:, 0]
            rho, _ = stats.spearmanr(pc1, years)
            bw  = bw_ratio(Xs, pids)
            unique = np.unique(pids)
            cents_tau = np.stack([Xs[pids == u].mean(0) for u in unique])
            mids_tau  = np.array([(periods[u].start + periods[u].end) / 2
                                  for u in unique])
            tau = centroid_tau(cents_tau, mids_tau)
            knn = knn_purity(Xs, pids)
            rob = template_robustness(embs, pids, tids, layer)
            cos, xrho = cross_axis_metrics(embs, years, pids, layer, year_dirs[layer])
            seq_rows.append({
                "sequence": seq_name, "layer": layer,
                "spearman_rho": abs(rho), "bw_ratio": bw,
                "centroid_tau": tau, "knn_purity": knn,
                "tmpl_robustness": rob,
                "cross_cosine": cos, "cross_rho": xrho,
            })

        df_seq = pd.DataFrame(seq_rows)
        all_rows.append(df_seq)

        # Confusion at best-BW layer
        best_bw_layer = int(df_seq["bw_ratio"].idxmax())
        Xs_best = StandardScaler().fit_transform(embs[:, best_bw_layer, :])
        conf    = confusion_matrix(Xs_best, pids)

        aux[seq_name] = {
            "embs": embs, "years": years, "pids": pids, "tids": tids,
            "periods": periods, "confusion": conf,
            "best_bw_layer": best_bw_layer,
        }

    return pd.concat(all_rows, ignore_index=True), aux, year_dirs


# ---------------------------------------------------------------------------
# Helper: extract metric series per sequence per layer
# ---------------------------------------------------------------------------

def get_vals(results: pd.DataFrame, seq: str, col: str) -> list[float]:
    sub = results[results["sequence"] == seq].sort_values("layer")
    return sub[col].tolist()


def get_layers(results: pd.DataFrame) -> list[int]:
    return sorted(results["layer"].unique().tolist())


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def seq_colors() -> dict[str, str]:
    cmap = plt.cm.tab10
    return {s: cmap(i / (len(SEQ_ORDER) - 1)) for i, s in enumerate(SEQ_ORDER)}


def seq_ls(seq_name: str) -> str:
    return "-o" if SEQUENCES[seq_name]["boundary_type"] == "strict" else ":s"


# ---------------------------------------------------------------------------
# Figure 1 — Cross-axis alignment
# ---------------------------------------------------------------------------

def fig_cross_axis(results: pd.DataFrame, model_key: str) -> None:
    layers = get_layers(results)
    colors = seq_colors()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, col, ylabel, title in [
        (axes[0], "cross_cosine",
         "|cosine(seq_dir, year_dir)|",
         "Cosine alignment — sequence PC1 vs year direction"),
        (axes[1], "cross_rho",
         "Spearman ρ",
         "Cross-projection ρ — seq embeddings onto year direction"),
    ]:
        for seq in SEQ_ORDER:
            vals = get_vals(results, seq, col)
            mono = SEQUENCES[seq]["monotonic"]
            lbl  = f"{'[S]' if SEQUENCES[seq]['boundary_type']=='strict' else '[A]'} " \
                   f"{SEQUENCES[seq]['title']}" + ("" if mono else " ⚠")
            ax.plot(layers, vals, seq_ls(seq), color=colors[seq],
                    markersize=4, lw=1.6, label=lbl)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvspan(4.5, 5.5, alpha=0.08, color="gold")
        ax.axvspan(6.5, 7.5, alpha=0.08, color="lightblue")
        ax.set_xlabel("Layer (0 = embedding)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
        ax.grid(linestyle=":", alpha=0.35)
        ax.legend(fontsize=6.5, ncol=1, loc="best")

    fig.suptitle(f"Cross-axis temporal alignment by layer — {MODEL_NAMES[model_key]}\n"
                 "[S]=strict  [A]=approx  ⚠=non-monotonic  "
                 "Gold=L5 zone  Blue=L7 zone", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"comp_cross_axis_{model_key}.png")


# ---------------------------------------------------------------------------
# Figure 2 — Metric line plots (all 4 metrics)
# ---------------------------------------------------------------------------

def fig_metric_lines(results: pd.DataFrame, model_key: str) -> None:
    layers = get_layers(results)
    colors = seq_colors()
    metrics = [
        ("spearman_rho",    "|Spearman ρ| (PC1 vs year)"),
        ("bw_ratio",        "BW ratio (between/within variance)"),
        ("centroid_tau",    "Centroid Kendall τ"),
        ("knn_purity",      "Cross-period kNN purity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, metrics):
        for seq in SEQ_ORDER:
            sub  = results[results["sequence"] == seq].sort_values("layer")
            vals = sub[col].abs().tolist() if col in ("centroid_tau",) else sub[col].tolist()
            mono = SEQUENCES[seq]["monotonic"]
            lbl  = f"{'[S]' if SEQUENCES[seq]['boundary_type']=='strict' else '[A]'} " \
                   f"{SEQUENCES[seq]['title']}" + ("" if mono else " ⚠")
            ax.plot(layers, vals, seq_ls(seq), color=colors[seq],
                    markersize=4, lw=1.6, label=lbl)
        ax.axvspan(4.5, 5.5, alpha=0.08, color="gold")
        ax.axvspan(6.5, 7.5, alpha=0.08, color="lightblue")
        ax.set_xlabel("Layer", fontsize=9); ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
        ax.grid(linestyle=":", alpha=0.35)
        ax.legend(fontsize=6.5, ncol=2, loc="best")
    fig.suptitle(f"{MODEL_NAMES[model_key]} — temporal geometry metrics by layer\n"
                 "[S]=strict  [A]=approx  ⚠=non-monotonic", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f"comp_metric_lines_{model_key}.png")


# ---------------------------------------------------------------------------
# Figure 3 — Metric heatmaps (sequence × layer)
# ---------------------------------------------------------------------------

def fig_metric_heatmaps(results: pd.DataFrame, model_key: str) -> None:
    layers = get_layers(results)
    metrics = [
        ("spearman_rho", "Spearman |ρ|",  "RdBu_r",  0, 1),
        ("bw_ratio",     "BW ratio",       "YlOrRd",  0, None),
        ("centroid_tau", "|Kendall τ|",    "RdBu_r",  0, 1),
        ("knn_purity",   "kNN purity",     "YlOrRd",  0, 1),
        ("cross_cosine", "|cos align|",    "YlOrRd",  0, 1),
        ("cross_rho",    "Cross ρ (signed)","RdBu_r", -1, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    axes = axes.flatten()
    seq_labels = [f"{'[S]' if SEQUENCES[s]['boundary_type']=='strict' else '[A]'} "
                  f"{SEQUENCES[s]['title']}" for s in SEQ_ORDER]

    for ax, (col, label, cmap, vmin, vmax) in zip(axes, metrics):
        mat = np.array([[
            float(results[(results["sequence"] == s) &
                          (results["layer"] == l)][col].iloc[0])
            if len(results[(results["sequence"] == s) &
                           (results["layer"] == l)]) else np.nan
            for l in layers] for s in SEQ_ORDER])
        # Take abs for tau
        if col == "centroid_tau":
            mat = np.abs(mat)
        vmax_ = vmax if vmax is not None else np.nanmax(mat)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax_)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
        ax.set_yticks(range(len(SEQ_ORDER)))
        ax.set_yticklabels(seq_labels, fontsize=7)
        ax.set_title(label, fontsize=9, fontweight="bold")
        for i in range(len(SEQ_ORDER)):
            for j in range(len(layers)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if (col != "cross_rho" and v > 0.6 * vmax_)
                                         else "black")
    fig.suptitle(f"{MODEL_NAMES[model_key]} — metric heatmaps: sequence × layer\n"
                 "(rows sorted: strict first, then approx)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"comp_heatmaps_{model_key}.png")


# ---------------------------------------------------------------------------
# Figure 4 & 5 — PCA scatter at L04 and best layer
# ---------------------------------------------------------------------------

def _pca_scatter_grid(aux: dict, results: pd.DataFrame,
                      layer_choice: int | str,
                      model_key: str, fig_tag: str, title_suffix: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    axes = axes.flatten()
    cmap_p = plt.cm.viridis

    for ax, seq_name in zip(axes, SEQ_ORDER):
        a = aux[seq_name]
        embs, pids, periods = a["embs"], a["pids"], a["periods"]

        layer = (a["best_bw_layer"] if layer_choice == "best"
                 else int(layer_choice))

        X  = embs[:, layer, :]
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2).fit(Xs)
        C   = pca.transform(Xs)

        n_per = int(pids.max()) + 1
        cols  = [cmap_p(i / max(n_per - 1, 1)) for i in range(n_per)]
        for p_idx in range(n_per):
            m = pids == p_idx
            ax.scatter(C[m, 0], C[m, 1], color=cols[p_idx],
                       alpha=0.7, s=22, edgecolors="white", linewidth=0.3,
                       label=periods[p_idx].label[:12])
            cx, cy = C[m, 0].mean(), C[m, 1].mean()
            ax.scatter(cx, cy, color=cols[p_idx], s=120, marker="*",
                       edgecolors="black", linewidth=0.8)

        ev = pca.explained_variance_ratio_
        bw = float(results[(results["sequence"] == seq_name) &
                           (results["layer"] == layer)]["bw_ratio"].iloc[0])
        mono = SEQUENCES[seq_name]["monotonic"]
        ax.set_title(
            f"{SEQUENCES[seq_name]['title']} @ L{layer}\n"
            f"BW={bw:.2f}  {'mono' if mono else '⚠non-mono'}",
            fontsize=8)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=7)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=7)
        ax.legend(fontsize=5.5, loc="best")
        ax.grid(linestyle=":", alpha=0.3)

    fig.suptitle(f"{MODEL_NAMES[model_key]} — PCA scatter {title_suffix}\n"
                 "Stars = period centroids  ·  colour = temporal order (early→dark)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"comp_{fig_tag}_{model_key}.png")


def fig_pca_L04(aux: dict, results: pd.DataFrame, model_key: str) -> None:
    _pca_scatter_grid(aux, results, 4, model_key, "pca_L04", "at L04")


def fig_pca_best(aux: dict, results: pd.DataFrame, model_key: str) -> None:
    _pca_scatter_grid(aux, results, "best", model_key, "pca_best",
                      "at best-BW layer per sequence")


# ---------------------------------------------------------------------------
# Figure 6 — Embedding distance vs temporal distance
# ---------------------------------------------------------------------------

def fig_centroid_dist(aux: dict, results: pd.DataFrame, model_key: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for ax, seq_name in zip(axes, SEQ_ORDER):
        a       = aux[seq_name]
        embs, pids, periods = a["embs"], a["pids"], a["periods"]
        best_l  = a["best_bw_layer"]

        X  = StandardScaler().fit_transform(embs[:, best_l, :])
        unique = np.unique(pids)
        cents  = np.stack([X[pids == u].mean(0) for u in unique])
        mids   = np.array([(periods[u].start + periods[u].end) / 2 for u in unique])

        n   = len(unique)
        ed, td, labels = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                ed.append(np.linalg.norm(cents[i] - cents[j]))
                td.append(abs(mids[i] - mids[j]))
                labels.append(f"{periods[unique[i]].label[:6]}–{periods[unique[j]].label[:6]}")

        if len(ed) < 2:
            ax.set_title(seq_name); ax.set_visible(False); continue

        ed, td = np.array(ed), np.array(td)
        ax.scatter(td, ed, s=35, color=seq_colors()[seq_name], alpha=0.8, edgecolors="k", lw=0.4)
        for lbl, x, y in zip(labels, td, ed):
            ax.annotate(lbl, (x, y), fontsize=5, xytext=(3, 2),
                        textcoords="offset points", alpha=0.7)

        # Fit line
        if len(ed) >= 3:
            m, b, r, p, _ = stats.linregress(td, ed)
            xfit = np.linspace(td.min(), td.max(), 100)
            ax.plot(xfit, m * xfit + b, "r--", lw=1.2, alpha=0.7,
                    label=f"r={r:.2f} p={p:.3f}")
            ax.legend(fontsize=6)

        tau_val = float(results[(results["sequence"] == seq_name) &
                                (results["layer"] == best_l)]["centroid_tau"].iloc[0])
        ax.set_title(f"{SEQUENCES[seq_name]['title']} @ L{best_l}\nτ={tau_val:+.2f}",
                     fontsize=8)
        ax.set_xlabel("Temporal distance (years)", fontsize=7)
        ax.set_ylabel("Embedding distance (Euclidean)", fontsize=7)
        ax.grid(linestyle=":", alpha=0.3)

    fig.suptitle(f"{MODEL_NAMES[model_key]} — Centroid embedding dist vs temporal dist\n"
                 "Pairwise period centroids at best-BW layer  ·  r = Pearson correlation",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"comp_centroid_dist_{model_key}.png")


# ---------------------------------------------------------------------------
# Figure 7 — Confusion matrices
# ---------------------------------------------------------------------------

def fig_confusion(aux: dict, results: pd.DataFrame, model_key: str) -> None:
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 12))
    axes = axes.flatten()

    for ax, seq_name in zip(axes, SEQ_ORDER):
        a      = aux[seq_name]
        conf   = a["confusion"]
        periods= a["periods"]
        best_l = a["best_bw_layer"]
        labels = [p.label[:12] for p in periods]

        im = ax.imshow(conf, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right",
                           fontsize=7 if len(periods) <= 5 else 5.5)
        ax.set_yticklabels(labels,
                           fontsize=7 if len(periods) <= 5 else 5.5)
        for i in range(len(periods)):
            for j in range(len(periods)):
                if i == j:
                    continue
                if conf[i, j] > 0.02:
                    ax.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if conf[i, j] > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        bw = float(results[(results["sequence"] == seq_name) &
                           (results["layer"] == best_l)]["bw_ratio"].iloc[0])
        mono = SEQUENCES[seq_name]["monotonic"]
        ax.set_title(f"{SEQUENCES[seq_name]['title']} @ L{best_l}\n"
                     f"BW={bw:.2f}  {'⚠non-mono' if not mono else ''}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Nearest cross-period neighbour", fontsize=7)
        ax.set_ylabel("True period", fontsize=7)

    fig.suptitle(f"{MODEL_NAMES[model_key]} — Period confusion matrices at best-BW layer\n"
                 "Row = true period  ·  Col = nearest cross-period neighbour  ·  Diagonal suppressed",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"comp_confusion_{model_key}.png")


# ---------------------------------------------------------------------------
# Figure 8 — Template robustness across layers
# ---------------------------------------------------------------------------

def fig_template_robust(results: pd.DataFrame, model_key: str) -> None:
    layers = get_layers(results)
    colors = seq_colors()
    fig, ax = plt.subplots(figsize=(12, 6))

    for seq in SEQ_ORDER:
        vals = get_vals(results, seq, "tmpl_robustness")
        mono = SEQUENCES[seq]["monotonic"]
        lbl  = f"{'[S]' if SEQUENCES[seq]['boundary_type']=='strict' else '[A]'} " \
               f"{SEQUENCES[seq]['title']}" + ("" if mono else " ⚠")
        ax.plot(layers, vals, seq_ls(seq), color=colors[seq],
                markersize=4, lw=1.6, label=lbl)

    ax.axvspan(4.5, 5.5, alpha=0.08, color="gold")
    ax.axvspan(6.5, 7.5, alpha=0.08, color="lightblue")
    ax.set_xlabel("Layer (0 = embedding)", fontsize=9)
    ax.set_ylabel("Mean within-period template std (L2 norm)", fontsize=9)
    ax.set_title(
        f"{MODEL_NAMES[model_key]} — Template robustness by layer\n"
        "Higher = more variance across sentence templates within the same period\n"
        "(lower is better: representation stable regardless of phrasing)",
        fontsize=9)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    _save(fig, f"comp_template_robust_{model_key}.png")


# ---------------------------------------------------------------------------
# Figure 9 — Composite spider chart
# ---------------------------------------------------------------------------

def fig_composite(results: pd.DataFrame, model_key: str) -> None:
    """Radar chart: each sequence is one ring; 5 metrics at best layer."""
    metric_cols  = ["spearman_rho", "bw_ratio", "centroid_tau", "knn_purity", "cross_cosine"]
    metric_names = ["Spearman |ρ|", "BW ratio\n(norm)", "Centroid |τ|",
                    "kNN purity", "Cross\nalignment"]

    # Normalise each metric to [0,1] across all sequences
    best_per_seq = {}
    for seq in SEQ_ORDER:
        sub = results[results["sequence"] == seq]
        vals = {}
        for col in metric_cols:
            v = sub[col].abs().max()
            vals[col] = float(v) if not np.isnan(v) else 0.0
        best_per_seq[seq] = vals

    # Normalise BW ratio by its max across sequences
    max_bw = max(best_per_seq[s]["bw_ratio"] for s in SEQ_ORDER)
    for s in SEQ_ORDER:
        best_per_seq[s]["bw_ratio"] /= (max_bw + 1e-9)

    N    = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close

    colors = seq_colors()
    n_seqs = len(SEQ_ORDER)
    ncols  = 4
    nrows  = (n_seqs + ncols - 1) // ncols
    fig    = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for idx, seq in enumerate(SEQ_ORDER):
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)
        vals_list = [best_per_seq[seq][c] for c in metric_cols]
        vals_list += vals_list[:1]
        ax.plot(angles, vals_list, "o-", lw=2, color=colors[seq])
        ax.fill(angles, vals_list, alpha=0.2, color=colors[seq])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=5)
        mono = SEQUENCES[seq]["monotonic"]
        btype = "S" if SEQUENCES[seq]["boundary_type"] == "strict" else "A"
        ax.set_title(
            f"[{btype}] {SEQUENCES[seq]['title']}" + ("" if mono else " ⚠"),
            fontsize=8, fontweight="bold", pad=12)

    fig.suptitle(
        f"{MODEL_NAMES[model_key]} — Composite metric spider chart\n"
        "Values = best across all layers  ·  BW normalised by max across sequences",
        fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"comp_spider_{model_key}.png")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=["bert", "macberth", "sikubert", "openai"],
                        default="bert")
    args      = parser.parse_args()
    model_key = args.model

    embed_fn, n_layers = load_embedder(model_key)

    # Use classical Chinese year-carrier templates for Chinese-corpus models
    year_tmpls = YEAR_CARRIER_ZH if model_key == "sikubert" else YEAR_CARRIER

    print("Running sequences …")
    results, aux, year_dirs = run_all(embed_fn, n_layers, year_tmpls)

    csv_path = OUT_DIR / f"comp_results_{model_key}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}\n")

    print("Generating figures …")
    # Layer-sweep figures only make sense for multi-layer models
    if n_layers > 1:
        fig_cross_axis(results, model_key)
        fig_metric_lines(results, model_key)
        fig_metric_heatmaps(results, model_key)
        fig_template_robust(results, model_key)
    if n_layers > 4:
        fig_pca_L04(aux, results, model_key)
    fig_pca_best(aux, results, model_key)
    fig_centroid_dist(aux, results, model_key)
    fig_confusion(aux, results, model_key)
    fig_composite(results, model_key)
    print("\nDone.")


if __name__ == "__main__":
    main()
