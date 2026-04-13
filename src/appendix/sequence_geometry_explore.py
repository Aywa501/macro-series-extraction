"""Exploratory geometric analysis of BERT embeddings for historical sequences.

Five analyses:
  1. Cross-sequence co-temporal clustering at L04
  2. Distance-time scatter per sequence at L04
  3. Layer trajectory of dynasty centroids (ruling_dynasty only)
  4. Between-sequence temporal direction alignment at L04
  5. Sentence-level variance decomposition for ruling_dynasty at L04

Saves plots to src/appendix/explore_*.png
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Sequence definitions (copied from sequence_geometry_probe.py)
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
        "title": "Country name",
        "subtitle": "strict: 1707, 1801",
        "boundary_type": "strict",
        "transitions": [1707, 1801],
        "monotonic": True,
        "periods": [
            Period("Kingdom of England",        1600, 1706),
            Period("Kingdom of Great Britain",  1707, 1800),
            Period("United Kingdom",            1801, 1900),
        ],
        "templates": [
            "the country was known as {label} .",
            "england was then called {label} .",
            "the nation of {label} was ruled by a monarch .",
            "the official name of the realm was {label} .",
        ],
    },
    "state_religion": {
        "title": "State religion",
        "subtitle": "strict: 1534, 1553, 1558",
        "boundary_type": "strict",
        "monotonic": False,
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
        "title": "Form of government",
        "subtitle": "strict: 1649, 1660",
        "boundary_type": "strict",
        "monotonic": False,
        "transitions": [1649, 1660],
        "periods": [
            Period("monarchy",      1500, 1648),
            Period("commonwealth",  1649, 1659),
            Period("monarchy",      1660, 1800),
        ],
        "templates": [
            "england was governed as a {label} .",
            "the system of rule was a {label} .",
            "the form of government was {label} .",
            "power was held through a {label} .",
        ],
    },
    "ruling_dynasty": {
        "title": "Ruling dynasty",
        "subtitle": "strict: exact accession years",
        "boundary_type": "strict",
        "monotonic": True,
        "transitions": [1399, 1461, 1485, 1603, 1714, 1837, 1901],
        "periods": [
            Period("Plantagenet",  1200, 1398),
            Period("Lancaster",    1399, 1460),
            Period("York",         1461, 1484),
            Period("Tudor",        1485, 1602),
            Period("Stuart",       1603, 1713),
            Period("Hanover",      1714, 1836),
            Period("Windsor",      1837, 1936),
        ],
        "templates": [
            "the {label} dynasty ruled england .",
            "the house of {label} held the throne .",
            "england was ruled by the {label} family .",
            "the {label} monarch sat on the english throne .",
        ],
    },
    "calendar": {
        "title": "Calendar system",
        "subtitle": "strict: 1752",
        "boundary_type": "strict",
        "monotonic": True,
        "transitions": [1752],
        "periods": [
            Period("Julian calendar",     1500, 1751),
            Period("Gregorian calendar",  1752, 1900),
        ],
        "templates": [
            "dates were recorded in the {label} .",
            "the {label} was used to track time .",
            "england used the {label} for official records .",
            "the year was measured by the {label} .",
        ],
    },
    "primary_weapon": {
        "title": "Primary weapon",
        "subtitle": "approx: ~1500, ~1650, ~1850",
        "boundary_type": "approximate",
        "monotonic": True,
        "transitions": [1500, 1650, 1850],
        "periods": [
            Period("longbow",   1200, 1499),
            Period("pike",      1500, 1649),
            Period("musket",    1650, 1849),
            Period("rifle",     1850, 1930),
        ],
        "templates": [
            "the soldier carried a {label} into battle .",
            "troops were armed with the {label} .",
            "the primary weapon of the infantry was the {label} .",
            "soldiers fought with the {label} .",
        ],
    },
    "ship_construction": {
        "title": "Ship construction",
        "subtitle": "approx: ~1820, ~1870",
        "boundary_type": "approximate",
        "monotonic": True,
        "transitions": [1820, 1870],
        "periods": [
            Period("wooden ship",   1200, 1819),
            Period("iron ship",     1820, 1869),
            Period("steel ship",    1870, 1930),
        ],
        "templates": [
            "the navy sailed in a {label} .",
            "the vessel was a {label} .",
            "the fleet consisted of {label}s .",
            "the warship was a {label} .",
        ],
    },
    "primary_fuel": {
        "title": "Primary fuel",
        "subtitle": "approx: ~1700, ~1820",
        "boundary_type": "approximate",
        "monotonic": True,
        "transitions": [1700, 1820],
        "periods": [
            Period("wood and peat",  1200, 1699),
            Period("coal",           1700, 1819),
            Period("steam coal",     1820, 1930),
        ],
        "templates": [
            "homes were heated with {label} .",
            "the furnace burned {label} .",
            "industry relied on {label} for energy .",
            "heat was produced by burning {label} .",
        ],
    },
}

# ---------------------------------------------------------------------------
# Model loading and embedding
# ---------------------------------------------------------------------------

def load_model():
    from transformers import AutoModel, AutoTokenizer
    name = "bert-base-uncased"
    print(f"Loading {name} ...", flush=True)
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_hidden_states=True).to(device).eval()
    print("Loaded.\n", flush=True)
    return model, tokenizer, device


def embed_sentence(model, tokenizer, device, sentence: str) -> np.ndarray:
    """CLS embedding at all layers: shape (13, 768)."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    hidden = out.hidden_states
    cls = np.stack([h[0, 0, :].cpu().numpy() for h in hidden], axis=0)
    return cls  # (13, 768)


def build_dataset(seq_key: str, seq_def: dict, model, tokenizer, device,
                  n_years_per_period: int = 5):
    """
    Returns dict with:
      embeddings: (N, 13, 768)
      period_ids: (N,) int
      mid_years:  (N,) float  (mid-year of the period this sentence belongs to)
      labels:     (N,) str
      periods:    list of Period
    """
    periods = seq_def["periods"]
    templates = seq_def["templates"]

    all_embs, all_period_ids, all_mid_years, all_labels = [], [], [], []

    for p_idx, period in enumerate(periods):
        mid_year = (period.start + period.end) / 2.0
        for tmpl in templates:
            sentence = tmpl.format(label=period.label)
            emb = embed_sentence(model, tokenizer, device, sentence)
            all_embs.append(emb)
            all_period_ids.append(p_idx)
            all_mid_years.append(mid_year)
            all_labels.append(period.label)

    return {
        "embeddings": np.stack(all_embs, axis=0),
        "period_ids": np.array(all_period_ids, dtype=int),
        "mid_years": np.array(all_mid_years, dtype=float),
        "labels": np.array(all_labels),
        "periods": periods,
    }


# ---------------------------------------------------------------------------
# Analysis 1: Cross-sequence co-temporal clustering at L04
# ---------------------------------------------------------------------------

def analysis1_cross_sequence_clustering(all_data: dict):
    print("=" * 70)
    print("ANALYSIS 1: Cross-sequence co-temporal clustering at L04")
    print("=" * 70)

    LAYER = 4
    centroids = []   # list of (seq_key, period_label, mid_year, centroid_vec)

    for seq_key, data in all_data.items():
        embs = data["embeddings"][:, LAYER, :]   # (N, 768)
        period_ids = data["period_ids"]
        mid_years = data["mid_years"]
        periods = data["periods"]

        for p_idx, period in enumerate(periods):
            mask = period_ids == p_idx
            if mask.sum() == 0:
                continue
            centroid = embs[mask].mean(axis=0)
            mid_year = mid_years[mask][0]
            centroids.append((seq_key, period.label, mid_year, centroid))

    print(f"  Total period centroids across all sequences: {len(centroids)}")

    # Stack centroids for PCA
    vecs = np.stack([c[3] for c in centroids], axis=0)
    mid_years_all = np.array([c[2] for c in centroids])
    seq_keys_all = [c[0] for c in centroids]
    labels_all = [c[1] for c in centroids]

    scaler = StandardScaler()
    vecs_scaled = scaler.fit_transform(vecs)

    pca = PCA(n_components=2)
    pca2 = pca.fit_transform(vecs_scaled)

    print(f"  PCA var explained: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: color by mid_year (continuous)
    ax = axes[0]
    norm = plt.Normalize(mid_years_all.min(), mid_years_all.max())
    cmap = cm.plasma
    scatter = ax.scatter(pca2[:, 0], pca2[:, 1],
                         c=mid_years_all, cmap=cmap, norm=norm,
                         s=120, alpha=0.85, edgecolors='k', linewidths=0.4)
    plt.colorbar(scatter, ax=ax, label="Mid-year of period")
    ax.set_title("L04 period centroids — all sequences\n(colored by mid-year)", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Annotate with short labels
    for i, (sk, lbl, yr, _) in enumerate(centroids):
        short = lbl[:10]
        ax.annotate(short, (pca2[i, 0], pca2[i, 1]),
                    fontsize=6, ha='center', va='bottom', alpha=0.7)

    # Right: color by sequence
    ax = axes[1]
    seq_list = list(SEQUENCES.keys())
    seq_colors = cm.tab10(np.linspace(0, 1, len(seq_list)))
    seq_color_map = {sk: seq_colors[i] for i, sk in enumerate(seq_list)}
    for i, (sk, lbl, yr, _) in enumerate(centroids):
        ax.scatter(pca2[i, 0], pca2[i, 1],
                   color=seq_color_map[sk], s=120, alpha=0.85,
                   edgecolors='k', linewidths=0.4)
    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=seq_color_map[sk],
                           markersize=8, label=SEQUENCES[sk]["title"])
               for sk in seq_list]
    ax.legend(handles=handles, fontsize=7, loc='best')
    ax.set_title("L04 period centroids — all sequences\n(colored by sequence)", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    for i, (sk, lbl, yr, _) in enumerate(centroids):
        ax.annotate(lbl[:10], (pca2[i, 0], pca2[i, 1]),
                    fontsize=6, ha='center', va='bottom', alpha=0.7)

    plt.tight_layout()
    out_path = OUT_DIR / "explore_1_cross_sequence_clustering.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")

    # Compute: for pairs from different sequences, correlate (embedding distance, |Δyear|)
    n = len(centroids)
    dist_matrix = squareform(pdist(vecs_scaled, metric='euclidean'))

    cross_seq_dists = []
    cross_seq_dyear = []
    same_seq_dists = []
    same_seq_dyear = []

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            dy = abs(mid_years_all[i] - mid_years_all[j])
            if seq_keys_all[i] != seq_keys_all[j]:
                cross_seq_dists.append(d)
                cross_seq_dyear.append(dy)
            else:
                same_seq_dists.append(d)
                same_seq_dyear.append(dy)

    cross_seq_dists = np.array(cross_seq_dists)
    cross_seq_dyear = np.array(cross_seq_dyear)
    same_seq_dists = np.array(same_seq_dists)
    same_seq_dyear = np.array(same_seq_dyear)

    r_cross, p_cross = stats.pearsonr(cross_seq_dyear, cross_seq_dists)
    r_same, p_same = stats.pearsonr(same_seq_dyear, same_seq_dists)

    print(f"\n  Cross-sequence pairs (embedding dist vs |Δyear|):")
    print(f"    Pearson r = {r_cross:.4f}, p = {p_cross:.4f}")
    print(f"    n_pairs = {len(cross_seq_dists)}")
    print(f"  Same-sequence pairs (embedding dist vs |Δyear|):")
    print(f"    Pearson r = {r_same:.4f}, p = {p_same:.4f}")
    print(f"    n_pairs = {len(same_seq_dists)}")

    if r_cross > 0.3 and p_cross < 0.05:
        print("\n  >> FINDING: Positive correlation across sequences — BERT has a")
        print("     SHARED temporal axis across content domains (century co-clustering).")
    elif r_cross > 0 and p_cross < 0.05:
        print("\n  >> FINDING: Weak but significant positive correlation across sequences —")
        print("     some shared temporal signal but sequence content dominates.")
    else:
        print("\n  >> FINDING: No significant cross-sequence temporal correlation —")
        print("     sequences form SEPARATE CLOUDS in embedding space; content type")
        print("     dominates over shared century.")

    # Check if content groups or time groups cluster better
    # Compute average within-sequence distance vs average cross-sequence distance
    avg_within = np.mean(same_seq_dists)
    avg_cross = np.mean(cross_seq_dists)
    print(f"\n  Average within-sequence centroid distance: {avg_within:.3f}")
    print(f"  Average cross-sequence centroid distance:  {avg_cross:.3f}")
    print(f"  Ratio (cross/within) = {avg_cross/avg_within:.2f}")
    if avg_cross > avg_within * 1.5:
        print("  >> Sequences form tight within-sequence clouds (content type > time).")
    elif avg_cross < avg_within * 1.2:
        print("  >> Cross-sequence distances ~ within-sequence: no strong content-type grouping.")
    else:
        print("  >> Moderate separation: some content-type grouping, some shared temporal axis.")

    print()
    return centroids, vecs_scaled, mid_years_all, seq_keys_all


# ---------------------------------------------------------------------------
# Analysis 2: Distance-time scatter per sequence at L04
# ---------------------------------------------------------------------------

def analysis2_distance_time_scatter(all_data: dict):
    print("=" * 70)
    print("ANALYSIS 2: Distance-time scatter per sequence at L04")
    print("=" * 70)

    LAYER = 4
    seq_list = list(SEQUENCES.keys())
    n_seqs = len(seq_list)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    findings = []

    for ax_idx, seq_key in enumerate(seq_list):
        seq_def = SEQUENCES[seq_key]
        data = all_data[seq_key]
        embs = data["embeddings"][:, LAYER, :]  # (N, 768)
        period_ids = data["period_ids"]
        periods = data["periods"]

        # Compute centroid per period
        n_periods = len(periods)
        centroids = []
        for p_idx in range(n_periods):
            mask = period_ids == p_idx
            centroids.append(embs[mask].mean(axis=0))
        centroids = np.array(centroids)

        # Standardize
        scaler = StandardScaler()
        centroids_s = scaler.fit_transform(centroids)

        # Pairwise distances and |Δmid_year|
        mid_years = np.array([(p.start + p.end) / 2.0 for p in periods])
        pair_dists = []
        pair_dyears = []
        pair_labels = []

        for i in range(n_periods):
            for j in range(i + 1, n_periods):
                d = np.linalg.norm(centroids_s[i] - centroids_s[j])
                dy = abs(mid_years[i] - mid_years[j])
                pair_dists.append(d)
                pair_dyears.append(dy)
                pair_labels.append(f"{periods[i].label[:8]}↔{periods[j].label[:8]}")

        pair_dists = np.array(pair_dists)
        pair_dyears = np.array(pair_dyears)

        ax = axes[ax_idx]

        if len(pair_dists) >= 2:
            ax.scatter(pair_dyears, pair_dists, s=60, alpha=0.8, color='steelblue', edgecolors='k', linewidths=0.3)
            # Linear fit
            m, b, r, p_val, se = stats.linregress(pair_dyears, pair_dists)
            x_line = np.array([pair_dyears.min(), pair_dyears.max()])
            ax.plot(x_line, m * x_line + b, 'r--', linewidth=1.5, label=f'r={r:.2f}, p={p_val:.3f}')
            # Annotate pairs
            for k, lbl in enumerate(pair_labels):
                ax.annotate(lbl, (pair_dyears[k], pair_dists[k]),
                            fontsize=5, ha='center', va='bottom', alpha=0.6)
            ax.legend(fontsize=7)

            # Spearman
            rho, p_rho = stats.spearmanr(pair_dyears, pair_dists)

            btype = seq_def["boundary_type"]
            mono = seq_def.get("monotonic", True)
            title = f"{seq_def['title']}\n({btype}, {'mono' if mono else 'non-mono'})"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("|Δmid-year|", fontsize=8)
            ax.set_ylabel("L2 dist (std)", fontsize=8)

            findings.append({
                "seq": seq_key,
                "title": seq_def["title"],
                "n_pairs": len(pair_dists),
                "pearson_r": r,
                "pearson_p": p_val,
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "boundary_type": btype,
                "monotonic": mono,
            })
        else:
            ax.set_title(seq_def["title"] + "\n(too few periods)", fontsize=9)

    plt.suptitle("L04: Embedding distance vs temporal gap (period centroid pairs)", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "explore_2_distance_time_scatter.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")
    print()

    print(f"  {'Sequence':<25} {'n_pairs':>7} {'Pearson r':>10} {'p':>8} {'Spearman ρ':>12} {'p':>8} {'Type':>12}")
    print(f"  {'-'*90}")
    for f in findings:
        print(f"  {f['title']:<25} {f['n_pairs']:>7} {f['pearson_r']:>10.3f} {f['pearson_p']:>8.3f} "
              f"{f['spearman_rho']:>12.3f} {f['spearman_p']:>8.3f} {f['boundary_type']:>12}")

    print()
    print("  Interpretation:")
    for f in findings:
        if f['pearson_r'] > 0.6 and f['pearson_p'] < 0.05:
            interp = "Strong linear dist-time relationship — BERT encodes temporal gap"
        elif f['pearson_r'] > 0.3 and f['pearson_p'] < 0.05:
            interp = "Moderate linear dist-time relationship"
        elif f['spearman_rho'] > 0.5 and f['spearman_p'] < 0.05:
            interp = "Non-linear but monotone dist-time: rank-order preserved"
        elif f['pearson_r'] < 0 and f['pearson_p'] < 0.05:
            interp = "NEGATIVE correlation — closer in time = farther in embedding!"
        else:
            interp = "No significant dist-time correlation"
        print(f"  >> {f['title']}: {interp}")

    print()
    return findings


# ---------------------------------------------------------------------------
# Analysis 3: Layer trajectory of dynasty centroids
# ---------------------------------------------------------------------------

def analysis3_dynasty_layer_trajectory(all_data: dict):
    print("=" * 70)
    print("ANALYSIS 3: Layer trajectory of dynasty centroids (ruling_dynasty)")
    print("=" * 70)

    data = all_data["ruling_dynasty"]
    embs = data["embeddings"]   # (N, 13, 768)
    period_ids = data["period_ids"]
    periods = data["periods"]
    n_layers = embs.shape[1]
    n_periods = len(periods)

    print(f"  Dynasties: {[p.label for p in periods]}")
    print(f"  Layers 0..{n_layers-1}, computing centroid per dynasty per layer")

    # centroid_traj[p_idx, layer] = centroid vector (768,)
    centroid_traj = np.zeros((n_periods, n_layers, 768))
    for p_idx in range(n_periods):
        mask = period_ids == p_idx
        for layer in range(n_layers):
            centroid_traj[p_idx, layer] = embs[mask, layer, :].mean(axis=0)

    # Flatten to (n_periods * n_layers, 768) for PCA
    flat = centroid_traj.reshape(-1, 768)
    scaler = StandardScaler()
    flat_s = scaler.fit_transform(flat)

    pca = PCA(n_components=2)
    pca2 = pca.fit_transform(flat_s)
    pca2_traj = pca2.reshape(n_periods, n_layers, 2)  # (dynasty, layer, 2)

    print(f"  PCA var explained: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    # Plot trajectory
    dynasty_colors = cm.tab10(np.linspace(0, 1, n_periods))
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    for p_idx, period in enumerate(periods):
        xy = pca2_traj[p_idx]  # (13, 2)
        color = dynasty_colors[p_idx]
        ax.plot(xy[:, 0], xy[:, 1], '-', color=color, linewidth=1.5, alpha=0.7)
        # Mark early layers (L0, L4, L8, L12)
        for layer_mark in [0, 4, 8, 12]:
            if layer_mark < n_layers:
                ax.scatter(xy[layer_mark, 0], xy[layer_mark, 1],
                           color=color, s=60, zorder=5,
                           marker='o' if layer_mark == 0 else ('s' if layer_mark == 4 else
                                                                '^' if layer_mark == 8 else 'D'),
                           alpha=0.9)
        ax.annotate(f"{period.label}\n({period.start}-{period.end})",
                    (xy[12, 0], xy[12, 1]),  # label at final layer
                    fontsize=7, ha='center',
                    color=color, fontweight='bold')

    ax.set_title("Dynasty centroid trajectories L0→L12\n(all dynasties in shared PCA2 space)", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Legend for markers
    from matplotlib.lines import Line2D
    marker_legend = [
        Line2D([0], [0], marker='o', color='gray', markersize=7, label='L0', linestyle='none'),
        Line2D([0], [0], marker='s', color='gray', markersize=7, label='L4', linestyle='none'),
        Line2D([0], [0], marker='^', color='gray', markersize=7, label='L8', linestyle='none'),
        Line2D([0], [0], marker='D', color='gray', markersize=7, label='L12', linestyle='none'),
    ]
    ax.legend(handles=marker_legend, fontsize=8, loc='lower right')

    # Right panel: pairwise centroid distance matrix at L4 vs L12
    ax2 = axes[1]
    for layer_target, col, marker, label_str in [(4, 'steelblue', 'o', 'L04'), (12, 'tomato', 's', 'L12')]:
        centroids_layer = centroid_traj[:, layer_target, :]
        scaler2 = StandardScaler()
        centroids_s = scaler2.fit_transform(centroids_layer)
        dists = [np.linalg.norm(centroids_s[i] - centroids_s[i+1])
                 for i in range(n_periods - 1)]
        pair_names = [f"{periods[i].label[:4]}→{periods[i+1].label[:4]}"
                      for i in range(n_periods - 1)]
        ax2.plot(range(len(dists)), dists, marker=marker, color=col,
                 linewidth=1.5, label=label_str, markersize=8)

    ax2.set_xticks(range(n_periods - 1))
    ax2.set_xticklabels(pair_names, rotation=30, ha='right', fontsize=8)
    ax2.set_title("Adjacent-dynasty centroid L2 distance\nat L04 vs L12", fontsize=11)
    ax2.set_ylabel("L2 distance (standardised)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "explore_3_dynasty_trajectories.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")
    print()

    # Findings: do trajectories converge or diverge?
    # Compute average pairwise distance at L0 vs L4 vs L12
    for lyr in [0, 4, 12]:
        centroids_l = centroid_traj[:, lyr, :]
        scaler3 = StandardScaler()
        cl_s = scaler3.fit_transform(centroids_l)
        dm = squareform(pdist(cl_s, 'euclidean'))
        avg_d = dm[np.triu_indices(n_periods, k=1)].mean()
        print(f"  L{lyr:02d}: avg pairwise dynasty centroid dist = {avg_d:.4f}")

    print()
    # Which dynasties are most confused at L04?
    centroids_l4 = centroid_traj[:, 4, :]
    scaler4 = StandardScaler()
    cl4_s = scaler4.fit_transform(centroids_l4)
    dm4 = squareform(pdist(cl4_s, 'euclidean'))
    print("  L04 pairwise distances (adjacent pairs):")
    for i in range(n_periods - 1):
        print(f"    {periods[i].label:<12} ↔ {periods[i+1].label:<12}: {dm4[i, i+1]:.4f}")

    # Most confused pair
    min_d = np.inf
    min_pair = None
    for i in range(n_periods):
        for j in range(i + 1, n_periods):
            if dm4[i, j] < min_d:
                min_d = dm4[i, j]
                min_pair = (i, j)
    print(f"\n  Most confused pair at L04: {periods[min_pair[0]].label} ↔ "
          f"{periods[min_pair[1]].label} (dist={min_d:.4f})")

    # Check if Tudor passes "through" Lancaster
    # Tudor's L04 position — is it between Plantagenet and Stuart?
    tudor_idx = [p.label for p in periods].index("Tudor")
    lanc_idx = [p.label for p in periods].index("Lancaster")
    york_idx = [p.label for p in periods].index("York")

    for lyr in [0, 4, 8, 12]:
        centroids_l = centroid_traj[:, lyr, :]
        scaler5 = StandardScaler()
        cl_s = scaler5.fit_transform(centroids_l)
        d_tudor_lanc = np.linalg.norm(cl_s[tudor_idx] - cl_s[lanc_idx])
        d_tudor_york = np.linalg.norm(cl_s[tudor_idx] - cl_s[york_idx])
        print(f"  L{lyr:02d}: Tudor↔Lancaster={d_tudor_lanc:.4f}, Tudor↔York={d_tudor_york:.4f}")

    print()


# ---------------------------------------------------------------------------
# Analysis 4: Between-sequence temporal direction alignment
# ---------------------------------------------------------------------------

def analysis4_direction_alignment(all_data: dict):
    print("=" * 70)
    print("ANALYSIS 4: Between-sequence temporal direction alignment at L04")
    print("=" * 70)

    LAYER = 4
    seq_list = list(SEQUENCES.keys())
    directions = {}

    for seq_key in seq_list:
        seq_def = SEQUENCES[seq_key]
        data = all_data[seq_key]
        embs = data["embeddings"][:, LAYER, :]
        period_ids = data["period_ids"]
        periods = data["periods"]

        # Centroid of earliest and latest period
        earliest_mask = period_ids == 0
        latest_mask = period_ids == (len(periods) - 1)
        earliest_centroid = embs[earliest_mask].mean(axis=0)
        latest_centroid = embs[latest_mask].mean(axis=0)

        direction = latest_centroid - earliest_centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        else:
            direction = np.zeros_like(direction)

        directions[seq_key] = direction
        print(f"  {SEQUENCES[seq_key]['title']:<25}: direction norm = {norm:.4f} "
              f"  earliest={periods[0].label}, latest={periods[-1].label}")

    # Pairwise cosine similarity
    print("\n  Pairwise cosine similarity matrix between temporal direction vectors:")
    print(f"  {'':25}", end="")
    short_names = [SEQUENCES[sk]["title"][:12] for sk in seq_list]
    for sn in short_names:
        print(f"  {sn:>13}", end="")
    print()

    cos_matrix = np.zeros((len(seq_list), len(seq_list)))
    for i, sk_i in enumerate(seq_list):
        for j, sk_j in enumerate(seq_list):
            cos_matrix[i, j] = np.dot(directions[sk_i], directions[sk_j])

    for i, sk_i in enumerate(seq_list):
        print(f"  {SEQUENCES[sk_i]['title']:<25}", end="")
        for j in range(len(seq_list)):
            print(f"  {cos_matrix[i, j]:>13.3f}", end="")
        print()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cos_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(len(seq_list)))
    ax.set_yticks(range(len(seq_list)))
    labels = [SEQUENCES[sk]["title"] for sk in seq_list]
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(seq_list)):
        for j in range(len(seq_list)):
            ax.text(j, i, f"{cos_matrix[i, j]:.2f}", ha='center', va='center',
                    fontsize=8, color='black' if abs(cos_matrix[i, j]) < 0.7 else 'white')
    ax.set_title("Cosine similarity: temporal direction vectors\n(earliest→latest centroid at L04)", fontsize=11)
    plt.tight_layout()
    out_path = OUT_DIR / "explore_4_direction_alignment.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Saved: {out_path}")

    # Summary stats
    off_diag = cos_matrix[np.triu_indices(len(seq_list), k=1)]
    print(f"\n  Off-diagonal cosine similarities:")
    print(f"    Mean = {off_diag.mean():.4f}")
    print(f"    Std  = {off_diag.std():.4f}")
    print(f"    Min  = {off_diag.min():.4f}")
    print(f"    Max  = {off_diag.max():.4f}")

    # Find most/least aligned pairs
    max_idx = np.argmax(np.abs(off_diag))
    min_idx = np.argmin(np.abs(off_diag))
    triu_i, triu_j = np.triu_indices(len(seq_list), k=1)

    # Most aligned
    i_max, j_max = triu_i[np.argmax(off_diag)], triu_j[np.argmax(off_diag)]
    i_min, j_min = triu_i[np.argmin(off_diag)], triu_j[np.argmin(off_diag)]
    i_orth, j_orth = triu_i[np.argmin(np.abs(off_diag))], triu_j[np.argmin(np.abs(off_diag))]

    print(f"\n  Most aligned (highest cos): {SEQUENCES[seq_list[i_max]]['title']} ↔ "
          f"{SEQUENCES[seq_list[j_max]]['title']} = {cos_matrix[i_max, j_max]:.3f}")
    print(f"  Most anti-aligned:          {SEQUENCES[seq_list[i_min]]['title']} ↔ "
          f"{SEQUENCES[seq_list[j_min]]['title']} = {cos_matrix[i_min, j_min]:.3f}")
    print(f"  Most orthogonal:            {SEQUENCES[seq_list[i_orth]]['title']} ↔ "
          f"{SEQUENCES[seq_list[j_orth]]['title']} = {cos_matrix[i_orth, j_orth]:.3f}")

    if abs(off_diag.mean()) < 0.1:
        print("\n  >> FINDING: Direction vectors are roughly orthogonal on average —")
        print("     each sequence occupies its OWN subspace in 768-d. There is NO")
        print("     single shared 'time direction' vector.")
    elif off_diag.mean() > 0.3:
        print("\n  >> FINDING: Direction vectors are positively correlated — BERT uses")
        print("     a PARTIALLY SHARED temporal direction across content domains.")
    elif off_diag.mean() < -0.1:
        print("\n  >> FINDING: Direction vectors are on average negatively correlated —")
        print("     sequences point in opposite directions in 768-d (unexpected).")
    else:
        print("\n  >> FINDING: Mixed alignment — some pairs share direction, others orthogonal.")

    print()


# ---------------------------------------------------------------------------
# Analysis 5: Variance decomposition for ruling_dynasty at L04
# ---------------------------------------------------------------------------

def analysis5_variance_decomposition(all_data: dict):
    print("=" * 70)
    print("ANALYSIS 5: Variance decomposition for ruling_dynasty at L04")
    print("=" * 70)

    LAYER = 4
    seq_def = SEQUENCES["ruling_dynasty"]
    data = all_data["ruling_dynasty"]
    embs = data["embeddings"][:, LAYER, :]  # (N, 768)
    period_ids = data["period_ids"]
    periods = seq_def["periods"]
    templates = seq_def["templates"]
    n_periods = len(periods)
    n_templates = len(templates)

    # Standardize
    scaler = StandardScaler()
    embs_s = scaler.fit_transform(embs)

    # --- Between-dynasty variance ---
    centroids = np.array([embs_s[period_ids == p_idx].mean(axis=0)
                          for p_idx in range(n_periods)])
    grand_mean = embs_s.mean(axis=0)
    between_dynasty_var = np.mean([np.linalg.norm(centroids[p] - grand_mean)**2
                                   for p in range(n_periods)])

    # --- Within-dynasty variance ---
    within_vars = []
    for p_idx in range(n_periods):
        mask = period_ids == p_idx
        grp = embs_s[mask]
        centroid = grp.mean(axis=0)
        within_v = np.mean([np.linalg.norm(grp[k] - centroid)**2 for k in range(len(grp))])
        within_vars.append(within_v)
    within_dynasty_var = np.mean(within_vars)
    bw_ratio = between_dynasty_var / within_dynasty_var if within_dynasty_var > 0 else np.inf

    print(f"  Between-dynasty variance (mean ||centroid - grand_mean||^2):  {between_dynasty_var:.4f}")
    print(f"  Within-dynasty variance  (mean per-dynasty avg ||x-centroid||^2): {within_dynasty_var:.4f}")
    print(f"  Between/Within ratio: {bw_ratio:.4f}")
    print()

    # --- Per-template analysis ---
    # For each dynasty, compute variance across the 4 templates
    # Templates are embedded as: for each period, templates cycle through
    # With n_years_per_period samples per template, we need to identify which template each row is.
    # From build_dataset: for each period, we iterate templates only (no year sampling here)
    # So within each period block: rows are template_0, template_1, ..., template_{n_tmpl-1}
    # Check: total N = n_periods * n_templates
    N = embs_s.shape[0]
    expected_N = n_periods * n_templates
    print(f"  Total sentences: {N} (expected: {expected_N})")

    template_ids = np.zeros(N, dtype=int)
    for i in range(N):
        template_ids[i] = i % n_templates  # templates cycle within each period block

    # Verify that period assignment matches
    # period_ids should be 0,0,0,0, 1,1,1,1, ... etc for n_templates per period
    periods_reconstructed = np.repeat(np.arange(n_periods), n_templates)
    if not np.array_equal(period_ids, periods_reconstructed[:len(period_ids)]):
        print("  WARNING: period_ids don't match expected pattern — template assignment may be wrong")
    else:
        print("  Period ID pattern verified.")

    # Per-dynasty: between-template variance
    between_template_vars = []
    for p_idx in range(n_periods):
        mask = period_ids == p_idx
        grp = embs_s[mask]  # shape (n_templates, 768)
        tmpl_ids = template_ids[mask]
        tmpl_centroids = np.array([grp[tmpl_ids == t].mean(axis=0)
                                   for t in range(n_templates) if (tmpl_ids == t).any()])
        dynasty_mean = grp.mean(axis=0)
        bt_var = np.mean([np.linalg.norm(tc - dynasty_mean)**2 for tc in tmpl_centroids])
        between_template_vars.append(bt_var)

    avg_between_template_var = np.mean(between_template_vars)
    template_signal_ratio = avg_between_template_var / between_dynasty_var

    print(f"\n  Per-dynasty between-template variance (avg): {avg_between_template_var:.4f}")
    print(f"  Between-dynasty variance:                    {between_dynasty_var:.4f}")
    print(f"  Template-noise / dynasty-signal ratio:       {template_signal_ratio:.4f}")
    print()

    # Print per-dynasty breakdown
    print(f"  {'Dynasty':<15} {'within_var':>12} {'btw_template_var':>18}")
    print(f"  {'-'*48}")
    for p_idx, period in enumerate(periods):
        print(f"  {period.label:<15} {within_vars[p_idx]:>12.4f} {between_template_vars[p_idx]:>18.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x = np.arange(n_periods)
    width = 0.35
    ax.bar(x - width/2, within_vars, width, label='Within-dynasty var', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, between_template_vars, width, label='Between-template var', color='tomato', alpha=0.8)
    ax.axhline(between_dynasty_var, color='green', linestyle='--', linewidth=1.5,
               label=f'Between-dynasty var ({between_dynasty_var:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels([p.label for p in periods], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Variance (in std-scaled 768-d space)")
    ax.set_title(f"Ruling dynasty L04: variance decomposition\nBW ratio = {bw_ratio:.2f}", fontsize=11)
    ax.legend(fontsize=8)

    # Right: PCA2 of per-template embeddings, colored by dynasty, shaped by template
    ax2 = axes[1]
    pca = PCA(n_components=2)
    pca2 = pca.fit_transform(embs_s)
    dynasty_colors = cm.tab10(np.linspace(0, 1, n_periods))
    markers = ['o', 's', '^', 'D']
    for p_idx in range(n_periods):
        mask = period_ids == p_idx
        grp_pca = pca2[mask]
        tmpl_ids_grp = template_ids[mask]
        for t in range(n_templates):
            t_mask = tmpl_ids_grp == t
            if t_mask.any():
                ax2.scatter(grp_pca[t_mask, 0], grp_pca[t_mask, 1],
                            color=dynasty_colors[p_idx],
                            marker=markers[t], s=80, alpha=0.8,
                            edgecolors='k', linewidths=0.3,
                            label=f"{periods[p_idx].label[:5]}/T{t}" if p_idx == 0 else "")

    # Simplified legend: just dynasties
    dynasty_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=dynasty_colors[p_idx],
                                  markersize=9, label=periods[p_idx].label)
                       for p_idx in range(n_periods)]
    tmpl_handles = [plt.Line2D([0], [0], marker=markers[t], color='gray',
                               markersize=8, label=f"Template {t}", linestyle='none')
                    for t in range(n_templates)]
    ax2.legend(handles=dynasty_handles + tmpl_handles, fontsize=7, loc='best',
               ncol=2, framealpha=0.7)
    ax2.set_title("PCA2: ruling dynasty L04\n(color=dynasty, shape=template)", fontsize=11)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    plt.tight_layout()
    out_path = OUT_DIR / "explore_5_dynasty_variance.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Saved: {out_path}")
    print()

    if template_signal_ratio < 0.3:
        print("  >> FINDING: Between-template variance is << between-dynasty variance.")
        print("     Dynasty name dominates: BERT ignores template framing.")
    elif template_signal_ratio > 1.0:
        print("  >> FINDING: Between-template variance EXCEEDS between-dynasty variance!")
        print("     Template noise is larger than dynasty signal — framing swamps content.")
    else:
        print(f"  >> FINDING: Template noise is {template_signal_ratio:.0%} of dynasty signal.")
        print("     Non-trivial template effect, but dynasty name still dominates.")

    if bw_ratio > 5:
        print(f"  >> High BW ratio ({bw_ratio:.1f}): dynasties are very well separated at L04.")
    elif bw_ratio > 2:
        print(f"  >> Moderate BW ratio ({bw_ratio:.1f}): dynasties are reasonably separated.")
    else:
        print(f"  >> Low BW ratio ({bw_ratio:.1f}): dynasties are poorly separated.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading BERT model...")
    model, tokenizer, device = load_model()

    print("Embedding all sequences (this will take a few minutes)...")
    all_data = {}
    for seq_key, seq_def in SEQUENCES.items():
        print(f"  Embedding: {seq_def['title']} ({len(seq_def['periods'])} periods, "
              f"{len(seq_def['templates'])} templates each)...", flush=True)
        all_data[seq_key] = build_dataset(seq_key, seq_def, model, tokenizer, device)
        N = all_data[seq_key]["embeddings"].shape[0]
        print(f"    -> {N} sentences embedded", flush=True)

    print("\nAll sequences embedded.\n")

    # Run analyses
    analysis1_cross_sequence_clustering(all_data)
    analysis2_distance_time_scatter(all_data)
    analysis3_dynasty_layer_trajectory(all_data)
    analysis4_direction_alignment(all_data)
    analysis5_variance_decomposition(all_data)

    print("=" * 70)
    print("All analyses complete. Plots saved to:")
    for name in [
        "explore_1_cross_sequence_clustering.png",
        "explore_2_distance_time_scatter.png",
        "explore_3_dynasty_trajectories.png",
        "explore_4_direction_alignment.png",
        "explore_5_dynasty_variance.png",
    ]:
        print(f"  {OUT_DIR / name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
