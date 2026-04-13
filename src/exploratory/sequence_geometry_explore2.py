"""Exploratory geometric analysis of BERT embeddings — Part 2.

Four analyses:
  A. Template residualisation for ruling dynasty
  B. Full pairwise distance matrix for ruling dynasty
  C. Intrinsic dimensionality per sequence (participation ratio)
  D. Shared temporal direction across sequences + word-projection test

Saves plots to src/appendix/explore2_A.png, explore2_B.png, explore2_C.png, explore2_D.png
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

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sequence definitions
# ---------------------------------------------------------------------------

class Period(NamedTuple):
    label: str
    start: int
    end:   int


SEQUENCES: dict[str, dict] = {
    "country_name": {
        "title": "Country name",
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
        "monotonic": False,
        "repeated_labels": True,
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
        "monotonic": False,
        "repeated_labels": True,
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
        "monotonic": True,
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
        "monotonic": True,
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
        "monotonic": True,
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
        "monotonic": True,
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
        "monotonic": True,
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

DYNASTY_MID_YEARS = {
    "Plantagenet": 1299,
    "Lancaster":   1430,
    "York":        1473,
    "Tudor":       1544,
    "Stuart":      1658,
    "Hanover":     1775,
    "Windsor":     1869,
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


def build_dataset(seq_key: str, seq_def: dict, model, tokenizer, device):
    """
    Returns dict with:
      embeddings: (N, 13, 768)
      period_ids: (N,) int
      mid_years:  (N,) float
      labels:     (N,) str
      template_ids: (N,) int
      sentences:  (N,) str
      periods:    list of Period
    """
    periods = seq_def["periods"]
    templates = seq_def["templates"]

    all_embs, all_period_ids, all_mid_years, all_labels = [], [], [], []
    all_template_ids, all_sentences = [], []

    for p_idx, period in enumerate(periods):
        mid_year = (period.start + period.end) / 2.0
        for t_idx, tmpl in enumerate(templates):
            sentence = tmpl.format(label=period.label)
            emb = embed_sentence(model, tokenizer, device, sentence)
            all_embs.append(emb)
            all_period_ids.append(p_idx)
            all_mid_years.append(mid_year)
            all_labels.append(period.label)
            all_template_ids.append(t_idx)
            all_sentences.append(sentence)

    return {
        "embeddings": np.stack(all_embs, axis=0),
        "period_ids": np.array(all_period_ids, dtype=int),
        "mid_years":  np.array(all_mid_years, dtype=float),
        "labels":     np.array(all_labels),
        "template_ids": np.array(all_template_ids, dtype=int),
        "sentences":  all_sentences,
        "periods":    periods,
    }


# ---------------------------------------------------------------------------
# Utility: BW ratio
# ---------------------------------------------------------------------------

def bw_ratio(X: np.ndarray, group_ids: np.ndarray) -> float:
    """Between/Within variance ratio. X: (N, D), group_ids: (N,) int."""
    groups = np.unique(group_ids)
    grand_mean = X.mean(axis=0)

    # Between-cluster variance (weighted by cluster size)
    between = 0.0
    for g in groups:
        mask = group_ids == g
        c = X[mask].mean(axis=0)
        between += mask.sum() * np.sum((c - grand_mean) ** 2)
    between /= X.shape[0]

    # Within-cluster variance
    within = 0.0
    for g in groups:
        mask = group_ids == g
        c = X[mask].mean(axis=0)
        within += np.sum((X[mask] - c) ** 2)
    within /= X.shape[0]

    return between / (within + 1e-12)


# ---------------------------------------------------------------------------
# Analysis A: Template residualisation for ruling dynasty
# ---------------------------------------------------------------------------

def analysis_A(model, tokenizer, device):
    print("=" * 70)
    print("ANALYSIS A: Template residualisation for ruling dynasty")
    print("=" * 70)

    LAYER = 4
    seq_def = SEQUENCES["ruling_dynasty"]
    data = build_dataset("ruling_dynasty", seq_def, model, tokenizer, device)

    embs       = data["embeddings"][:, LAYER, :]   # (28, 768)
    period_ids = data["period_ids"]                # (28,) dynasty index
    template_ids = data["template_ids"]            # (28,) template index
    labels     = data["labels"]                    # (28,) dynasty name
    periods    = data["periods"]                   # 7 Period objects

    # StandardScaler before BW ratio
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    # Dynasty mid-years array in order of period index
    dynasty_names = [p.label for p in periods]
    mid_years_ordered = np.array([DYNASTY_MID_YEARS[n] for n in dynasty_names])
    mid_years = np.array([mid_years_ordered[pid] for pid in period_ids])

    # ---- Before residualisation ----
    bw_before = bw_ratio(embs_scaled, period_ids)
    pca_before = PCA(n_components=2)
    pc_before = pca_before.fit_transform(embs_scaled)
    rho_before, p_rho_before = stats.spearmanr(pc_before[:, 0], mid_years)

    print(f"\n  Before residualisation:")
    print(f"    BW ratio              = {bw_before:.4f}")
    print(f"    Spearman ρ (PC1 vs mid-year) = {rho_before:.4f}  (p={p_rho_before:.4f})")

    # ---- Template means ----
    n_templates = len(seq_def["templates"])
    template_means = []
    for t_idx in range(n_templates):
        mask = template_ids == t_idx
        tmean = embs_scaled[mask].mean(axis=0)
        template_means.append(tmean)
    template_means = np.stack(template_means, axis=0)  # (4, 768)

    # Subtract template mean from each sentence
    embs_resid = embs_scaled.copy()
    for i in range(len(embs_resid)):
        embs_resid[i] -= template_means[template_ids[i]]

    # ---- After residualisation ----
    bw_after = bw_ratio(embs_resid, period_ids)
    pca_after = PCA(n_components=2)
    pc_after = pca_after.fit_transform(embs_resid)
    rho_after, p_rho_after = stats.spearmanr(pc_after[:, 0], mid_years)

    print(f"\n  After residualisation:")
    print(f"    BW ratio              = {bw_after:.4f}")
    print(f"    Spearman ρ (PC1 vs mid-year) = {rho_after:.4f}  (p={p_rho_after:.4f})")

    print(f"\n  BW ratio change: {bw_after - bw_before:+.4f}  "
          f"({'INCREASED' if bw_after > bw_before else 'DECREASED'})")

    # ---- Plot ----
    dynasty_colors = cm.tab10(np.linspace(0, 1, len(periods)))
    dynasty_color_map = {p.label: dynasty_colors[i] for i, p in enumerate(periods)}
    template_markers = ['o', 's', '^', 'D']
    template_labels  = [f"T{i+1}" for i in range(n_templates)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, pc, title in [
        (axes[0], pc_before, f"BEFORE residualisation\nBW={bw_before:.3f}, ρ={rho_before:.3f}"),
        (axes[1], pc_after,  f"AFTER residualisation\nBW={bw_after:.3f}, ρ={rho_after:.3f}"),
    ]:
        for i in range(len(embs)):
            dyn = labels[i]
            t = template_ids[i]
            ax.scatter(pc[i, 0], pc[i, 1],
                       color=dynasty_color_map[dyn],
                       marker=template_markers[t],
                       s=120, alpha=0.85, edgecolors='k', linewidths=0.5)
        # Legend: dynasties (color)
        dyn_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=dynasty_color_map[p.label],
                       markersize=9, label=p.label)
            for p in periods
        ]
        # Legend: templates (shape) — using a neutral dark color
        tmpl_handles = [
            plt.Line2D([0], [0], marker=template_markers[t], color='w',
                       markerfacecolor='dimgray', markersize=9,
                       label=f"T{t+1}")
            for t in range(n_templates)
        ]
        ax.legend(handles=dyn_handles + tmpl_handles, fontsize=7,
                  loc='best', ncol=2, title="Dynasty / Template")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle("Analysis A: Template residualisation — Ruling dynasty at L04",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = OUT_DIR / "explore2_A.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Saved: {out_path}")

    if bw_after > bw_before:
        interp = (
            "BW ratio INCREASED after removing template means: the templates were "
            "adding within-cluster noise that blurred dynasty boundaries. "
            "The residualised embeddings show cleaner dynasty separation, "
            "confirming that dynasty identity is genuinely encoded at L04."
        )
    else:
        interp = (
            "BW ratio DECREASED after removing template means: the template "
            "structure was actually helping to separate dynasties (perhaps "
            "through syntactic variation correlated with dynasty). "
            "Pure dynasty identity survives residualisation but is weaker, "
            "suggesting BERT conflates template style with dynasty signal."
        )
    print(f"\n  INTERPRETATION: {interp}")


# ---------------------------------------------------------------------------
# Analysis B: Pairwise distance matrix + threshold classification
# ---------------------------------------------------------------------------

def analysis_B(model, tokenizer, device):
    print("\n" + "=" * 70)
    print("ANALYSIS B: Full pairwise distance matrix for ruling dynasty")
    print("=" * 70)

    LAYER = 4
    seq_def = SEQUENCES["ruling_dynasty"]
    data = build_dataset("ruling_dynasty", seq_def, model, tokenizer, device)

    embs       = data["embeddings"][:, LAYER, :]
    period_ids = data["period_ids"]
    periods    = data["periods"]

    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    # Compute centroids
    dynasty_names = [p.label for p in periods]
    n_dyn = len(dynasty_names)
    centroids = np.zeros((n_dyn, embs_scaled.shape[1]))
    for i in range(n_dyn):
        mask = period_ids == i
        centroids[i] = embs_scaled[mask].mean(axis=0)

    # 7×7 embedding distance matrix
    emb_dist = squareform(pdist(centroids, metric='euclidean'))

    # 7×7 temporal distance matrix
    mid_years_arr = np.array([DYNASTY_MID_YEARS[n] for n in dynasty_names])
    temp_dist = np.abs(mid_years_arr[:, None] - mid_years_arr[None, :])

    # Upper triangle (excluding diagonal)
    upper_idx = np.triu_indices(n_dyn, k=1)
    emb_upper  = emb_dist[upper_idx]
    temp_upper = temp_dist[upper_idx]

    r_pearson,  p_pearson  = stats.pearsonr(temp_upper, emb_upper)
    r_spearman, p_spearman = stats.spearmanr(temp_upper, emb_upper)

    print(f"\n  Pairwise centroid distances (upper triangle, n={len(emb_upper)} pairs):")
    print(f"    Pearson  r = {r_pearson:.4f}  (p={p_pearson:.4f})")
    print(f"    Spearman ρ = {r_spearman:.4f}  (p={p_spearman:.4f})")
    print(f"\n  Embedding distance matrix (standardised embeddings, euclidean):")
    header = "          " + "".join(f"{n[:6]:>9}" for n in dynasty_names)
    print(header)
    for i, row_name in enumerate(dynasty_names):
        row_str = f"  {row_name[:8]:<9}" + "".join(f"{emb_dist[i,j]:9.3f}" for j in range(n_dyn))
        print(row_str)

    print(f"\n  Temporal distance matrix (|mid_year_i - mid_year_j|):")
    print(header)
    for i, row_name in enumerate(dynasty_names):
        row_str = f"  {row_name[:8]:<9}" + "".join(f"{temp_dist[i,j]:9.0f}" for j in range(n_dyn))
        print(row_str)

    # ---- Threshold classification ----
    era_threshold = 100  # "same era" if within 100 years
    true_same_era = (temp_upper <= era_threshold)
    print(f"\n  Threshold model: classify pairs within {era_threshold} years as 'same era'")
    print(f"  Actual same-era pairs: {true_same_era.sum()} / {len(true_same_era)}")

    best_T = None
    best_acc = -1.0
    best_sens = None
    best_spec = None

    T_values = np.linspace(emb_upper.min(), emb_upper.max(), 200)
    for T in T_values:
        pred_same = (emb_upper < T)
        tp = (pred_same & true_same_era).sum()
        tn = (~pred_same & ~true_same_era).sum()
        fp = (pred_same & ~true_same_era).sum()
        fn = (~pred_same & true_same_era).sum()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        acc  = (tp + tn) / (tp + tn + fp + fn)
        if acc > best_acc:
            best_acc  = acc
            best_T    = T
            best_sens = sens
            best_spec = spec

    print(f"  Best threshold T = {best_T:.4f}")
    print(f"    Accuracy    = {best_acc:.4f}")
    print(f"    Sensitivity = {best_sens:.4f}  (true positive rate for 'same era')")
    print(f"    Specificity = {best_spec:.4f}  (true negative rate for 'different era')")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Normalise for coloring
    vmax_emb = emb_dist.max()
    vmax_tmp = temp_dist.max()

    im0 = axes[0].imshow(emb_dist, cmap='viridis', aspect='auto',
                          vmin=0, vmax=vmax_emb)
    axes[0].set_xticks(range(n_dyn))
    axes[0].set_xticklabels(dynasty_names, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticks(range(n_dyn))
    axes[0].set_yticklabels(dynasty_names, fontsize=9)
    axes[0].set_title(f"Embedding distance matrix (L04)\nPearson r={r_pearson:.3f}, "
                      f"Spearman ρ={r_spearman:.3f}", fontsize=11)
    plt.colorbar(im0, ax=axes[0], label="Euclidean distance")
    for i in range(n_dyn):
        for j in range(n_dyn):
            axes[0].text(j, i, f"{emb_dist[i,j]:.2f}", ha='center', va='center',
                         fontsize=7, color='white' if emb_dist[i,j] > vmax_emb*0.5 else 'black')

    im1 = axes[1].imshow(temp_dist, cmap='plasma', aspect='auto',
                          vmin=0, vmax=vmax_tmp)
    axes[1].set_xticks(range(n_dyn))
    axes[1].set_xticklabels(dynasty_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticks(range(n_dyn))
    axes[1].set_yticklabels(dynasty_names, fontsize=9)
    axes[1].set_title(f"Temporal distance matrix (|Δmid-year|)\n"
                      f"Best threshold T={best_T:.2f}: acc={best_acc:.2f}, "
                      f"sens={best_sens:.2f}, spec={best_spec:.2f}", fontsize=11)
    plt.colorbar(im1, ax=axes[1], label="Years apart")
    for i in range(n_dyn):
        for j in range(n_dyn):
            axes[1].text(j, i, f"{temp_dist[i,j]:.0f}", ha='center', va='center',
                         fontsize=7, color='white' if temp_dist[i,j] > vmax_tmp*0.5 else 'black')

    plt.suptitle("Analysis B: Pairwise distance matrices — Ruling dynasty at L04",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = OUT_DIR / "explore2_B.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Saved: {out_path}")

    if abs(r_pearson) > 0.5:
        strength = "moderate-to-strong"
    elif abs(r_pearson) > 0.3:
        strength = "weak-to-moderate"
    else:
        strength = "very weak"

    interp = (
        f"The {strength} correlation (Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}) "
        f"between embedding distances and temporal distances suggests that BERT's L04 "
        f"representations {'partially reflect' if abs(r_pearson) > 0.3 else 'do not clearly encode'} "
        f"the chronological ordering of English dynasties. "
        f"The threshold classifier (best T={best_T:.2f}) achieves {best_acc:.0%} accuracy at "
        f"distinguishing same-era (within 100 yr) from different-era dynasty pairs."
    )
    print(f"\n  INTERPRETATION: {interp}")


# ---------------------------------------------------------------------------
# Analysis C: Intrinsic dimensionality per sequence (participation ratio)
# ---------------------------------------------------------------------------

def analysis_C(model, tokenizer, device):
    print("\n" + "=" * 70)
    print("ANALYSIS C: Intrinsic dimensionality per sequence at L04")
    print("=" * 70)

    LAYER = 4
    N_PCS = 10

    results = []

    for seq_key, seq_def in SEQUENCES.items():
        data = build_dataset(seq_key, seq_def, model, tokenizer, device)
        embs = data["embeddings"][:, LAYER, :]
        period_ids = data["period_ids"]
        N = embs.shape[0]

        scaler = StandardScaler()
        embs_scaled = scaler.fit_transform(embs)

        pca = PCA(n_components=min(N_PCS, N))
        pca.fit(embs_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # Participation ratio using all non-zero eigenvalues
        # Use full PCA (up to rank of data)
        pca_full = PCA(n_components=min(N - 1, embs_scaled.shape[1]))
        pca_full.fit(embs_scaled)
        eigs = pca_full.explained_variance_  # proportional to eigenvalues
        PR_real = (eigs.sum() ** 2) / (eigs ** 2).sum()

        # Shuffled: randomly reassign sentences to periods
        rng = np.random.default_rng(42)
        shuffled_ids = period_ids.copy()
        rng.shuffle(shuffled_ids)
        # PR is purely about the embedding variance structure, shuffle is for BW context
        # But PR itself doesn't depend on labels — compute it on a shuffled version of embs
        idx_shuf = rng.permutation(N)
        embs_shuf = embs_scaled[idx_shuf]
        pca_shuf = PCA(n_components=min(N - 1, embs_scaled.shape[1]))
        pca_shuf.fit(embs_shuf)
        eigs_shuf = pca_shuf.explained_variance_
        PR_shuf = (eigs_shuf.sum() ** 2) / (eigs_shuf ** 2).sum()

        results.append({
            "seq_key": seq_key,
            "title": seq_def["title"],
            "N": N,
            "PR_real": PR_real,
            "PR_shuf": PR_shuf,
            "cumvar": cumvar,
            "var_ratios": pca.explained_variance_ratio_,
        })

        print(f"\n  {seq_def['title']} (N={N} sentences):")
        print(f"    Participation ratio (real)     = {PR_real:.2f}  effective dims")
        print(f"    Participation ratio (shuffled) = {PR_shuf:.2f}  effective dims")
        print(f"    PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}%  "
              f"| PC2: {pca.explained_variance_ratio_[1]*100:.1f}%  "
              f"| cumulative @ PC{N_PCS}: {cumvar[-1]*100:.1f}%")

    # ---- Plot: cumulative variance curves ----
    colors = cm.tab10(np.linspace(0, 1, len(results)))
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, res in enumerate(results):
        xpts = np.arange(1, len(res["cumvar"]) + 1)
        ax.plot(xpts, res["cumvar"] * 100,
                color=colors[i], marker='o', ms=5,
                label=f"{res['title']} (PR={res['PR_real']:.1f})")
    ax.axhline(90, color='gray', linestyle='--', linewidth=0.8, label='90% threshold')
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Analysis C: Cumulative variance curves per sequence at L04\n"
                 "(PR = participation ratio = effective dimensionality)", fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xticks(range(1, N_PCS + 1))
    ax.set_ylim(0, 105)
    plt.tight_layout()
    out_path = OUT_DIR / "explore2_C.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Saved: {out_path}")

    # Summary table
    print("\n  Summary (sorted by PR_real):")
    print(f"  {'Sequence':<25} {'N':>4} {'PR_real':>8} {'PR_shuf':>8} {'PC1%':>6}")
    for res in sorted(results, key=lambda r: r["PR_real"]):
        print(f"  {res['title']:<25} {res['N']:>4} {res['PR_real']:>8.2f} "
              f"{res['PR_shuf']:>8.2f} {res['var_ratios'][0]*100:>5.1f}%")

    best = max(results, key=lambda r: r["PR_real"])
    worst = min(results, key=lambda r: r["PR_real"])
    interp = (
        f"Sequences vary considerably in intrinsic dimensionality: "
        f"'{best['title']}' is most spread out ({best['PR_real']:.1f} effective dims), "
        f"while '{worst['title']}' is most concentrated ({worst['PR_real']:.1f} effective dims). "
        f"High PR relative to shuffled PR indicates the embedding cloud is not dominated by "
        f"one or two directions; low PR suggests most variance is captured by a single axis."
    )
    print(f"\n  INTERPRETATION: {interp}")

    return results


# ---------------------------------------------------------------------------
# Analysis D: Shared temporal direction across sequences
# ---------------------------------------------------------------------------

def analysis_D(model, tokenizer, device):
    print("\n" + "=" * 70)
    print("ANALYSIS D: Shared temporal direction across sequences")
    print("=" * 70)

    LAYER = 4

    # Sequences without repeated labels
    TARGET_SEQS = [
        "country_name", "ruling_dynasty", "calendar",
        "primary_weapon", "ship_construction", "primary_fuel",
    ]

    direction_vecs = []

    for seq_key in TARGET_SEQS:
        seq_def = SEQUENCES[seq_key]
        data = build_dataset(seq_key, seq_def, model, tokenizer, device)
        embs = data["embeddings"][:, LAYER, :]
        period_ids = data["period_ids"]
        periods = data["periods"]

        scaler = StandardScaler()
        embs_scaled = scaler.fit_transform(embs)

        # Centroid of each period
        unique_pids = sorted(np.unique(period_ids))
        centroids_list = []
        for pid in unique_pids:
            mask = period_ids == pid
            centroids_list.append(embs_scaled[mask].mean(axis=0))

        centroids_arr = np.stack(centroids_list, axis=0)  # (n_periods, 768)

        # Mid-years (ordered by period index)
        mid_years_ordered = np.array(
            [(periods[pid].start + periods[pid].end) / 2.0 for pid in unique_pids]
        )

        # Temporal direction: from earliest centroid to latest
        earliest_idx = np.argmin(mid_years_ordered)
        latest_idx   = np.argmax(mid_years_ordered)
        vec = centroids_arr[latest_idx] - centroids_arr[earliest_idx]
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        direction_vecs.append(vec)
        print(f"  {seq_def['title']}: direction vector norm before normalisation = {norm:.4f}")

    dir_matrix = np.stack(direction_vecs, axis=0)  # (6, 768)

    # PCA on direction vectors
    pca_dir = PCA(n_components=min(6, dir_matrix.shape[0]))
    pca_dir.fit(dir_matrix)

    print(f"\n  PCA of 6 temporal direction vectors:")
    cumvar = 0.0
    for i, var in enumerate(pca_dir.explained_variance_ratio_):
        cumvar += var
        print(f"    PC{i+1}: {var*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")

    pc1_var = pca_dir.explained_variance_ratio_[0]
    shared_pc1 = pca_dir.components_[0]  # (768,) the most shared temporal direction

    print(f"\n  PC1 explains {pc1_var*100:.1f}% of variance among temporal directions.")
    if pc1_var > 0.5:
        print("  >> HIGH: There IS a shared temporal axis across content domains (>50%).")
    elif pc1_var > 0.2:
        print("  >> MODERATE: Weak shared temporal axis (20-50%).")
    else:
        print("  >> LOW: Temporal directions are largely independent (<20%).")

    # ---- Pairwise cosine similarities between direction vectors ----
    print(f"\n  Pairwise cosine similarities between temporal direction vectors:")
    header = "              " + "".join(f"{k[:7]:>9}" for k in TARGET_SEQS)
    print(header)
    cos_mat = dir_matrix @ dir_matrix.T
    for i, ki in enumerate(TARGET_SEQS):
        row = f"  {ki[:12]:<14}" + "".join(f"{cos_mat[i,j]:9.4f}" for j in range(len(TARGET_SEQS)))
        print(row)

    mean_offdiag = (cos_mat.sum() - np.trace(cos_mat)) / (len(TARGET_SEQS) * (len(TARGET_SEQS) - 1))
    print(f"\n  Mean off-diagonal cosine similarity: {mean_offdiag:.4f}")

    # ---- Project 20 random words onto shared PC1 direction ----
    test_words = [
        "sword", "steam", "cannon", "parliament", "knight",
        "telegraph", "feudal", "empire", "rifle", "coal",
        "king", "republic", "ship", "gunpowder", "iron",
        "democracy", "cathedral", "factory", "crown", "railroad",
    ]

    print(f"\n  Projecting {len(test_words)} words onto shared temporal direction (PC1):")
    word_projections = []
    for word in test_words:
        sentence = f"the {word} ."
        emb_layers = embed_sentence(model, tokenizer, device, sentence)
        emb = emb_layers[LAYER]  # (768,)
        # Use a simple per-dimension standardisation based on the direction vector
        proj = float(emb @ shared_pc1)
        word_projections.append((word, proj))

    word_projections.sort(key=lambda x: x[1])
    print(f"\n  Words sorted from LOW to HIGH projection onto shared temporal direction:")
    for word, proj in word_projections:
        bar = "#" * int(abs(proj) * 3 + 1)
        print(f"    {word:<15} {proj:+.4f}  {bar}")

    low_words  = [w for w, _ in word_projections[:5]]
    high_words = [w for w, _ in word_projections[-5:]]
    print(f"\n  LOW end  (most 'early'): {low_words}")
    print(f"  HIGH end (most 'late'):  {high_words}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: explained variance of temporal direction PCA
    ax = axes[0]
    var_ratios = pca_dir.explained_variance_ratio_
    x = np.arange(1, len(var_ratios) + 1)
    ax.bar(x, var_ratios * 100, color='steelblue', edgecolor='k', linewidth=0.7)
    ax.axhline(50, color='red', linestyle='--', linewidth=1.0, label='50% threshold')
    ax.axhline(20, color='orange', linestyle='--', linewidth=1.0, label='20% threshold')
    ax.set_xlabel("PC of temporal direction matrix")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"PCA of 6 temporal direction vectors\nPC1 explains {pc1_var*100:.1f}% of variance",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])

    # Right: word projections onto shared temporal direction
    ax = axes[1]
    words_sorted = [w for w, _ in word_projections]
    projs_sorted = [p for _, p in word_projections]
    colors_bar = ['steelblue' if p >= 0 else 'salmon' for p in projs_sorted]
    y_pos = np.arange(len(words_sorted))
    ax.barh(y_pos, projs_sorted, color=colors_bar, edgecolor='k', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words_sorted, fontsize=9)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel("Projection onto shared temporal direction (PC1)")
    ax.set_title("Word projections onto shared temporal axis\n"
                 "(negative = 'early', positive = 'late')", fontsize=11)

    plt.suptitle("Analysis D: Shared temporal direction across content domains",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = OUT_DIR / "explore2_D.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Saved: {out_path}")

    if pc1_var > 0.5:
        shared_str = "a strong shared temporal axis exists"
        axis_interp = "This axis likely encodes general modernity/technological progress."
    elif pc1_var > 0.2:
        shared_str = "a weak shared temporal axis exists"
        axis_interp = "The shared component reflects partial overlap in how BERT encodes temporal context."
    else:
        shared_str = "no meaningful shared temporal axis exists"
        axis_interp = "Each domain has an independent temporal direction; content dominates entirely."

    interp = (
        f"PC1 of the 6 temporal direction vectors explains {pc1_var*100:.1f}% of variance, "
        f"meaning {shared_str} across content domains. "
        f"Words projecting high ({high_words}) vs low ({low_words}) reveal what this axis encodes. "
        f"{axis_interp}"
    )
    print(f"\n  INTERPRETATION: {interp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SEQUENCE GEOMETRY EXPLORATION 2 — BERT L04 analyses")
    print("=" * 70 + "\n")

    model, tokenizer, device = load_model()

    analysis_A(model, tokenizer, device)
    analysis_B(model, tokenizer, device)
    analysis_C(model, tokenizer, device)
    analysis_D(model, tokenizer, device)

    print("\n" + "=" * 70)
    print("All analyses complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
