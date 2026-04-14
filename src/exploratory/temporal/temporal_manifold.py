"""temporal_manifold.py — Characterise the geometric structure of time
in BERT/MacBERTh embeddings using all categorical time series as probes.

Each series contributes period centroids at known temporal positions.
The joint embedding reveals whether BERT has a single universal temporal
manifold shared across cultural domains, or separate series-specific ones.

Analyses
--------
1. Year manifold shape       PCA 2D+3D of year-carrier embeddings
2. Temporal velocity         ‖Δemb‖/Δyear along the trajectory
3. Curvature                 direction-change angle between consecutive steps
4. Period centroid overlay   where each series' periods land on the year manifold
5. Cross-cultural clustering contemporaneous periods from different series:
                             do they cluster more tightly than non-contemporaneous?
6. Joint MDS                 all period centroids from all series in one layout
7. Epoch detection           velocity+curvature changepoints → natural periods
8. Intrinsic dimensionality  local PCA dimensionality at each point on trajectory

Usage
-----
    python3 src/exploratory/temporal/temporal_manifold.py --model bert
    python3 src/exploratory/temporal/temporal_manifold.py --model macberth --layer 4
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_utils import (  # noqa: E402
    MODEL_NAMES, embed_all_layers, load_embedder,
    velocity, curvature, local_dim,
    centroid_tau, knn_purity,
)

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
TIME_SERIES_DIR = PROJECT_ROOT.parent / "time_series"
OUT_DIR         = PROJECT_ROOT / "data" / "outputs" / "temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Year grid for trajectory analysis
YEAR_STEP  = 25
YEAR_MIN   = -600   # covers roman consuls
YEAR_MAX   = 1925

# Templates for embedding period/entity names
# Both series use the same template frame — only the slot value differs.
# Year slot: "1066", "500", "-200", etc.
# Period slot: "William I", "Tang", "Leo I", etc.
SHARED_TEMPLATES = [
    "it is {x} .",
    "the time is {x} .",
    "this is the time of {x} .",
    "the current period is {x} .",
]

# Classical Chinese templates for models trained on classical Chinese corpora
# (e.g. SikuBERT). Same {x} slot; context is in literary Chinese.
SHARED_TEMPLATES_ZH = [
    "时在{x}。",         # Time is in {x}.
    "当{x}之时。",       # At the time of {x}.
    "此乃{x}也。",       # This is {x}.
    "今{x}是也。",       # Now it is {x}.
]

# Defaults (overridden per model_key in main())
PERIOD_TEMPLATES = SHARED_TEMPLATES
YEAR_TEMPLATES   = SHARED_TEMPLATES

# Colour palette for series
SERIES_COLORS = {
    "roman_consuls":    "#e41a1c",
    "popes":            "#377eb8",
    "chinese_emperors": "#ff7f00",
    "chinese_dynasties":"#984ea3",
    "english_monarchs": "#4daf4a",
    "japanese_nengo":   "#a65628",
    "year_carrier":     "#000000",   # black — the baseline year-is-XXXX series
}

# Max periods to embed per series (subsample dense ones)
MAX_PERIODS = {
    "roman_consuls":   60,   # 345 total → every ~6th
    "japanese_nengo":  60,   # 247 total → every ~4th
    "popes":           80,   # 265 total → every ~3rd
    "chinese_emperors": 97,  # all
    "english_monarchs": 63,  # all
    "chinese_dynasties": 41, # all
    "literary_periods":  41, # all
    "music_periods":     27, # all
    "art_periods":       55, # all
    "archaeological_cultures": 83, # all
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_series() -> dict[str, pd.DataFrame]:
    """Load all categorical CSVs. Returns {stem: df}."""
    # Exclude subjective/non-definitive periodisations and non-historical
    skip = {
        "time_entities.csv", "time_entities.json",
        "literary_periods.csv", "music_periods.csv",
        "art_periods.csv", "archaeological_cultures.csv",
    }
    series = {}
    for path in sorted(TIME_SERIES_DIR.glob("*.csv")):
        if path.name in skip:
            continue
        df = pd.read_csv(path)
        df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
        df["end_year"]   = pd.to_numeric(df["end_year"],   errors="coerce")
        df = df.dropna(subset=["start_year"]).sort_values("start_year").reset_index(drop=True)
        # Restrict to reasonable range
        df = df[df["start_year"] >= YEAR_MIN].copy()
        if len(df) == 0:
            continue
        # Subsample if too dense
        max_p = MAX_PERIODS.get(path.stem, 100)
        if len(df) > max_p:
            idx = np.round(np.linspace(0, len(df)-1, max_p)).astype(int)
            df  = df.iloc[idx].reset_index(drop=True)
        series[path.stem] = df
        mid = (df["start_year"].min() + df["end_year"].fillna(df["start_year"]).max()) / 2
        print(f"  {path.stem:<30} {len(df):>4} periods  "
              f"[{int(df['start_year'].min())}–{int(df['end_year'].fillna(df['start_year']).max())}]")
    return series


# ---------------------------------------------------------------------------
# Embedding (all layers in a single forward pass)
# ---------------------------------------------------------------------------

def compute_year_trajectory_all_layers(embed_fn,
                                       year_templates: list[str],
                                       ) -> tuple[np.ndarray, np.ndarray]:
    """Embed every year on the grid at ALL layers simultaneously.

    Returns:
        years          — (n_years,)
        year_embs_full — (n_years, n_layers, H)  mean over year_templates
    """
    years = np.arange(YEAR_MIN, YEAR_MAX + YEAR_STEP, YEAR_STEP, dtype=float)
    print(f"  Year trajectory (all layers): {len(years)} points …", flush=True)
    rows = []
    for y in years:
        tmpl_embs = np.stack([
            embed_fn(t.format(x=int(y)))
            for t in year_templates
        ])                      # (n_templates, n_layers, H)
        rows.append(tmpl_embs.mean(axis=0))   # (n_layers, H)
    return years, np.stack(rows)              # (n_years, n_layers, H)


def compute_period_centroids_all_layers(series: dict[str, pd.DataFrame],
                                        embed_fn,
                                        period_templates: list[str]) -> dict[str, dict]:
    """Embed each period label at ALL layers at once (avg over PERIOD_TEMPLATES).

    Returns {name: {'labels': [...], 'mids': (N,), 'embs': (N, n_layers, H)}}
    """
    result = {}
    for name, df in series.items():
        print(f"  [{name}] {len(df)} periods …", flush=True)
        labels, mids, embs_all = [], [], []
        for _, row in df.iterrows():
            lbl = str(row["name"])
            sy  = float(row["start_year"])
            ey  = float(row["end_year"]) if pd.notna(row["end_year"]) else sy
            mid = (sy + ey) / 2
            if mid < YEAR_MIN or mid > YEAR_MAX:
                continue
            tmpl_embs = np.stack([
                embed_fn(t.format(x=lbl))
                for t in period_templates
            ])                              # (n_templates, n_layers, H)
            embs_all.append(tmpl_embs.mean(axis=0))   # (n_layers, H)
            labels.append(lbl)
            mids.append(mid)
        if embs_all:
            result[name] = {
                "labels": labels,
                "mids":   np.array(mids),
                "embs":   np.stack(embs_all),   # (N, n_layers, H)
            }
    return result


def compute_year_dirs(year_embs_full: np.ndarray,
                      years: np.ndarray) -> dict[int, np.ndarray]:
    """PC1 of year-carrier embeddings per layer → {layer: unit_vector (H,)}.

    Oriented so the projection correlates positively with year.
    """
    _, n_layers, _ = year_embs_full.shape
    year_dirs: dict[int, np.ndarray] = {}
    for l in range(n_layers):
        X  = year_embs_full[:, l, :]
        Xc = X - X.mean(0)
        pca = PCA(n_components=1).fit(Xc)
        v   = pca.components_[0]
        if stats.spearmanr(Xc @ v, years)[0] < 0:
            v = -v
        year_dirs[l] = v
    return year_dirs


def compute_series_metrics(period_data_all: dict[str, dict],
                           year_dirs: dict[int, np.ndarray]) -> pd.DataFrame:
    """For each series × layer compute 5 comparable metrics.

    Metrics:
        spearman_rho  — |ρ| between centroid PC1 and mid-year
        centroid_tau  — Kendall τ of pairwise embedding distance vs temporal distance
        knn_purity    — fraction of centroids whose NN is temporally adjacent
        cross_cosine  — |cos(series PC1, year_dir[layer])|
        cross_rho     — Spearman ρ of (centroids @ year_dir) vs mid-year

    Note: bw_ratio is omitted — one centroid per period means within-variance = 0.
    """
    n_layers = len(year_dirs)
    rows = []
    for name, d in period_data_all.items():
        mids     = d["mids"]       # (N,)
        embs_all = d["embs"]       # (N, n_layers, H)
        N        = len(mids)
        if N < 3:
            continue
        # Assign temporal rank as period id (for knn_purity adjacency test)
        order = np.argsort(mids)
        pids  = np.empty(N, dtype=int)
        for rank, idx in enumerate(order):
            pids[idx] = rank

        for l in range(n_layers):
            cents    = embs_all[:, l, :]        # (N, H)
            year_dir = year_dirs[l]

            Xc  = cents - cents.mean(0)
            pca = PCA(n_components=1).fit(Xc)
            pc1 = pca.fit_transform(Xc)[:, 0]
            rho, _   = stats.spearmanr(pc1, mids)

            tau = centroid_tau(cents, mids)
            knn = knn_purity(cents, pids)

            seq_dir = pca.components_[0]
            cos = abs(float(np.dot(seq_dir, year_dir) /
                            (np.linalg.norm(seq_dir) * np.linalg.norm(year_dir) + 1e-12)))
            proj     = cents @ year_dir
            xrho, _  = stats.spearmanr(proj, mids)

            rows.append({
                "series":       name,
                "layer":        l,
                "spearman_rho": abs(float(rho)),
                "centroid_tau": float(tau),
                "knn_purity":   float(knn),
                "cross_cosine": float(cos),
                "cross_rho":    float(xrho),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-cultural clustering test
# ---------------------------------------------------------------------------

def cross_cultural_test(period_data: dict[str, dict],
                        test_years: np.ndarray,
                        window: float = 100.0) -> pd.DataFrame:
    """
    For each test year, collect all period centroids active within ±window years.
    Compute:
      within_dist  — mean pairwise distance among contemporaneous centroids
      between_dist — mean distance to non-contemporaneous centroids (same window away)
      n_series     — how many different series are represented
    Returns DataFrame indexed by test_year.
    """
    all_mids = np.concatenate([v["mids"] for v in period_data.values()])
    all_embs = np.concatenate([v["embs"] for v in period_data.values()], axis=0)
    all_sers = np.concatenate([
        [k] * len(v["mids"]) for k, v in period_data.items()
    ])

    rows = []
    for ty in test_years:
        conc_mask = np.abs(all_mids - ty) <= window
        if conc_mask.sum() < 2:
            rows.append({"year": ty, "within_dist": np.nan,
                         "between_dist": np.nan, "n_series": 0, "n_periods": 0})
            continue

        conc_embs  = all_embs[conc_mask]
        conc_sers  = all_sers[conc_mask]
        n_series   = len(set(conc_sers))

        # Within-contemporaneous pairwise distances
        n = len(conc_embs)
        wd = []
        for i in range(n):
            for j in range(i+1, n):
                wd.append(np.linalg.norm(conc_embs[i] - conc_embs[j]))
        within_dist = float(np.mean(wd)) if wd else np.nan

        # Non-contemporaneous: periods centred 2×window away
        far_mask = np.abs(all_mids - ty) > 2 * window
        if far_mask.sum() >= 2:
            far_embs = all_embs[far_mask]
            bd = []
            for ce in conc_embs:
                bd.extend(np.linalg.norm(far_embs - ce, axis=1).tolist())
            between_dist = float(np.mean(bd))
        else:
            between_dist = np.nan

        rows.append({"year": ty, "within_dist": within_dist,
                     "between_dist": between_dist,
                     "n_series": n_series, "n_periods": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _year_cmap(years: np.ndarray):
    norm = mcolors.Normalize(vmin=years.min(), vmax=years.max())
    return mcm.plasma, norm


def _save(fig, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p.name}")


# ---------------------------------------------------------------------------
# Fig 1: Year manifold shape (PCA 2D, coloured by year)
# ---------------------------------------------------------------------------

def plot_year_manifold_2d(years: np.ndarray, embs: np.ndarray,
                           period_data: dict, layer: int,
                           model_key: str) -> None:
    Xc  = embs - embs.mean(0)
    pca = PCA(n_components=3).fit(Xc)
    C   = pca.transform(Xc)   # (n_years, 3)

    cmap, norm = _year_cmap(years)

    # Project period centroids onto same PCA space
    per_proj = {}
    for name, d in period_data.items():
        Xp = d["embs"] - embs.mean(0)
        per_proj[name] = pca.transform(Xp)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (xi, yi, xlabel, ylabel) in zip(axes, [
        (0, 1, f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"),
        (0, 2, f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"),
    ]):
        # Year trajectory
        sc = ax.scatter(C[:, xi], C[:, yi], c=years, cmap=cmap, norm=norm,
                        s=22, zorder=5, edgecolors="none", alpha=0.85)
        ax.plot(C[:, xi], C[:, yi], lw=0.6, color="grey", alpha=0.4, zorder=3)
        plt.colorbar(sc, ax=ax, label="Year (CE)", fraction=0.03, pad=0.01)

        # Period centroids (small dots, per series colour)
        for name, proj in per_proj.items():
            color = SERIES_COLORS.get(name, "#888888")
            mids  = period_data[name]["mids"]
            ax.scatter(proj[:, xi], proj[:, yi],
                       c=[color] * len(proj), s=12, alpha=0.55,
                       marker="^", zorder=6, linewidths=0, label=name)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(linestyle=":", alpha=0.3)

    axes[0].legend(fontsize=5.5, ncol=2, loc="lower right",
                   title="series (triangles)", title_fontsize=6)
    fig.suptitle(
        f"Year manifold shape — {MODEL_NAMES[model_key]} L{layer}\n"
        f"Year-carrier embeddings (circles, colour=year) + period centroids (triangles)\n"
        f"Total variance explained: PC1–2 = "
        f"{sum(pca.explained_variance_ratio_[:2]):.1%}  "
        f"PC1–3 = {sum(pca.explained_variance_ratio_[:3]):.1%}",
        fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"manifold_shape_{model_key}_L{layer}.png")


# ---------------------------------------------------------------------------
# Fig 2: Velocity + curvature along the trajectory
# ---------------------------------------------------------------------------

def plot_velocity_curvature(years: np.ndarray, embs: np.ndarray,
                             period_data: dict, layer: int,
                             model_key: str) -> None:
    vel   = velocity(embs, years)
    curv  = curvature(embs)
    ldim  = local_dim(embs)

    vel_years  = (years[:-1] + years[1:]) / 2
    curv_years = years[1:-1]

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)

    # Velocity
    ax0 = axes[0]
    ax0.fill_between(vel_years, vel, alpha=0.3, color="#4575b4")
    ax0.plot(vel_years, vel, lw=1.2, color="#4575b4")
    ax0.set_ylabel("Velocity\n(‖Δemb‖ / year)", fontsize=9)
    ax0.grid(linestyle=":", alpha=0.3)

    # Mark velocity peaks (epochs of fast change)
    peaks, _ = find_peaks(vel, prominence=vel.std() * 0.8)
    for p in peaks:
        ax0.axvline(vel_years[p], color="red", lw=0.8, alpha=0.6)
        ax0.text(vel_years[p], vel[p] * 1.02, f"{int(vel_years[p])}",
                 fontsize=5.5, ha="center", color="red")

    # Curvature
    ax1 = axes[1]
    ax1.fill_between(curv_years, curv, alpha=0.3, color="#d73027")
    ax1.plot(curv_years, curv, lw=1.2, color="#d73027")
    ax1.axhline(90, color="grey", lw=0.8, linestyle="--", label="90°")
    ax1.set_ylabel("Curvature\n(turning angle, degrees)", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(linestyle=":", alpha=0.3)

    # Local dimensionality
    ax2 = axes[2]
    ax2.fill_between(years, ldim, alpha=0.3, color="#1a9641")
    ax2.plot(years, ldim, lw=1.2, color="#1a9641")
    ax2.set_ylabel("Local PR\n(intrinsic dim)", fontsize=9)
    ax2.set_xlabel("Year (negative = BCE)", fontsize=9)
    ax2.grid(linestyle=":", alpha=0.3)

    # Shade major historical eras
    eras = [(-600, 0, "Antiquity"), (0, 500, "Late\nAntiquity"),
            (500, 1000, "Early\nMedieval"), (1000, 1500, "High/Late\nMedieval"),
            (1500, 1800, "Early\nModern"), (1800, 1925, "Modern")]
    era_colors = ["#ffffcc", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    for (s, e, lbl), col in zip(eras, era_colors):
        if e < YEAR_MIN or s > YEAR_MAX:
            continue
        for ax in axes:
            ax.axvspan(max(s, YEAR_MIN), min(e, YEAR_MAX),
                       alpha=0.06, color=col, zorder=0)
        axes[0].text((max(s, YEAR_MIN) + min(e, YEAR_MAX)) / 2,
                     ax0.get_ylim()[1] * 0.85, lbl,
                     ha="center", fontsize=6, color="#666666")

    fig.suptitle(
        f"Temporal velocity, curvature & local dimensionality\n"
        f"{MODEL_NAMES[model_key]} L{layer}  ·  red verticals = velocity peaks",
        fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"manifold_dynamics_{model_key}_L{layer}.png")


# ---------------------------------------------------------------------------
# Fig 3: Cross-cultural clustering
# ---------------------------------------------------------------------------

def plot_cross_cultural(period_data: dict, years: np.ndarray,
                        layer: int, model_key: str) -> None:
    # Overlap region: where ≥3 series are active
    test_years = years[(years >= 0) & (years <= 1900)]
    cc = cross_cultural_test(period_data, test_years, window=75)
    cc = cc[cc["n_series"] >= 2].dropna(subset=["within_dist", "between_dist"])

    if len(cc) < 3:
        print("  Insufficient overlap for cross-cultural test")
        return

    ratio = cc["between_dist"] / cc["within_dist"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Within vs between distances
    ax0 = axes[0]
    ax0.plot(cc["year"], cc["within_dist"],  lw=1.6, color="#4575b4",
             label="within-contemporaneous (same ±75yr)")
    ax0.plot(cc["year"], cc["between_dist"], lw=1.6, color="#d73027",
             label="between-noncontemporaneous (>150yr away)")
    ax0.fill_between(cc["year"], cc["within_dist"], cc["between_dist"],
                     where=cc["between_dist"] > cc["within_dist"],
                     alpha=0.15, color="#1a9641", label="separation (good)")
    ax0.set_ylabel("Mean pairwise L2 distance", fontsize=9)
    ax0.legend(fontsize=8)
    ax0.grid(linestyle=":", alpha=0.3)

    # Separation ratio
    ax1 = axes[1]
    ax1.fill_between(cc["year"], ratio, 1, where=ratio > 1,
                     alpha=0.3, color="#1a9641")
    ax1.fill_between(cc["year"], ratio, 1, where=ratio < 1,
                     alpha=0.3, color="#d73027")
    ax1.plot(cc["year"], ratio, lw=1.4, color="black")
    ax1.axhline(1, lw=0.8, color="grey", linestyle="--")
    ax1.set_ylabel("Separation ratio\n(between / within)", fontsize=9)
    ax1.set_xlabel("Year (CE)", fontsize=9)
    ax1.grid(linestyle=":", alpha=0.3)

    # Annotate n_series
    for _, row in cc.iloc[::6].iterrows():
        ax1.text(row["year"], ratio[cc["year"] == row["year"]].values[0] + 0.02,
                 f"n={int(row['n_series'])}", fontsize=5, ha="center")

    fig.suptitle(
        f"Cross-cultural contemporaneous clustering — {MODEL_NAMES[model_key]} L{layer}\n"
        "Do period embeddings from different series cluster more tightly when contemporaneous?\n"
        "Ratio > 1 = yes (contemporaneous periods are closer than non-contemporaneous)",
        fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"manifold_cross_cultural_{model_key}_L{layer}.png")


# ---------------------------------------------------------------------------
# Fig 4: Joint MDS of all period centroids
# ---------------------------------------------------------------------------

def plot_joint_mds(period_data: dict, layer: int, model_key: str) -> None:
    all_embs, all_mids, all_sers = [], [], []
    for name, d in period_data.items():
        all_embs.append(d["embs"])
        all_mids.extend(d["mids"].tolist())
        all_sers.extend([name] * len(d["mids"]))

    X    = np.concatenate(all_embs, axis=0)
    mids = np.array(all_mids)
    sers = np.array(all_sers)

    if len(X) < 4:
        return

    Xs = StandardScaler().fit_transform(X)
    mds = MDS(n_components=2, dissimilarity="euclidean",
              random_state=42, n_init=4, max_iter=400)
    C   = mds.fit_transform(Xs)

    cmap, norm = _year_cmap(mids)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: colour by year
    ax = axes[0]
    for name in sorted(SERIES_COLORS):
        if name not in period_data:
            continue
        mask   = sers == name
        color  = SERIES_COLORS[name]
        sc_pts = ax.scatter(C[mask, 0], C[mask, 1],
                            c=mids[mask], cmap=cmap, norm=norm,
                            s=20, alpha=0.8, marker="o", linewidths=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Year (CE)", fraction=0.03, pad=0.01)
    ax.set_title("Coloured by year — universal manifold test\n"
                 "(if manifold is shared, contemporaneous points cluster "
                 "regardless of series)", fontsize=8)
    ax.set_xlabel("MDS dim 1"); ax.set_ylabel("MDS dim 2")
    ax.grid(linestyle=":", alpha=0.3)

    # Right: colour by series
    ax2 = axes[1]
    for name in sorted(period_data):
        mask  = sers == name
        color = SERIES_COLORS.get(name, "#888888")
        ax2.scatter(C[mask, 0], C[mask, 1], color=color, s=20, alpha=0.8,
                    linewidths=0, label=name)
    ax2.legend(fontsize=6, ncol=2, loc="best")
    ax2.set_title("Coloured by series — separation test\n"
                  "(if series-coloured clusters emerge, the manifold is series-specific)",
                  fontsize=8)
    ax2.set_xlabel("MDS dim 1"); ax2.set_ylabel("MDS dim 2")
    ax2.grid(linestyle=":", alpha=0.3)

    fig.suptitle(
        f"Joint MDS of all period centroids — {MODEL_NAMES[model_key]} L{layer}\n"
        f"N={len(X)} period centroids from {len(period_data)} series  ·  "
        "Stress=" + f"{mds.stress_:.1f}",
        fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"manifold_joint_mds_{model_key}_L{layer}.png")


# ---------------------------------------------------------------------------
# Fig 5: Period centroids projected onto year-PC1 vs actual year
# ---------------------------------------------------------------------------

def plot_pc1_projection(years: np.ndarray, embs: np.ndarray,
                        period_data: dict, layer: int, model_key: str) -> None:
    Xc  = embs - embs.mean(0)
    pca = PCA(n_components=1).fit(Xc)
    pc1 = pca.components_[0]

    # Orient: positive = later
    proj_years = Xc @ pc1
    if stats.spearmanr(proj_years, years)[0] < 0:
        pc1 = -pc1

    fig, ax = plt.subplots(figsize=(12, 8))

    for name, d in sorted(period_data.items()):
        Xp   = d["embs"] - embs.mean(0)
        proj = Xp @ pc1
        mids = d["mids"]
        color = SERIES_COLORS.get(name, "#888888")

        ax.scatter(mids, proj, color=color, s=16, alpha=0.65,
                   linewidths=0, label=name)

        # Trend line per series
        if len(mids) >= 4:
            m, b, r, p, _ = stats.linregress(mids, proj)
            xfit = np.linspace(mids.min(), mids.max(), 100)
            ax.plot(xfit, m * xfit + b, color=color, lw=1.2, alpha=0.5)

    # Year-carrier trajectory
    proj_yc = (embs - embs.mean(0)) @ pc1
    ax.plot(years, proj_yc, lw=2, color="black", alpha=0.6,
            label="year-carrier PC1", zorder=10)

    ax.set_xlabel("Period mid-year (CE; negative = BCE)", fontsize=9)
    ax.set_ylabel("Projection onto year manifold PC1", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(linestyle=":", alpha=0.3)
    ax.set_title(
        f"Period centroids projected onto year manifold PC1 — "
        f"{MODEL_NAMES[model_key]} L{layer}\n"
        "Black = year-carrier trajectory  ·  Coloured = period centroids per series  ·  "
        "Line = per-series trend",
        fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"manifold_pc1_projection_{model_key}_L{layer}.png")


# ---------------------------------------------------------------------------
# Fig 6: Velocity per series (how fast do consecutive periods move?)
# ---------------------------------------------------------------------------

def plot_series_velocity(period_data: dict, layer: int, model_key: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, d in sorted(period_data.items()):
        mids = d["mids"]
        embs = d["embs"]
        if len(mids) < 3:
            continue
        order = np.argsort(mids)
        mids_s = mids[order]
        embs_s = embs[order]
        vel = velocity(embs_s, mids_s)
        vel_mid = (mids_s[:-1] + mids_s[1:]) / 2
        # Only plot where velocity is finite
        finite = np.isfinite(vel)
        if finite.sum() < 2:
            continue
        color = SERIES_COLORS.get(name, "#888888")
        ax.plot(vel_mid[finite], vel[finite], lw=1.2, alpha=0.7,
                color=color, label=name)

    # Robust y-limit: 99th percentile across all series
    ax.set_xlabel("Year (CE; negative = BCE)", fontsize=9)
    ax.set_ylabel("Series velocity (‖Δemb‖ / year)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    ax.grid(linestyle=":", alpha=0.3)
    ax.set_title(
        f"Temporal velocity per series — {MODEL_NAMES[model_key]} L{layer}\n"
        "How fast does each series' embedding move per year through history",
        fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"manifold_series_velocity_{model_key}_L{layer}.png")


# ---------------------------------------------------------------------------
# Fig 7: Layer comparison — how does the manifold shape change across layers?
# ---------------------------------------------------------------------------

def plot_layer_comparison(years_traj: np.ndarray, all_layer_embs: dict[int, np.ndarray],
                           model_key: str) -> None:
    layers_to_show = [0, 2, 4, 6, 7, 9, 11, 12]
    layers_to_show = [l for l in layers_to_show if l in all_layer_embs]
    n = len(layers_to_show)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    cmap, norm = _year_cmap(years_traj)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax_i, layer in enumerate(layers_to_show):
        ax  = axes[ax_i]
        X   = all_layer_embs[layer]
        Xc  = X - X.mean(0)
        pca = PCA(n_components=2).fit(Xc)
        C   = pca.transform(Xc)
        ev  = pca.explained_variance_ratio_

        sc = ax.scatter(C[:, 0], C[:, 1], c=years_traj, cmap=cmap, norm=norm,
                        s=18, alpha=0.85, linewidths=0)
        ax.plot(C[:, 0], C[:, 1], lw=0.5, color="grey", alpha=0.3)
        ax.set_title(f"L{layer}  ({ev[0]:.0%}+{ev[1]:.0%}={ev[0]+ev[1]:.0%})",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(linestyle=":", alpha=0.25)

    # Hide unused axes
    for ax_i in range(len(layers_to_show), len(axes)):
        axes[ax_i].set_visible(False)

    plt.colorbar(sc, ax=axes[:len(layers_to_show)], label="Year (CE)",
                 fraction=0.015, pad=0.02, shrink=0.6)
    fig.suptitle(
        f"Year manifold shape across layers — {MODEL_NAMES[model_key]}\n"
        "Year-carrier embeddings in PC1-PC2  ·  colour = year",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"manifold_layer_comparison_{model_key}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=["bert", "macberth", "sikubert", "openai"],
                        default="bert")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for period analysis (default: 7 for bert/sikubert, "
                             "4 for macberth, 0 for openai)")
    args = parser.parse_args()

    model_key = args.model
    embed_fn, n_layers = load_embedder(model_key)
    default_layer = {"bert": 7, "macberth": 4, "sikubert": 7, "openai": 0}.get(model_key, 7)
    layer = args.layer if args.layer is not None else default_layer

    # Use classical Chinese templates for models trained on Chinese corpora
    if model_key == "sikubert":
        year_tmpls   = SHARED_TEMPLATES_ZH
        period_tmpls = SHARED_TEMPLATES_ZH
    else:
        year_tmpls   = YEAR_TEMPLATES
        period_tmpls = PERIOD_TEMPLATES

    print("Loading time series …")
    series = load_series()
    print(f"\n{len(series)} series loaded.\n")

    # ── Year trajectory at ALL layers (single pass set) ───────────────────
    print("Computing year trajectory at all layers …")
    years_traj, year_embs_full = compute_year_trajectory_all_layers(embed_fn, year_tmpls)
    # Slice to target layer for existing figure functions
    embs_traj = year_embs_full[:, layer, :]
    # Per-layer dict for the layer-comparison figure
    all_layer_embs: dict[int, np.ndarray] = {
        l: year_embs_full[:, l, :] for l in range(n_layers)
    }

    # ── Per-year-direction per layer ──────────────────────────────────────
    year_dirs = compute_year_dirs(year_embs_full, years_traj)

    # ── Period centroids at ALL layers ────────────────────────────────────
    print("\nEmbedding period centroids at all layers …")
    period_data_all = compute_period_centroids_all_layers(series, embed_fn, period_tmpls)

    # Add year-carrier as an explicit series (all layers)
    period_data_all["year_carrier"] = {
        "labels": [str(int(y)) for y in years_traj],
        "mids":   years_traj.copy(),
        "embs":   year_embs_full,    # (n_years, n_layers, H)
    }

    # Slice to target layer for figure functions that expect (N, H)
    period_data: dict[str, dict] = {
        name: {
            "labels": d["labels"],
            "mids":   d["mids"],
            "embs":   d["embs"][:, layer, :],
        }
        for name, d in period_data_all.items()
    }
    print(f"\n{sum(len(d['mids']) for d in period_data.values())} period centroids total "
          f"(includes {len(years_traj)} year-carrier points as baseline).\n")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("Generating figures …")
    plot_year_manifold_2d(years_traj, embs_traj, period_data, layer, model_key)
    plot_velocity_curvature(years_traj, embs_traj, period_data, layer, model_key)
    plot_cross_cultural(period_data, years_traj, layer, model_key)
    plot_joint_mds(period_data, layer, model_key)
    plot_pc1_projection(years_traj, embs_traj, period_data, layer, model_key)
    plot_series_velocity(period_data, layer, model_key)
    if n_layers > 1:
        plot_layer_comparison(years_traj, all_layer_embs, model_key)

    # ── Layer sweep metrics → CSV ─────────────────────────────────────────
    print("\nComputing per-series layer sweep metrics …")
    metrics_df = compute_series_metrics(period_data_all, year_dirs)
    csv_path   = OUT_DIR / f"manifold_results_{model_key}.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"  → {csv_path.name}  ({len(metrics_df)} rows, "
          f"{metrics_df['series'].nunique()} series × {n_layers} layers)")

    print("\nDone.")


if __name__ == "__main__":
    main()
