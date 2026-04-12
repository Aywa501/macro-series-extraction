"""Stage 01c — Descriptive geometry of time in BERT embedding space.

Explores what temporal structure actually looks like in the Penn embeddings
beyond the linear Ridge assumption.  Produces a set of diagnostics that
together answer: how good is linearity, and what are the alternatives?

Analyses
--------
1.  Decade centroids
    Compute mean embedding per decade at each layer.  These are the "anchor
    points" of the temporal manifold.  If time is linear, centroids should
    lie on a straight line; if curved, a higher-dimensional manifold.

2.  PCA on decade centroids
    How many principal components are needed to explain the variance in
    the centroid sequence?  Linear → 1 PC dominates.  Curved → 2–3 PCs.
    Reports cumulative explained variance and the angle between PC1 and
    the Ridge temporal direction (they should agree if Ridge is finding
    the right axis).

3.  Linearity test: Ridge vs kernel ridge vs kNN
    Fit three probes on the same embeddings and compare held-out R².
    If kNN >> Ridge, the temporal manifold is non-linear.

4.  Between- vs within-decade variance
    What fraction of total embedding variance is explained by decade?
    Low ratio → embeddings are dominated by content, not time.
    High ratio → time is the dominant organising axis.

5.  Centroid distance regularity
    Compute pairwise Euclidean distances between consecutive decade
    centroids.  If uniform, time is metrically regular in embedding space.
    If irregular, some periods change faster than others.

6.  Temporal consistency across text types
    Split Penn sentences by estimated genre (proxy: file prefix cm=ME,
    otherwise EME/MBE).  Fit separate Ridge directions per sub-corpus and
    compute the angle between them.  Large angle → temporal direction is
    genre-specific, not universal.

7.  Predicted-year calibration
    For each decade, plot mean predicted year vs actual year.  Should be
    on the diagonal if the probe is well-calibrated.

Outputs
-------
data/geometry/decade_centroids_L{k}.npy    — per layer
data/geometry/geometry_summary.csv         — all numeric results
data/geometry/plots/                       — all figures

Usage
-----
    python src/01c_temporal_geometry.py
    python src/01c_temporal_geometry.py --layer 12
    python src/01c_temporal_geometry.py --dry-run
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
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)

DECADE_SIZE = 25   # years per bin (25 = generation-level granularity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decade_bin(year: float, size: int = DECADE_SIZE) -> int:
    return int(year // size) * size


def _between_within_variance(X: np.ndarray, labels: np.ndarray) -> float:
    """Ratio of between-group variance to total variance (η²)."""
    grand_mean = X.mean(axis=0)
    unique = np.unique(labels)
    between = sum(
        np.sum(labels == g) * np.linalg.norm(X[labels == g].mean(axis=0) - grand_mean) ** 2
        for g in unique
    )
    total = np.sum(np.linalg.norm(X - grand_mean, axis=1) ** 2)
    return float(between / total) if total > 0 else 0.0


_PROBE_MAX_N = 5_000   # KernelRidge builds an n×n matrix — cap to avoid OOM


def _fit_probes(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict[str, float]:
    """Cross-validated R² for Ridge, kernel-Ridge (RBF), and kNN.

    Always subsamples to at most _PROBE_MAX_N rows: KernelRidge with an RBF
    kernel requires an n×n kernel matrix, which exceeds available RAM for the
    full Penn corpus (~100k sentences).
    """
    if len(X) > _PROBE_MAX_N:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(X), _PROBE_MAX_N, replace=False)
        X, y = X[idx], y[idx]
        logger.debug("_fit_probes: subsampled to %d rows", _PROBE_MAX_N)

    scaler = StandardScaler(with_std=False)
    X_c = scaler.fit_transform(X)

    ridge  = RidgeCV(alphas=np.logspace(-3, 6, 20))
    kridge = KernelRidge(kernel="rbf", alpha=1.0, gamma=0.001)
    knn    = KNeighborsRegressor(n_neighbors=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2_ridge  = cross_val_score(ridge,  X_c, y, cv=cv, scoring="r2").mean()
        r2_kridge = cross_val_score(kridge, X_c, y, cv=cv, scoring="r2").mean()
        r2_knn    = cross_val_score(knn,    X_c, y, cv=cv, scoring="r2").mean()

    return {"ridge_cv_r2": r2_ridge, "kernel_ridge_cv_r2": r2_kridge, "knn_cv_r2": r2_knn}


# ---------------------------------------------------------------------------
# Main analyses
# ---------------------------------------------------------------------------

def analyse_layer(
    X: np.ndarray,        # (n, 768)
    years: np.ndarray,    # (n,)
    temporal_dir: np.ndarray,  # (768,) unit vector
    layer_idx: int,
    plots_dir: Path,
    dry_run: bool,
) -> dict:
    layer = layer_idx + 1
    logger.info("Analysing layer %d …", layer)
    result: dict = {"layer": layer}

    # ---- 1. Decade centroids ----
    decades = np.array([_decade_bin(y) for y in years])
    unique_decades = np.sort(np.unique(decades))
    centroids = np.stack([X[decades == d].mean(axis=0) for d in unique_decades])
    # (n_decades, 768)

    # ---- 2. PCA on centroids ----
    pca = PCA(n_components=min(10, len(unique_decades) - 1))
    pca.fit(centroids)
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    # Angle between PC1 and temporal direction
    pc1 = pca.components_[0]
    cos = float(np.clip(abs(np.dot(pc1, temporal_dir)), 0, 1))
    angle_pc1_temporal = float(np.degrees(np.arccos(cos)))

    result["pca_pc1_var"]           = float(evr[0])
    result["pca_pc2_var"]           = float(evr[1]) if len(evr) > 1 else 0.0
    result["pca_pc3_var"]           = float(evr[2]) if len(evr) > 2 else 0.0
    result["pca_cum_var_2pc"]       = float(cum_evr[1]) if len(cum_evr) > 1 else float(evr[0])
    result["angle_pc1_temporal_deg"] = angle_pc1_temporal

    logger.info(
        "  PCA: PC1=%.1f%%  PC2=%.1f%%  PC3=%.1f%%  "
        "angle(PC1, temporal_dir)=%.1f°",
        evr[0]*100, (evr[1] if len(evr)>1 else 0)*100,
        (evr[2] if len(evr)>2 else 0)*100, angle_pc1_temporal,
    )

    # ---- 3. Probe comparison ----
    # Always runs; _fit_probes subsamples internally to avoid OOM.
    probe_r2s = _fit_probes(X, years)
    result.update(probe_r2s)
    logger.info(
        "  CV-R²: Ridge=%.3f  KernelRidge=%.3f  kNN=%.3f",
        probe_r2s["ridge_cv_r2"], probe_r2s["kernel_ridge_cv_r2"],
        probe_r2s["knn_cv_r2"],
    )

    # ---- 4. Between/within-decade variance ----
    eta2 = _between_within_variance(X, decades)
    result["between_within_eta2"] = eta2
    logger.info("  Between/within decade variance (η²): %.4f", eta2)

    # ---- 5. Centroid distance regularity ----
    dists = np.array([
        np.linalg.norm(centroids[i+1] - centroids[i])
        for i in range(len(centroids) - 1)
    ])
    result["centroid_dist_mean"] = float(dists.mean())
    result["centroid_dist_std"]  = float(dists.std())
    result["centroid_dist_cv"]   = float(dists.std() / dists.mean()) if dists.mean() > 0 else 0.0
    logger.info(
        "  Centroid distances: mean=%.4f  std=%.4f  CV=%.3f",
        dists.mean(), dists.std(), result["centroid_dist_cv"],
    )

    # ---- 6. Predicted year calibration ----
    scaler = StandardScaler(with_std=False)
    X_c = scaler.fit_transform(X)
    ridge = RidgeCV(alphas=np.logspace(-3, 6, 20), fit_intercept=True)
    ridge.fit(X_c, years)
    y_pred = ridge.predict(X_c)

    calib_rows = []
    for d in unique_decades:
        mask = decades == d
        calib_rows.append({
            "decade": d,
            "actual_year_mean": float(years[mask].mean()),
            "predicted_year_mean": float(y_pred[mask].mean()),
            "n": int(mask.sum()),
        })
    calib_df = pd.DataFrame(calib_rows)
    calib_corr = float(np.corrcoef(
        calib_df["actual_year_mean"], calib_df["predicted_year_mean"]
    )[0, 1])
    result["calibration_r"] = calib_corr

    # ---- Plots ----
    # Plot A: PCA of centroids (first 2 PCs), coloured by decade
    pca2 = PCA(n_components=2)
    C2 = pca2.fit_transform(centroids)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(C2[:, 0], C2[:, 1], c=unique_decades, cmap="plasma", s=60)
    plt.colorbar(sc, ax=ax, label="Decade start year")
    for j, d in enumerate(unique_decades[::4]):
        idx = list(unique_decades).index(d)
        ax.annotate(str(d), C2[idx], fontsize=7, alpha=0.7)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({(evr[1] if len(evr)>1 else 0)*100:.1f}% var)")
    ax.set_title(f"Decade centroids in PCA space — Layer {layer}")
    fig.tight_layout()
    fig.savefig(plots_dir / f"centroids_pca_L{layer:02d}.png", dpi=130)
    plt.close(fig)

    # Plot B: Predicted vs actual year
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(calib_df["actual_year_mean"], calib_df["predicted_year_mean"],
               s=20, alpha=0.7)
    lo = min(calib_df["actual_year_mean"].min(), calib_df["predicted_year_mean"].min())
    hi = max(calib_df["actual_year_mean"].max(), calib_df["predicted_year_mean"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Actual decade mean year")
    ax.set_ylabel("Predicted year (Ridge)")
    ax.set_title(f"Temporal probe calibration — Layer {layer}  (r={calib_corr:.3f})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"calibration_L{layer:02d}.png", dpi=130)
    plt.close(fig)

    # Plot C: Consecutive centroid distances across time
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(unique_decades[:-1], dists, width=DECADE_SIZE * 0.8, alpha=0.7)
    ax.axhline(dists.mean(), color="red", linestyle="--", linewidth=0.8,
               label=f"Mean = {dists.mean():.3f}")
    ax.set_xlabel("Decade start year")
    ax.set_ylabel("Distance to next centroid")
    ax.set_title(f"Rate of change in embedding space — Layer {layer}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"centroid_distances_L{layer:02d}.png", dpi=130)
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict, layers_to_analyse: list[int], dry_run: bool) -> None:
    paths    = cfg["paths"]
    n_layers = cfg["model"]["n_layers"]

    emb_path  = PROJECT_ROOT / paths["penn_embeddings"]
    sent_path = PROJECT_ROOT / paths["penn_sentences"]
    dirs_path = PROJECT_ROOT / paths["temporal_directions"]

    geom_dir  = PROJECT_ROOT / paths["geometry_dir"]
    plots_dir = geom_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Penn embeddings …")
    embeddings = np.load(emb_path)                      # (n, 12, 768)
    df         = pd.read_parquet(sent_path)
    years      = df["year"].values.astype(float)
    dirs       = np.load(dirs_path).astype(np.float64)  # (12, 768)

    if dry_run:
        idx = np.random.choice(len(years), min(5000, len(years)), replace=False)
        embeddings = embeddings[idx]
        years      = years[idx]
        logger.info("[dry-run] Subsampled to %d sentences", len(years))

    all_results = []
    for layer_idx in layers_to_analyse:
        X = embeddings[:, layer_idx, :]
        result = analyse_layer(
            X, years, dirs[layer_idx], layer_idx, plots_dir, dry_run
        )
        all_results.append(result)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(geom_dir / "geometry_summary.csv", index=False)
    logger.info("Saved → %s", geom_dir / "geometry_summary.csv")

    print("\n=== Temporal geometry summary ===\n")
    cols = ["layer", "pca_pc1_var", "pca_pc2_var", "pca_cum_var_2pc",
            "angle_pc1_temporal_deg", "between_within_eta2",
            "centroid_dist_cv", "calibration_r",
            "ridge_cv_r2", "kernel_ridge_cv_r2", "knn_cv_r2"]
    print(summary_df[cols].to_string(index=False))

    print("\n--- Key questions ---")
    for _, row in summary_df.iterrows():
        linearity = "LINEAR" if row["pca_pc1_var"] > 0.8 else \
                    "MOSTLY LINEAR" if row["pca_pc1_var"] > 0.6 else "NON-LINEAR"
        alignment = "ALIGNED" if row["angle_pc1_temporal_deg"] < 20 else \
                    "MISALIGNED" if row["angle_pc1_temporal_deg"] > 45 else "PARTIAL"
        print(f"  Layer {int(row['layer']):2d}: geometry={linearity:14s}  "
              f"PC1↔temporal={alignment:9s}  "
              f"η²={row['between_within_eta2']:.3f}  "
              f"dist_CV={row['centroid_dist_cv']:.3f}")


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    n_layers = cfg["model"]["n_layers"]

    parser = argparse.ArgumentParser(
        description="Stage 01c: Descriptive geometry of time in BERT embedding space."
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Analyse a single layer (1-indexed). Default: all layers.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Subsample to 5000 sentences, skip slow probes.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.layer is not None:
        layers = [args.layer - 1]   # convert to 0-indexed
    else:
        layers = list(range(n_layers))

    run(cfg, layers, args.dry_run)


if __name__ == "__main__":
    main()
