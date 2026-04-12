"""Stage 01b — Temporal regression parameters and residualisation.

Loads the Penn embeddings saved by Stage 01 and fits Ridge regression
(year ~ CLS_embedding) at each of the 12 BERT layers.  Extracts and saves
every parameter needed for year-calibrated interventions in Stage 04:

  - unit temporal directions          (12, 768)
  - coefficient norms                 (12,)   — converts λ to years
  - regression intercepts             (12,)   — predicted year at corpus mean
  - mean embeddings                   (12, 768)
  - vocabulary-residualised directions (12, 768)
  - diagnostics CSV                   R², alpha, coef_norm per layer

Why coef_norm matters
---------------------
The Ridge regression is fitted on mean-centred embeddings:

    year ≈ intercept + coef @ (embedding − mean_embedding)

The raw coefficient vector has L2 norm = coef_norm.  Writing it as
coef = coef_norm × unit_dir, we get:

    year_hat = intercept + coef_norm × (unit_dir @ (e − mean_embedding))

To shift a given embedding e to a target year Y, the required perturbation is:

    Δembedding = ((Y − year_hat) / coef_norm) × unit_dir

This is how Stage 04 translates λ_years into embedding-space shifts.

Residualisation
---------------
Layers 2–12 directions are projected to remove any component along the
layer-1 (vocabulary) axis.  Empirically the angle is ~87° so the effect
is negligible, but the clean directions are saved for completeness.

Outputs
-------
data/temporal/directions.npy       — (12, 768) unit vectors
data/temporal/direction_final.npy          — (768,) layer-12 direction
data/temporal/coef_norms.npy                — (12,)
data/temporal/intercepts.npy                — (12,)
data/temporal/mean_embeddings.npy           — (12, 768)
data/temporal/directions_clean.npy  — (12, 768)
data/temporal/direction_final_clean.npy    — (768,)
data/temporal/diagnostics.csv     — R², alpha, coef_norm
data/temporal/params_summary.csv            — all params in one table

Usage
-----
    python src/01b_temporal_params.py
    python src/01b_temporal_params.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)


def run(cfg: dict, dry_run: bool) -> None:
    paths    = cfg["paths"]
    n_layers = cfg["model"]["n_layers"]

    emb_path   = PROJECT_ROOT / paths["penn_embeddings"]
    sent_path  = PROJECT_ROOT / paths["penn_sentences"]
    out_dir    = emb_path.parent

    dir_layer_path   = PROJECT_ROOT / paths["temporal_directions"]
    dir_out_path     = PROJECT_ROOT / paths["temporal_dir_output"]
    diag_path        = PROJECT_ROOT / paths["temporal_diagnostics"]
    norms_path       = PROJECT_ROOT / paths["temporal_coef_norms"]
    intercepts_path  = out_dir / "temporal_intercepts.npy"
    means_path       = out_dir / "temporal_mean_embeddings.npy"
    clean_layer_path = PROJECT_ROOT / paths["temporal_directions_clean"]
    clean_out_path   = PROJECT_ROOT / paths["temporal_dir_output_clean"]
    summary_path     = out_dir / "temporal_params_summary.csv"

    # --- Load embeddings and years ---
    logger.info("Loading Penn embeddings from %s …", emb_path)
    embeddings = np.load(emb_path)                      # (n, 12, 768)
    df         = pd.read_parquet(sent_path)
    years      = df["year"].values.astype(float)
    logger.info("Loaded %d sentences (years %d–%d)", len(years), int(years.min()), int(years.max()))

    if dry_run:
        idx        = np.random.choice(len(years), min(2000, len(years)), replace=False)
        embeddings = embeddings[idx]
        years      = years[idx]
        logger.info("[dry-run] Subsampled to %d sentences", len(years))

    # --- Fit Ridge at each layer ---
    alphas = np.logspace(-3, 6, 20)

    directions  = np.zeros((n_layers, 768), dtype=np.float64)
    coef_norms  = np.zeros(n_layers,        dtype=np.float64)
    intercepts  = np.zeros(n_layers,        dtype=np.float64)
    means       = np.zeros((n_layers, 768), dtype=np.float64)
    dirs_clean  = np.zeros((n_layers, 768), dtype=np.float64)
    diag_rows   = []
    summary_rows = []

    vocab_dir = None   # set from layer 0, used for residualisation of layers 1–11

    for i in range(n_layers):
        X      = embeddings[:, i, :].astype(np.float64)
        scaler = StandardScaler(with_std=False)
        X_c    = scaler.fit_transform(X)
        mean_i = scaler.mean_

        cv = RidgeCV(alphas=alphas, fit_intercept=True, scoring="r2")
        cv.fit(X_c, years)

        coef      = cv.coef_                              # (768,)
        norm      = float(np.linalg.norm(coef))
        unit      = coef / norm if norm > 1e-10 else coef.copy()
        intercept = float(cv.intercept_)

        y_pred = cv.predict(X_c)
        ss_res = float(np.sum((years - y_pred) ** 2))
        ss_tot = float(np.sum((years - years.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        directions[i] = unit
        coef_norms[i] = norm
        intercepts[i] = intercept
        means[i]      = mean_i

        # Residualise against layer-0 vocabulary direction
        if i == 0:
            vocab_dir    = unit.copy()
            unit_clean   = unit.copy()
            angle_deg    = 0.0
            resid_norm   = 1.0
        else:
            cos_angle  = float(np.clip(np.dot(unit, vocab_dir), -1, 1))
            angle_deg  = float(np.degrees(np.arccos(abs(cos_angle))))
            residual   = unit - np.dot(unit, vocab_dir) * vocab_dir
            resid_norm = float(np.linalg.norm(residual))
            unit_clean = residual / resid_norm if resid_norm > 1e-10 else unit.copy()

        dirs_clean[i] = unit_clean

        logger.info(
            "  Layer %2d: R²=%.4f  α=%.3g  coef_norm=%.1f  "
            "intercept=%.0f  angle_vocab=%.1f°",
            i + 1, r2, cv.alpha_, norm, intercept, angle_deg,
        )

        diag_rows.append({
            "layer": i + 1, "r2": r2, "alpha": float(cv.alpha_),
            "coef_norm": norm, "years_per_unit_lambda": norm,
        })
        summary_rows.append({
            "layer":                i + 1,
            "intercept":            intercept,
            "coef_norm":            norm,
            "years_per_unit_lam":   norm,
            "angle_from_vocab_deg": angle_deg,
            "residual_norm":        resid_norm,
            "r2":                   r2,
            "alpha":                float(cv.alpha_),
        })

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(dir_layer_path,   directions.astype(np.float32))
    np.save(dir_out_path,     directions[n_layers - 1].astype(np.float32))
    np.save(norms_path,       coef_norms)
    np.save(intercepts_path,  intercepts)
    np.save(means_path,       means.astype(np.float32))
    np.save(clean_layer_path, dirs_clean.astype(np.float32))
    np.save(clean_out_path,   dirs_clean[n_layers - 1].astype(np.float32))

    diag_df    = pd.DataFrame(diag_rows)
    summary_df = pd.DataFrame(summary_rows)
    diag_df.to_csv(diag_path,    index=False)
    summary_df.to_csv(summary_path, index=False)

    logger.info("Saved all temporal parameters → %s", out_dir)

    print("\n=== Temporal direction diagnostics ===")
    print(diag_df.to_string(index=False))
    print(f"\nBest layer by R²: {diag_df.loc[diag_df['r2'].idxmax(), 'layer']}")

    print("\n=== Residualisation summary ===")
    cols = ["layer", "angle_from_vocab_deg", "residual_norm", "coef_norm", "intercept"]
    print(summary_df[cols].to_string(index=False))

    print(f"\nCorpus mean year ≈ {years.mean():.0f}  "
          f"(intercepts should all be close to this value).")


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    parser = argparse.ArgumentParser(
        description="Stage 01b: Fit temporal regression on Penn embeddings."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Subsample to 2000 sentences for speed.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(cfg, args.dry_run)


if __name__ == "__main__":
    main()
