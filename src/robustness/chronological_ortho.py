#!/usr/bin/env python3
"""Chronological Vector Orthogonalisation Robustness Check.

Tests whether the score_paraphrase signal is confounded by temporal language
change encoded in the MacBERTh embedding space.

Steps
-----
1. Load MacBERTh embeddings + metadata.
2. Estimate the chronological direction via ridge regression (year ~ embeddings).
3. Orthogonalise the triviality axis against the chronological direction.
4. Re-score all embeddings using the cleaned axis.
5. Compare regressions: original vs orthogonalised scores (levels + first diffs).
6. Write summary diagnostics.

Flags
-----
- Chronological probe R² > 0.4 → flagged before proceeding.
- Angle between axes > 45°     → flagged, ask whether to proceed.
- First-diff β collapses to 0  → flagged, potential confound identified.

Usage
-----
    python src/robustness/chronological_ortho.py
    python src/robustness/chronological_ortho.py --data-dir data/gutenberg \
        --results-dir data/gutenberg/robustness/results
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOMINAL_VALUES: dict[str, float] = {
    "penny": 1.0, "pennies": 1.0, "farthing": 0.25,
    "shilling": 12.0, "pound": 240.0, "guinea": 252.0,
}
NORMALISE_YEAR = 1800

FARTHINGS     = {"uttermost farthing", "brass farthing", "not worth a farthing"}
WELL_COVERED  = {
    "a pretty penny", "uttermost farthing", "brass farthing",
    "penny for your thoughts", "penny dreadful",
    "turn an honest penny", "not worth a farthing",
}

R2_FLAG_THRESHOLD    = 0.4
ANGLE_FLAG_THRESHOLD = 45.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_denom(d) -> str | None:
    try:
        for x in list(d):
            if isinstance(x, str) and x.lower() in NOMINAL_VALUES:
                return x.lower()
    except Exception:
        pass
    return None


def _compute_log_rv(denom: str | None, pi_norm: float) -> float:
    if denom is None or pd.isna(pi_norm):
        return float("nan")
    nominal = NOMINAL_VALUES.get(denom, float("nan"))
    if np.isnan(nominal) or pi_norm <= 0:
        return float("nan")
    return float(np.log(nominal / (pi_norm / 100.0)))


def _ols_hc3(df: pd.DataFrame, y_col: str, x_col: str, label: str = "") -> dict:
    """OLS with HC3 robust SEs at observation level."""
    sub = df[[y_col, x_col]].dropna()
    if len(sub) < 10:
        logger.warning("[%s] Too few rows (%d) — skipped.", label, len(sub))
        return {}
    X = sm.add_constant(sub[[x_col]].astype(float))
    y = sub[y_col].astype(float)
    res = sm.OLS(y, X).fit(cov_type="HC3")
    b  = float(res.params[x_col])
    se = float(res.bse[x_col])
    t  = float(res.tvalues[x_col])
    p  = float(res.pvalues[x_col])
    logger.info("  [%s] β=%.5f  SE=%.5f  t=%.3f  p=%.4f  n=%d",
                label, b, se, t, p, int(res.nobs))
    return {"specification": label, "beta": round(b, 6), "se": round(se, 6),
            "t_stat": round(t, 4), "p_value": round(p, 6), "n_obs": int(res.nobs)}


def _first_diff_ols(df: pd.DataFrame, y_col: str, x_col: str, label: str = "") -> dict:
    """First-difference within idiom by decade, then OLS with HC3."""
    df2 = df.copy()
    df2["decade"] = (df2["year"] // 10) * 10
    agg = (
        df2.groupby(["idiom", "decade"])[[y_col, x_col]]
        .mean()
        .reset_index()
        .sort_values(["idiom", "decade"])
    )
    agg[f"d_{y_col}"] = agg.groupby("idiom")[y_col].diff()
    agg[f"d_{x_col}"] = agg.groupby("idiom")[x_col].diff()
    sub = agg.dropna(subset=[f"d_{y_col}", f"d_{x_col}"])
    if len(sub) < 5:
        logger.warning("[%s] Too few first-diff rows (%d) — skipped.", label, len(sub))
        return {}
    X = sm.add_constant(sub[[f"d_{x_col}"]].astype(float))
    y = sub[f"d_{y_col}"].astype(float)
    res = sm.OLS(y, X).fit(cov_type="HC3")
    b  = float(res.params[f"d_{x_col}"])
    se = float(res.bse[f"d_{x_col}"])
    t  = float(res.tvalues[f"d_{x_col}"])
    p  = float(res.pvalues[f"d_{x_col}"])
    logger.info("  [%s | first-diff] β=%.5f  SE=%.5f  t=%.3f  p=%.4f  n=%d",
                label, b, se, t, p, int(res.nobs))
    return {"specification": f"{label}_first_diff", "beta": round(b, 6),
            "se": round(se, 6), "t_stat": round(t, 4), "p_value": round(p, 6),
            "n_obs": int(res.nobs)}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    flags: list[str] = []

    # ── Step 1: Load embeddings + metadata ───────────────────────────────────
    logger.info("=== Step 1: Loading embeddings and metadata ===")
    emb_dir = data_dir / "processed" / "embeddings"

    E   = np.load(emb_dir / "macberth_L9.npy").astype(np.float64)  # (N, 768)
    idx = pd.read_parquet(emb_dir / "index.parquet")                # (N,) id

    obs   = pd.read_parquet(data_dir / "interim" / "observations.parquet")
    obs   = obs[obs["is_idiomatic"] == True].copy()
    sc    = pd.read_parquet(data_dir / "processed" / "scores.parquet")
    sc_mb = sc[sc["model"] == "macberth"][["id", "score_paraphrase"]].copy()

    meta = (
        idx
        .merge(obs[["id", "idiom", "group", "year", "denomination"]], on="id", how="left")
        .merge(sc_mb, on="id", how="left")
    )
    meta["primary_denom"] = meta["denomination"].apply(_primary_denom)

    assert len(meta) == len(E), f"Length mismatch: meta={len(meta)}, E={len(E)}"
    logger.info("Embeddings: shape=%s | year=%d–%d | idioms=%d",
                E.shape, int(meta.year.min()), int(meta.year.max()),
                meta.idiom.nunique())

    # ── Step 2: Chronological direction ──────────────────────────────────────
    logger.info("=== Step 2: Estimating chronological direction ===")

    valid = meta["year"].notna().values
    E_yr  = E[valid]
    y_yr  = meta.loc[valid, "year"].values.astype(np.float64)

    # Standardise target for ridge stability
    y_mean, y_std = y_yr.mean(), y_yr.std()
    y_sc = (y_yr - y_mean) / y_std

    alphas = np.logspace(-3, 6, 30)
    rcv = RidgeCV(alphas=alphas, cv=5)
    rcv.fit(E_yr, y_sc)
    selected_alpha = float(rcv.alpha_)
    logger.info("RidgeCV selected α=%.4f", selected_alpha)

    ridge = Ridge(alpha=selected_alpha)
    ridge.fit(E_yr, y_sc)
    r2_probe = float(ridge.score(E_yr, y_sc))
    logger.info("Chronological probe R²=%.4f", r2_probe)

    if r2_probe > R2_FLAG_THRESHOLD:
        msg = (f"FLAG: Chronological probe R²={r2_probe:.4f} exceeds {R2_FLAG_THRESHOLD}. "
               f"Embedding space is strongly temporally structured — orthogonalisation "
               f"may remove substantive signal alongside temporal confound. Proceeding.")
        logger.warning(msg)
        flags.append(msg)

    chron_dir = ridge.coef_.copy()
    chron_dir /= np.linalg.norm(chron_dir)
    np.save(results_dir / "chronological_direction.npy", chron_dir)
    logger.info("Saved chronological direction → %s", results_dir / "chronological_direction.npy")

    # Decade-level monotonicity check
    meta_v = meta[valid].copy()
    meta_v["chron_proj"] = E_yr @ chron_dir
    meta_v["decade"] = (meta_v["year"] // 10) * 10
    decade_proj = meta_v.groupby("decade")["chron_proj"].mean().sort_index()
    diffs = np.diff(decade_proj.values)
    n_inc = int((diffs > 0).sum())
    n_tot = len(diffs)
    pct   = n_inc / n_tot
    mono_msg = (f"Decade projections onto chronological axis: "
                f"{n_inc}/{n_tot} consecutive pairs increasing ({pct:.0%} monotone).")
    logger.info(mono_msg)
    if pct < 0.70:
        msg = (f"FLAG: Monotonicity only {pct:.0%} — temporal signal is not "
               f"linearly structured. Orthogonalisation may not be well-founded.")
        logger.warning(msg)
        flags.append(msg)

    # ── Step 3: Triviality axis + orthogonalisation ──────────────────────────
    logger.info("=== Step 3: Estimating and orthogonalising triviality axis ===")

    has_score = meta["score_paraphrase"].notna().values
    E_sc = E[has_score]
    s_sc = meta.loc[has_score, "score_paraphrase"].values.astype(np.float64)

    # Estimate triviality axis: direction in embedding space explaining scores
    # Use ridge with mild regularisation (α=1) to avoid overfitting 768-dim space.
    ridge_triv = Ridge(alpha=1.0)
    ridge_triv.fit(E_sc, s_sc)
    triv_axis = ridge_triv.coef_.copy()
    triv_axis /= np.linalg.norm(triv_axis)
    np.save(results_dir / "triviality_axis_original.npy", triv_axis)

    # Orthogonalise: remove the component along the chronological direction
    proj_onto_chron = float(np.dot(triv_axis, chron_dir))
    triv_axis_clean = triv_axis - proj_onto_chron * chron_dir
    norm_clean = np.linalg.norm(triv_axis_clean)
    if norm_clean < 1e-10:
        logger.error("Cleaned triviality axis has near-zero norm — axes are collinear.")
        sys.exit(1)
    triv_axis_clean /= norm_clean
    np.save(results_dir / "triviality_axis_cleaned.npy", triv_axis_clean)
    logger.info("Saved cleaned triviality axis → %s", results_dir / "triviality_axis_cleaned.npy")

    dot_val   = float(np.clip(np.dot(triv_axis, triv_axis_clean), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot_val)))
    logger.info("Angle between original and cleaned axes: %.2f°", angle_deg)
    logger.info("Temporal component removed from triviality axis: %.4f (dot with chron_dir)", proj_onto_chron)

    if angle_deg > ANGLE_FLAG_THRESHOLD:
        msg = (f"FLAG: Angle between original and cleaned axes = {angle_deg:.2f}° > "
               f"{ANGLE_FLAG_THRESHOLD}°. The cleaned scores measure something "
               f"substantially different from the original triviality construct. "
               f"Interpret with caution.")
        logger.warning(msg)
        flags.append(msg)

    # ── Step 4: Re-score all embeddings ──────────────────────────────────────
    logger.info("=== Step 4: Re-scoring embeddings ===")

    scores_orig  = (E @ triv_axis).astype(np.float64)
    scores_clean = (E @ triv_axis_clean).astype(np.float64)

    meta["score_orig"]  = scores_orig
    meta["score_clean"] = scores_clean

    # Load price index and compute log_rv
    raw_dir = PROJECT_ROOT / "data" / "raw"
    price = pd.read_csv(raw_dir / "price_index.csv")
    base  = float(price.loc[price["year"] == NORMALISE_YEAR, "price_index"].iloc[0])
    price["pi_norm"] = price["price_index"] / base * 100.0

    panel = meta.merge(price[["year", "pi_norm"]], on="year", how="left")
    panel["log_rv"] = [
        _compute_log_rv(d, pi)
        for d, pi in zip(panel["primary_denom"], panel["pi_norm"])
    ]

    # Save panel CSVs (treatment only, has log_rv)
    treat = panel[panel["group"] == "treatment"].copy()
    treat.to_csv(results_dir / "panel.csv", index=False)

    treat_clean = treat.copy()
    treat_clean["score_paraphrase"] = treat_clean["score_clean"]
    treat_clean.to_csv(results_dir / "panel_orthogonalised.csv", index=False)

    logger.info("Saved panel.csv (%d rows) and panel_orthogonalised.csv → %s",
                len(treat), results_dir)

    # ── Step 5: Regression comparison ────────────────────────────────────────
    logger.info("=== Step 5: Regression comparison ===")

    specs = [
        ("farthing_only", FARTHINGS),
        ("well_covered",  WELL_COVERED),
    ]

    rows: list[dict] = []

    for spec_name, idiom_set in specs:
        tr  = treat[treat["idiom"].isin(idiom_set)].dropna(subset=["log_rv"])
        trc = treat_clean[treat_clean["idiom"].isin(idiom_set)].dropna(subset=["log_rv"])

        logger.info("--- %s: n=%d ---", spec_name, len(tr))

        # Levels
        r = _ols_hc3(tr,  "score_orig",  "log_rv", f"{spec_name}|original|levels")
        if r: rows.append(r)
        r = _ols_hc3(trc, "score_clean", "log_rv", f"{spec_name}|orthogonalised|levels")
        if r: rows.append(r)

        # First differences
        r = _first_diff_ols(tr,  "score_orig",  "log_rv", f"{spec_name}|original")
        if r: rows.append(r)
        r = _first_diff_ols(trc, "score_clean", "log_rv", f"{spec_name}|orthogonalised")
        if r: rows.append(r)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_dir / "chronological_robustness.csv", index=False)
    logger.info("Saved → %s", results_dir / "chronological_robustness.csv")

    # Check first-diff collapse
    for _, row in results_df.iterrows():
        if "first_diff" in row["specification"] and "orthogonalised" in row["specification"]:
            orig_label = row["specification"].replace("orthogonalised", "original")
            orig_row = results_df[results_df["specification"] == orig_label]
            if not orig_row.empty:
                orig_b = float(orig_row["beta"].iloc[0])
                if orig_b != 0 and abs(row["beta"]) < abs(orig_b) * 0.1:
                    msg = (f"FLAG: First-difference β collapsed after orthogonalisation "
                           f"in {row['specification']} ({orig_b:.5f} → {row['beta']:.5f}). "
                           f"Temporal drift may be a real confound in the original scores.")
                    logger.warning(msg)
                    flags.append(msg)

    # ── Step 6: Summary diagnostic ───────────────────────────────────────────
    logger.info("=== Step 6: Writing summary ===")

    # Interpretation
    farthing_orig = results_df[
        (results_df["specification"].str.contains("farthing_only")) &
        (results_df["specification"].str.contains("original")) &
        (~results_df["specification"].str.contains("first_diff"))
    ]
    farthing_clean = results_df[
        (results_df["specification"].str.contains("farthing_only")) &
        (results_df["specification"].str.contains("orthogonalised")) &
        (~results_df["specification"].str.contains("first_diff"))
    ]

    if not farthing_orig.empty and not farthing_clean.empty:
        b_orig  = float(farthing_orig["beta"].iloc[0])
        p_orig  = float(farthing_orig["p_value"].iloc[0])
        b_clean = float(farthing_clean["beta"].iloc[0])
        p_clean = float(farthing_clean["p_value"].iloc[0])

        if abs(b_clean) > abs(b_orig) * 0.5 and p_clean < 0.1:
            interp = ("The result survives orthogonalisation. The headline farthing-idiom "
                      f"coefficient is largely unchanged (original β={b_orig:.5f} p={p_orig:.4f}; "
                      f"cleaned β={b_clean:.5f} p={p_clean:.4f}), indicating the triviality "
                      "signal is not primarily driven by temporal language change encoded in "
                      "the MacBERTh embedding space.")
        elif abs(b_clean) > abs(b_orig) * 0.2:
            interp = ("The result partially survives orthogonalisation. "
                      f"Original β={b_orig:.5f} (p={p_orig:.4f}); "
                      f"cleaned β={b_clean:.5f} (p={p_clean:.4f}). "
                      "Some of the original signal is attributable to temporal drift in the "
                      "embedding space, but a meaningful component remains after cleaning. "
                      "Interpret the main result with appropriate caution.")
        else:
            interp = ("The result largely collapses after orthogonalisation. "
                      f"Original β={b_orig:.5f} (p={p_orig:.4f}); "
                      f"cleaned β={b_clean:.5f} (p={p_clean:.4f}). "
                      "This suggests the original score_paraphrase signal is substantially "
                      "confounded by temporal language change in the embedding space. "
                      "The thesis requires re-evaluation with a temporal-drift-robust measure.")
    else:
        interp = "Insufficient data for farthing-only interpretation."

    table_str = results_df.to_string(index=False)

    summary = textwrap.dedent(f"""\
        ============================================================
        CHRONOLOGICAL ORTHOGONALISATION ROBUSTNESS CHECK
        Model: MacBERTh (macberth_L9.npy)
        ============================================================

        1. CHRONOLOGICAL PROBE
           R² (ridge, CV-alpha={selected_alpha:.4f}): {r2_probe:.4f}
           {"*** FLAGGED: R² > " + str(R2_FLAG_THRESHOLD) if r2_probe > R2_FLAG_THRESHOLD else "OK: R² within acceptable range"}

        2. DECADE MONOTONICITY
           {mono_msg}

        3. AXIS ANGLE
           Angle between original and cleaned triviality axes: {angle_deg:.2f}°
           Temporal component of original axis (dot with chron_dir): {proj_onto_chron:.4f}
           {"*** FLAGGED: angle > " + str(ANGLE_FLAG_THRESHOLD) + "°" if angle_deg > ANGLE_FLAG_THRESHOLD else "OK: angle within expected range"}

        4. BETA COMPARISON TABLE
        {table_str}

        5. FLAGS RAISED
        {"  None" if not flags else chr(10).join("  - " + f for f in flags)}

        6. INTERPRETATION
        {textwrap.fill(interp, width=72, subsequent_indent="  ")}
    """)

    summary_path = results_dir / "chronological_robustness_summary.txt"
    summary_path.write_text(summary)
    logger.info("Saved summary → %s", summary_path)
    print("\n" + summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "gutenberg",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "gutenberg" / "robustness" / "results",
    )
    args = parser.parse_args()
    run(args.data_dir, args.results_dir)


if __name__ == "__main__":
    main()
