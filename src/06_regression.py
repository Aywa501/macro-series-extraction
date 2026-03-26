"""Stage 06 — Panel regression of drift rate on log(RV) levels.

Main specification
------------------
Equation:
    drift_{i,t} = δ_t + β · log(RV)_{u(i),t} + ε_{i,t}

where
    i        = idiom
    t        = decade_start
    δ_t      = decade fixed effect (absorbs aggregate inflation / common shocks)
    log(RV)  = log real value of idiom i's denomination at t
    u(i)     = denomination of idiom i
    β        = expected negative: lower real value → more distributional drift

Dependent variables
-------------------
Two metrics from stage 04 are used as dependent variables:

  apd          (primary)   Average Pairwise Distance — mean cosine distance
                           across all cross-decade embedding pairs.  State-of-
                           the-art for contextualised diachronic change.
  drift_cosine (robustness) Prototype/centroid drift — cosine distance between
                           decade centroids.  Lower variance, lower power;
                           included to confirm results are not artefacts of
                           the APD estimator.

The headline result is the APD regression.  Results should agree in sign and
significance; any divergence is noted in the output.

Why levels rather than first differences
-----------------------------------------
Δlog(RV) = −Δlog(price_level) for every denomination (the nominal value is a
fixed constant that vanishes under differencing).  All denominations therefore
have an identical Δlog(RV) series — the regressor cannot distinguish penny
idioms from pound idioms.

log(RV) *levels* differ permanently across denominations by log(nominal):
    log(RV_{farthing}) = log(0.25) + constant  ← always cheapest
    log(RV_{penny})    = log(1.0)  + constant
    log(RV_{pound})    = log(240)  + constant  ← always most valuable

Decade FE absorbs the common price-level trend, leaving the cross-
denominational spread as the identifying variation.  β answers: in a given
decade, controlling for when it is, do idioms about cheaper denominations
drift more?

Why per-denomination sub-regressions are dropped
-------------------------------------------------
Within a single denomination all idioms share the same log(RV) value in a
given decade — there is no within-decade cross-sectional variation after the
decade FE is applied.  Only the pooled regression is retained.

Falsification — permutation test
---------------------------------
Within each decade, log(RV) values are randomly shuffled across idioms 200
times.  This destroys the denomination-specific assignment while preserving
the panel structure, decade distribution, and marginal distributions.  Run on
both APD and PRT regressions.

Usage
-----
    python src/06_regression.py
    python src/06_regression.py --data-dir /path/to/project --force
    python src/06_regression.py --n-permutations 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Newey-West lag order.  At decade granularity 2 lags = 20 years of serial
# correlation buffer.
NW_LAGS: int = 2
N_PERMUTATIONS: int = 200


# ---------------------------------------------------------------------------
# OLS with fixed effects
# ---------------------------------------------------------------------------

def _ols_with_fe(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    fe_col: str,
    nw_lags: int = NW_LAGS,
    label: str = "",
) -> dict:
    """OLS with one-way fixed effects and Newey-West HAC standard errors.

    Parameters
    ----------
    df:
        Long-format panel with y_col, x_col, fe_col columns.
    y_col:
        Dependent variable (drift_cosine).
    x_col:
        Key regressor (log_rv_start).
    fe_col:
        Column whose levels are absorbed as fixed effects (decade_start).
    nw_lags:
        Newey-West lag order.
    label:
        Written into the ``regression_type`` field.

    Returns
    -------
    dict
        Keys: regression_type, beta, se_nw, t_stat, p_value,
              n, n_fe_levels, r2, r2_adj.
    """
    panel = df[[y_col, x_col, fe_col]].dropna().copy()
    n_levels = panel[fe_col].nunique()

    if len(panel) < max(5, n_levels + 2):
        logger.warning(
            "Too few obs for '%s' (N=%d, FE levels=%d). Returning NaN.",
            label, len(panel), n_levels,
        )
        return {
            "regression_type": label,
            "beta": np.nan, "se_nw": np.nan,
            "t_stat": np.nan, "p_value": np.nan,
            "n": len(panel), "n_fe_levels": n_levels,
            "r2": np.nan, "r2_adj": np.nan,
        }

    fe_dummies = pd.get_dummies(panel[fe_col], prefix="fe", drop_first=True)
    X = pd.concat([panel[[x_col]], fe_dummies], axis=1).astype(float)
    X = sm.add_constant(X)
    y = panel[y_col].astype(float)

    try:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        beta    = float(res.params.get(x_col, np.nan))
        se      = float(res.bse.get(x_col, np.nan))
        t_stat  = float(res.tvalues.get(x_col, np.nan))
        p_value = float(res.pvalues.get(x_col, np.nan))
        r2      = float(res.rsquared)
        r2_adj  = float(res.rsquared_adj)
    except Exception as exc:
        logger.error("OLS failed for '%s': %s", label, exc)
        beta = se = t_stat = p_value = r2 = r2_adj = np.nan

    return {
        "regression_type": label,
        "beta": beta, "se_nw": se,
        "t_stat": t_stat, "p_value": p_value,
        "n": len(panel), "n_fe_levels": n_levels,
        "r2": r2, "r2_adj": r2_adj,
    }


# ---------------------------------------------------------------------------
# Permutation falsification
# ---------------------------------------------------------------------------

def permutation_test(
    treat: pd.DataFrame,
    y_col: str = "apd",
    n_shuffles: int = N_PERMUTATIONS,
    seed: int = 42,
    nw_lags: int = NW_LAGS,
) -> dict:
    """Within-decade permutation test of the treatment regression.

    In each of *n_shuffles* iterations, the log(RV) values are randomly
    permuted across idioms *within each decade*.  This destroys the
    denomination-specific assignment (which idiom gets which log(RV)) while
    preserving:
      - the decade distribution of observations
      - the marginal distribution of both drift and log(RV)
      - the panel structure (which decades each idiom appears in)

    The empirical p-value is the fraction of permuted |β| ≥ observed |β|.

    Parameters
    ----------
    treat:
        Treatment rows from drift_index with log_rv_start and the drift metric.
    y_col:
        Dependent variable column (``"apd"`` or ``"drift_cosine"``).
    n_shuffles:
        Number of permutation draws.
    seed:
        RNG seed for reproducibility.
    nw_lags:
        Newey-West lag order passed to _ols_with_fe.

    Returns
    -------
    dict
        Keys: y_col, beta_obs, p_permutation, n_shuffles, beta_null_mean,
              beta_null_p95 (95th percentile of |β_null|).
    """
    rng = np.random.default_rng(seed)

    obs_result = _ols_with_fe(
        treat, y_col, "log_rv_start", "decade_start",
        nw_lags=nw_lags, label="observed",
    )
    beta_obs = obs_result["beta"]

    beta_null: list[float] = []

    for _ in range(n_shuffles):
        shuffled = treat.copy()
        for decade in shuffled["decade_start"].unique():
            mask = shuffled["decade_start"] == decade
            rv_vals = shuffled.loc[mask, "log_rv_start"].values.copy()
            shuffled.loc[mask, "log_rv_start"] = rng.permutation(rv_vals)

        r = _ols_with_fe(
            shuffled, y_col, "log_rv_start", "decade_start",
            nw_lags=nw_lags, label="permuted",
        )
        if not np.isnan(r["beta"]):
            beta_null.append(r["beta"])

    arr = np.array(beta_null)
    p_perm = float(np.mean(np.abs(arr) >= abs(beta_obs))) if len(arr) > 0 else np.nan

    return {
        "y_col": y_col,
        "beta_obs": float(beta_obs),
        "p_permutation": p_perm,
        "n_shuffles": len(arr),
        "beta_null_mean": float(arr.mean()) if len(arr) > 0 else np.nan,
        "beta_null_p95": float(np.percentile(np.abs(arr), 95)) if len(arr) > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Denomination helpers
# ---------------------------------------------------------------------------

_KNOWN_DENOMS = {"penny", "pennies", "farthing", "shilling", "pound", "guinea"}


def _primary_denomination(val) -> str | None:
    if isinstance(val, (list, np.ndarray)):
        for d in val:
            if str(d).lower() in _KNOWN_DENOMS:
                return str(d).lower()
    if isinstance(val, str) and val.lower() in _KNOWN_DENOMS:
        return val.lower()
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_regression(
    data_dir: Path,
    force: bool = False,
    n_permutations: int = N_PERMUTATIONS,
) -> None:
    """Run treatment regression + permutation falsification and save results.

    Parameters
    ----------
    data_dir:
        Project data root.
    force:
        Overwrite existing outputs.
    n_permutations:
        Number of within-decade permutation draws for the falsification test.
    """
    index_path = data_dir / "processed" / "drift_index.parquet"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reg_output   = tables_dir / "regression_results.csv"
    perm_output  = tables_dir / "permutation_results.csv"

    if reg_output.exists() and not force:
        logger.info("regression_results.csv already exists. Use --force.")
        return

    if not index_path.exists():
        logger.error("drift_index.parquet not found: %s. Run stage 05 first.", index_path)
        sys.exit(1)

    df = pd.read_parquet(index_path)
    logger.info("Loaded drift index: %d rows.", len(df))
    df["primary_denom"] = df["denomination"].apply(_primary_denomination)

    all_results:  list[dict] = []
    all_perm:     list[dict] = []

    for model_key in sorted(df["model"].unique()):
        mdf = df[df["model"] == model_key].copy()
        logger.info("=== Model: %s ===", model_key)

        # ------------------------------------------------------------------ #
        # Treatment regression                                                #
        # Spec: drift ~ decade_FE + β·log_rv_start                           #
        # Identification: cross-denominational variation within decades.      #
        # Per-denomination sub-regressions are NOT run — within a single     #
        # denomination, log_rv is constant across idioms in a decade and is  #
        # therefore collinear with the decade FE.                            #
        # ------------------------------------------------------------------ #
        treat = mdf[
            (mdf["group"] == "treatment") & mdf["log_rv_start"].notna()
        ].copy()

        n_denoms_active = treat["primary_denom"].nunique()
        logger.info(
            "  Treatment: %d pairs, %d idioms, %d denominations, %d decades",
            len(treat), treat["idiom"].nunique(),
            n_denoms_active, treat["decade_start"].nunique(),
        )

        if n_denoms_active < 2:
            logger.warning(
                "  Only %d denomination(s) active — cross-denominational "
                "identification requires ≥2.  Skipping pooled regression.",
                n_denoms_active,
            )
        else:
            # ---------------------------------------------------------------- #
            # Run regression for each drift metric                             #
            #   apd          → primary result                                  #
            #   drift_cosine → robustness check (PRT)                         #
            # ---------------------------------------------------------------- #
            for y_col, role in [("apd", "primary"), ("drift_cosine", "robustness_PRT")]:
                if y_col not in treat.columns or treat[y_col].isna().all():
                    logger.warning("  Column '%s' missing or all-NaN — skipping.", y_col)
                    continue

                label = f"treatment_decade_FE_{role}"
                res = _ols_with_fe(
                    treat,
                    y_col=y_col, x_col="log_rv_start",
                    fe_col="decade_start",
                    label=label,
                )
                res["model"] = model_key
                res["metric"] = y_col
                all_results.append(res)
                logger.info(
                    "  [%s/%s]  β=%.5f  SE=%.5f  t=%.2f  p=%.4f  N=%d  decades=%d",
                    y_col, role,
                    res["beta"], res["se_nw"], res["t_stat"], res["p_value"],
                    res["n"], res["n_fe_levels"],
                )

            # -------------------------------------------------------------- #
            # Permutation falsification — run on APD (primary)               #
            # Also run on PRT if the column is available.                     #
            # -------------------------------------------------------------- #
            logger.info("  Running permutation test (%d shuffles) …", n_permutations)
            for y_col in ["apd", "drift_cosine"]:
                if y_col not in treat.columns or treat[y_col].isna().all():
                    continue
                perm = permutation_test(
                    treat, y_col=y_col, n_shuffles=n_permutations,
                    # Different seeds so the two tests are independent
                    seed=42 if y_col == "apd" else 43,
                )
                perm["model"] = model_key
                perm["regression_type"] = f"permutation_{y_col}"
                all_perm.append(perm)
                logger.info(
                    "  [permutation/%s]  β_obs=%.5f  p_perm=%.4f  "
                    "β_null_mean=%.5f  |β_null|_p95=%.5f  N_shuffles=%d",
                    y_col,
                    perm["beta_obs"], perm["p_permutation"],
                    perm["beta_null_mean"], perm["beta_null_p95"],
                    perm["n_shuffles"],
                )

    # ---- Save tables -------------------------------------------------------
    if all_results:
        reg_df = pd.DataFrame(all_results)
        col_order = [
            "model", "metric", "regression_type",
            "beta", "se_nw", "t_stat", "p_value",
            "n", "n_fe_levels", "r2", "r2_adj",
        ]
        reg_df = reg_df[[c for c in col_order if c in reg_df.columns]]
        reg_df.to_csv(reg_output, index=False, float_format="%.6f")
        logger.info("Saved regression results → %s", reg_output)
        logger.info("\n%s", reg_df.to_string(index=False))

    if all_perm:
        perm_df = pd.DataFrame(all_perm)
        col_order = [
            "model", "regression_type", "y_col",
            "beta_obs", "p_permutation",
            "beta_null_mean", "beta_null_p95", "n_shuffles",
        ]
        perm_df = perm_df[[c for c in col_order if c in perm_df.columns]]
        perm_df.to_csv(perm_output, index=False, float_format="%.6f")
        logger.info("Saved permutation results → %s", perm_output)
        logger.info("\n%s", perm_df.to_string(index=False))

    # ---- Plots -------------------------------------------------------------
    if all_results:
        _plot_coefficients(pd.DataFrame(all_results), figures_dir)
    if all_perm:
        _plot_permutation(pd.DataFrame(all_perm), figures_dir)


def _plot_coefficients(reg_df: pd.DataFrame, figures_dir: Path) -> None:
    """Forest plot of β estimates with 95% CI bars, separated by metric."""
    plot_df = reg_df.dropna(subset=["beta", "se_nw"]).copy()
    if plot_df.empty:
        return

    plot_df["ci95"] = 1.96 * plot_df["se_nw"]
    metric_col = "metric" if "metric" in plot_df.columns else "regression_type"
    plot_df["label"] = plot_df["model"] + " / " + plot_df[metric_col]

    # Colour by model; solid = APD, hatched = PRT robustness
    model_colors = {
        "bge": "#2166ac", "macberth": "#d6604d", "bert": "#1a9850",
    }
    colors = [
        model_colors.get(m, "grey") for m in plot_df["model"]
    ]
    # Hatch pattern for robustness rows
    hatches = [
        "///" if "drift_cosine" in str(row.get(metric_col, "")) else ""
        for _, row in plot_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, max(3.0, len(plot_df) * 0.75 + 1)))
    y_pos = list(range(len(plot_df)))

    for i, (yp, beta, ci, color, hatch) in enumerate(
        zip(y_pos, plot_df["beta"], plot_df["ci95"], colors, hatches)
    ):
        ax.barh(yp, beta, xerr=ci, color=color, alpha=0.75, height=0.5,
                hatch=hatch,
                error_kw={"elinewidth": 1.4, "capsize": 5})

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.set_xlabel("β  (drift ~ log(RV) + decade FE)", fontsize=9)
    ax.set_title(
        "Treatment regression: β ± 95% CI (Newey-West HAC)\n"
        "Solid = APD (primary)  ///  = PRT robustness check\n"
        "β < 0: lower real value → more distributional drift",
        fontsize=9,
    )
    fig.tight_layout()
    out = figures_dir / "regression_coefficients.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved coefficient plot → %s", out)


def _plot_permutation(perm_df: pd.DataFrame, figures_dir: Path) -> None:
    """Visualise permutation null distribution vs observed β."""
    # Re-run the permutation draws from the saved summary isn't possible here
    # (we only saved summary stats, not the full null distribution).
    # Plot a simple comparison bar chart instead.
    if perm_df.empty:
        return

    fig, axes = plt.subplots(1, len(perm_df), figsize=(5 * len(perm_df), 4),
                              squeeze=False)
    for ax, (_, row) in zip(axes[0], perm_df.iterrows()):
        bars = ["Observed β", "Null mean", "|Null| p95"]
        vals = [row["beta_obs"], row["beta_null_mean"], -row["beta_null_p95"]]
        colors = ["#2166ac" if "bge" in row["model"] else "#d6604d",
                  "grey", "lightcoral"]
        ax.bar(bars, vals, color=colors, alpha=0.8, width=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"{row['model']}\np_perm={row['p_permutation']:.3f}  "
            f"(N={row['n_shuffles']} shuffles)",
            fontsize=9,
        )
        ax.set_ylabel("β")

    fig.suptitle(
        "Permutation falsification: observed β vs within-decade shuffled null",
        fontsize=10,
    )
    fig.tight_layout()
    out = figures_dir / "permutation_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved permutation plot → %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 06: Panel regression (drift ~ log(RV) + decade FE) "
            "with permutation falsification."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--n-permutations", type=int, default=N_PERMUTATIONS,
        help=f"Within-decade permutation draws (default: {N_PERMUTATIONS}).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_regression(
        data_dir=args.data_dir,
        force=args.force,
        n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
