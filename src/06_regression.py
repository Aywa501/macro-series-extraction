"""Stage 06 — Regression of idiom significance score on log(RV).

Thesis
------
As the real value of a currency denomination falls, idioms referencing that
denomination are used in less weighty / significant contexts.

Example: when the penny had high purchasing power, invoking "a pretty penny"
carried genuine weight — the sum was real.  As the penny's real value
collapsed the significance claim weakened.

Measurement
-----------
score_paraphrase (stage 03) = sim(context, trivial_pole) − sim(context, significant_pole)

    score > 0  →  idiom invoked in a lightweight / trivial sense
    score < 0  →  idiom invoked in a weighty / significant sense

Poles are idiom-specific but consistently oriented: the "significant_pole"
always describes the idiom's claim as "considerable / substantial / binding"
and the "trivial_pole" as "negligible."

Primary prediction:   β < 0
    score_paraphrase_it = α + β · log_rv(denom(i), t) + ε_it

Higher real value of the denomination → lower score → idiom carries more weight.

No fixed effects are used in the primary specification.  Both sources of
identifying variation are valid:

  1. Cross-idiom (cross-denomination): farthing idioms (low nominal) have lower
     log_rv than shilling idioms (high nominal) in every decade.  If cheaper
     denominations anchor weaker claims, farthing idioms should score higher.

  2. Within-idiom over time: as prices rise, log_rv falls for all idioms.
     Do usages become more trivial as the denomination loses real value?

Adding idiom or decade fixed effects would absorb exactly this variation and
leave nothing to identify β.  Standard errors are clustered by idiom to
account for within-idiom serial correlation.

Note on 'shilling a dozen'
--------------------------
This idiom uses a high-denomination coin to express cheapness / abundance
(structural analogue of "ten a penny").  The significant_pole is "rare and of
considerable worth," which is the OPPOSITE of its surface meaning.  As a
result its per-idiom β is expected to be positive — a known anomaly noted in
the results.  Results are reported both including and excluding it.

Falsification
-------------
Denomination assignments are permuted 200 times: each idiom is randomly
reassigned another treatment idiom's log_rv series (shuffling which
denomination's price history it gets).  This destroys the denomination–idiom
mapping while preserving within-idiom temporal structure.

Usage
-----
    python src/06_regression.py [--data-dir PATH] [--force] [--n-permutations N]
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
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

N_PERMUTATIONS: int = 200

NOMINAL_VALUES: dict[str, float] = {
    "penny":    1.0,
    "pennies":  1.0,
    "farthing": 0.25,
    "shilling": 12.0,
    "pound":    240.0,
    "guinea":   252.0,
}
NORMALISE_YEAR: int = 1800

# This idiom uses denomination for cheapness — β expected positive (anomaly)
ANOMALOUS_IDIOMS = {"shilling a dozen"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _primary_denom(denom_val) -> str | None:
    try:
        items = list(denom_val)
    except TypeError:
        return None
    for d in items:
        if isinstance(d, str) and d.lower() in NOMINAL_VALUES:
            return d.lower()
    return None


def build_panel(data_dir: Path) -> pd.DataFrame:
    """Observation-level panel: score_paraphrase + log_rv + metadata."""
    obs_path   = data_dir / "interim" / "observations.parquet"
    score_path = data_dir / "processed" / "scores.parquet"
    price_path = data_dir.parent / "raw" / "price_index.csv"
    if not price_path.exists():
        price_path = data_dir / "raw" / "price_index.csv"

    obs    = pd.read_parquet(obs_path)
    scores = pd.read_parquet(score_path)
    price  = pd.read_csv(price_path)

    base = float(price.loc[price["year"] == NORMALISE_YEAR, "price_index"].iloc[0])
    price["pi_norm"] = price["price_index"] / base * 100.0

    obs = obs[obs["is_idiomatic"] == True].copy()
    panel = obs.merge(scores, on="id", how="inner")
    panel["decade"] = (panel["year"] // 10) * 10
    panel = panel.merge(price[["year", "pi_norm"]], on="year", how="left")
    panel["primary_denom"] = panel["denomination"].apply(_primary_denom)
    panel["log_rv"] = panel.apply(
        lambda r: (
            np.log(NOMINAL_VALUES[r["primary_denom"]] / (r["pi_norm"] / 100.0))
            if pd.notna(r["primary_denom"]) and r["primary_denom"] in NOMINAL_VALUES
            and pd.notna(r["pi_norm"]) else np.nan
        ),
        axis=1,
    )
    logger.info(
        "Panel: %d rows | %d models | log_rv NaN %.1f%%",
        len(panel), panel["model"].nunique(), panel["log_rv"].isna().mean() * 100,
    )
    return panel


# ---------------------------------------------------------------------------
# OLS — observation-level with HC3 robust SEs
# ---------------------------------------------------------------------------

def _ols_hc3(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    label: str = "",
    intercept: bool = True,
    idiom_fe: bool = False,
) -> dict:
    """OLS at observation level with HC3 heteroskedasticity-robust SEs.

    Clustering by idiom collapses effective N to the number of idioms (too
    few for inference).  HC3 preserves observation-level power while still
    correcting for heteroskedasticity.  Within-idiom serial correlation is
    a mild concern noted in results but does not justify discarding variation.

    If idiom_fe=True, idiom dummies are included to control for baseline
    score differences across idioms; identification then comes purely from
    within-idiom time variation in log_rv.
    """
    needed = [y_col, x_col, "idiom"] if idiom_fe else [y_col, x_col]
    sub = df[needed + (["idiom"] if "idiom" not in needed else [])].dropna(subset=[y_col, x_col])
    if len(sub) < 10:
        logger.warning("Too few rows (%d) for '%s' — skipped.", len(sub), label)
        return {}

    X = sub[[x_col]].astype(float)
    if idiom_fe:
        dummies = pd.get_dummies(sub["idiom"], drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    if intercept:
        X = sm.add_constant(X)

    y = sub[y_col].astype(float)
    res = sm.OLS(y, X).fit(cov_type="HC3")

    beta = float(res.params[x_col])
    se   = float(res.bse[x_col])
    t    = float(res.tvalues[x_col])
    p    = float(res.pvalues[x_col])
    r2   = float(res.rsquared)
    n    = int(res.nobs)
    n_id = int(sub["idiom"].nunique())

    logger.info(
        "  [%s] β=%.5f  SE=%.5f  t=%.3f  p=%.4f  n=%d  n_idioms=%d  R²=%.4f",
        label, beta, se, t, p, n, n_id, r2,
    )
    return {
        "regression_type": label,
        "beta": beta, "se_hc3": se, "t_stat": t,
        "p_value": p, "n": n, "n_idioms": n_id, "r2": r2,
    }


# ---------------------------------------------------------------------------
# Per-idiom OLS
# ---------------------------------------------------------------------------

def _per_idiom_ols(panel: pd.DataFrame, model: str) -> pd.DataFrame:
    """OLS score ~ log_rv for each treatment idiom independently."""
    mp = panel[(panel["model"] == model) & (panel["group"] == "treatment")]
    rows = []
    for idiom, g in mp.groupby("idiom"):
        sub = g.dropna(subset=["score_paraphrase", "log_rv"])
        if len(sub) < 5:
            continue
        X = sm.add_constant(sub[["log_rv"]].astype(float))
        y = sub["score_paraphrase"].astype(float)
        res = sm.OLS(y, X).fit()
        beta = float(res.params["log_rv"])
        p    = float(res.pvalues["log_rv"])
        r, pr = pearsonr(sub["log_rv"], sub["score_paraphrase"])
        rows.append({
            "model": model,
            "idiom": idiom,
            "primary_denom": sub["primary_denom"].iloc[0],
            "anomalous": idiom in ANOMALOUS_IDIOMS,
            "beta": round(beta, 5),
            "p_value": round(p, 4),
            "pearson_r": round(r, 4),
            "pearson_p": round(pr, 4),
            "n": len(sub),
        })
    return pd.DataFrame(rows).sort_values("beta")


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _permutation_test(
    panel: pd.DataFrame,
    model: str,
    beta_obs: float,
    label: str,
    subset_idioms: set | None = None,
    n_shuffles: int = N_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Shuffle log_rv series across idioms; rerun OLS at observation level."""
    if rng is None:
        rng = np.random.default_rng(42)

    mp = panel[(panel["model"] == model) & (panel["group"] == "treatment")].copy()
    if subset_idioms:
        mp = mp[mp["idiom"].isin(subset_idioms)]
    mp = mp.dropna(subset=["score_paraphrase", "log_rv"])

    # Map idiom → log_rv series (using primary_denom as the key)
    idioms = mp["idiom"].unique()
    # For each idiom, get its full log_rv time series keyed by year
    idiom_logrv = {
        idiom: grp.set_index("year")["log_rv"].to_dict()
        for idiom, grp in mp.groupby("idiom")
    }

    null_betas: list[float] = []
    for _ in range(n_shuffles):
        perm_idioms = rng.permutation(idioms)
        mapping = dict(zip(idioms, perm_idioms))  # idiom → donor idiom
        shuffled = mp.copy()
        shuffled["log_rv"] = shuffled.apply(
            lambda r: idiom_logrv[mapping[r["idiom"]]].get(r["year"], np.nan),
            axis=1,
        )
        row = _ols_hc3(
            shuffled, "score_paraphrase", "log_rv", label="perm"
        )
        if row:
            null_betas.append(row["beta"])

    null = np.array(null_betas)
    p_perm = float(np.mean(null <= beta_obs))  # one-sided H1: β < 0
    return {
        "model": model,
        "regression_type": label,
        "beta_obs": round(beta_obs, 6),
        "p_permutation": round(p_perm, 4),
        "beta_null_mean": round(float(null.mean()), 6),
        "beta_null_p05": round(float(np.percentile(null, 5)), 6),
        "n_shuffles": len(null_betas),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_regression(
    data_dir: Path,
    n_permutations: int = N_PERMUTATIONS,
    force: bool = False,
) -> None:
    out_dir = data_dir / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = data_dir / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    reg_path   = out_dir / "regression_results.csv"
    perm_path  = out_dir / "permutation_results.csv"
    idiom_path = out_dir / "per_idiom_results.csv"

    if not force and reg_path.exists():
        logger.info("regression_results.csv already exists. Use --force to rerun.")
        return

    panel = build_panel(data_dir)
    models = sorted(panel["model"].unique())

    reg_rows:  list[dict] = []
    perm_rows: list[dict] = []
    idiom_rows: list[pd.DataFrame] = []

    WELL_COVERED = {
        "a pretty penny", "uttermost farthing", "brass farthing",
        "penny for your thoughts", "penny dreadful", "turn an honest penny",
        "not worth a farthing",
    }
    FARTHINGS = {"uttermost farthing", "brass farthing", "not worth a farthing"}

    for model in models:
        logger.info("=== Model: %s ===", model)
        mp = panel[(panel["model"] == model) & (panel["group"] == "treatment")]
        mp = mp.dropna(subset=["score_paraphrase", "log_rv"])

        specs = [
            ("all_treatment",        mp,                                          False),
            ("well_covered",         mp[mp["idiom"].isin(WELL_COVERED)],          False),
            ("farthing_only",        mp[mp["idiom"].isin(FARTHINGS)],             False),
            ("farthing_idiom_fe",    mp[mp["idiom"].isin(FARTHINGS)],             True),
            ("well_covered_idiom_fe",mp[mp["idiom"].isin(WELL_COVERED)],          True),
        ]

        for label, sub, fe in specs:
            row = _ols_hc3(sub, "score_paraphrase", "log_rv",
                           label=label, idiom_fe=fe)
            if row:
                row["model"] = model
                row["note"] = ("idiom FE" if fe else "no FE") + f" | {label}"
                reg_rows.append(row)

        # --- Permutation test on the headline farthing spec ---
        frow = next((r for r in reg_rows
                     if r["model"] == model and r["regression_type"] == "farthing_only"), None)
        if frow:
            perm_rows.append(_permutation_test(
                panel, model, frow["beta"],
                subset_idioms=FARTHINGS,
                label="farthing_only",
                n_shuffles=n_permutations,
            ))

        # --- Placebo falsification: assign penny log_rv to placebo idioms ---
        penny_logrv = (
            panel[(panel["model"] == model) & (panel["primary_denom"] == "penny")]
            .set_index("year")["log_rv"].drop_duplicates().to_dict()
        )
        plac = panel[(panel["model"] == model) & (panel["group"] == "placebo")].copy()
        plac["log_rv"] = plac["year"].map(penny_logrv)
        row_plac = _ols_hc3(plac, "score_paraphrase", "log_rv",
                            label="placebo_penny_logrv")
        if row_plac:
            row_plac["model"] = model
            row_plac["note"] = "placebo idioms, penny log_rv assigned"
            reg_rows.append(row_plac)

        # --- Per-idiom OLS ---
        idiom_df = _per_idiom_ols(panel, model)
        idiom_rows.append(idiom_df)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    reg_df   = pd.DataFrame(reg_rows)
    perm_df  = pd.DataFrame(perm_rows)
    idiom_df = pd.concat(idiom_rows, ignore_index=True)

    reg_df.to_csv(reg_path,   index=False)
    perm_df.to_csv(perm_path, index=False)
    idiom_df.to_csv(idiom_path, index=False)

    logger.info("Saved → %s", reg_path)
    logger.info("Saved → %s", perm_path)
    logger.info("Saved → %s", idiom_path)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 80)
    logger.info("\n%s", reg_df[["model","regression_type","beta","se_cluster","t_stat","p_value","n","r2","note"]].to_string(index=False))
    logger.info("\n=== Per-idiom β (macberth) ===\n%s",
        idiom_df[idiom_df["model"]=="macberth"].to_string(index=False))
    logger.info("\n%s", perm_df.to_string(index=False))

    # ----------------------------------------------------------------
    # Figure: scatter + OLS line per model
    # ----------------------------------------------------------------
    for model in models:
        mp = panel[(panel["model"] == model) & (panel["group"] == "treatment")]
        mp = mp.dropna(subset=["score_paraphrase", "log_rv"])

        decade_agg = (
            mp.groupby(["idiom", "primary_denom", "decade"])
            .agg(score=("score_paraphrase", "mean"), log_rv=("log_rv", "mean"))
            .reset_index()
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colours = plt.cm.tab20.colors

        for ax, excl in zip(axes, [False, True]):
            subset = decade_agg
            if excl:
                subset = subset[~subset["idiom"].isin(ANOMALOUS_IDIOMS)]

            idiom_list = sorted(subset["idiom"].unique())
            for i, idiom in enumerate(idiom_list):
                g = subset[subset["idiom"] == idiom]
                colour = colours[i % len(colours)]
                ax.scatter(g["log_rv"], g["score"], alpha=0.55, s=18,
                           color=colour, label=idiom)

            valid = subset.dropna()
            if len(valid) > 2:
                m, b = np.polyfit(valid["log_rv"], valid["score"], 1)
                xr = np.linspace(valid["log_rv"].min(), valid["log_rv"].max(), 100)
                ax.plot(xr, m * xr + b, "k--", linewidth=1.5,
                        label=f"OLS β={m:.4f}")

            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
            ax.set_xlabel("log(real value of denomination)")
            ax.set_ylabel("score_paraphrase  (trivial ← 0 → significant)")
            ttl = f"{model}: {'excl. anomalous' if excl else 'all idioms'}"
            ax.set_title(ttl)
            ax.legend(fontsize=5, ncol=2)

        fig.tight_layout()
        fig_path = fig_dir / f"score_vs_logrv_{model}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info("Saved figure → %s", fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 06: Regression of idiom significance score on log(RV)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_regression(
        data_dir=args.data_dir,
        n_permutations=args.n_permutations,
        force=args.force,
    )


if __name__ == "__main__":
    main()
