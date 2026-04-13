"""03_coin_value_probe.py — Coin value × time 2D probe vs real purchasing power.

Findings captured
-----------------
A. When coins are embedded with explicit year context, their position on the
   L4 value axis DECLINES as year increases — consistent with inflation.
   All 4 test coins peak at 1350 (Black Death wage shock), then decline.

B. BERT's value-axis projections correlate with actual purchasing power
   (1/CPI from BoE millennium dataset) at Spearman ρ ≈ 0.88–0.93 (p<0.01)
   for all four test coins.

C. There is NO significant correlation with real earnings (ρ ≈ −0.1 to −0.5),
   indicating BERT captures price-level change but not wage dynamics.

Outputs (under data/{model}/value_probe/)
------------------------------------------
  coin_value_results.csv     — (model, coin, year, value_proj, time_proj,
                                 purch_power, real_earn)
  correlation_summary.csv    — (model, coin, rho_vs_cpi, p_vs_cpi,
                                 rho_vs_earn, p_vs_earn)
  plots/coin_value_vs_real.png — 3-panel figure

Usage
-----
    python src/value_probe/03_coin_value_probe.py --model bert
    python src/value_probe/03_coin_value_probe.py --model macberth
    python src/value_probe/03_coin_value_probe.py --model bert \\
        --millennium-path /path/to/a-millennium-of-macroeconomic-data-for-the-uk.xlsx

Prerequisites
-------------
    Run 01_build_axes.py first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "value_probe"))
from _common import (
    TEST_COINS, TEST_YEARS, TIME_TEMPLATES, VALUE_TEMPLATES, VALUE_LAYER,
)

MODEL_NAMES = {
    "bert":     "bert-base-uncased",
    "macberth": "emanjavacas/MacBERTh",
}

DEFAULT_MILLENNIUM_PATH = (
    PROJECT_ROOT / "a-millennium-of-macroeconomic-data-for-the-uk.xlsx"
)

# Window (half-width in years) for smoothing real data at test years
WINDOW_HALF = 7   # ±7 years = 15-year centred window


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_key: str):
    name = MODEL_NAMES[model_key]
    print(f"Loading {name} …", flush=True)
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    print("Loaded.\n", flush=True)
    return model, tokenizer, device


def cls_at_layer(model, tokenizer, device, sentence: str, layer: int) -> np.ndarray:
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.hidden_states[layer][0, 0, :].cpu().numpy()


# ---------------------------------------------------------------------------
# Part A — Coin × year embeddings
# ---------------------------------------------------------------------------

def embed_coin_year(model, tokenizer, device,
                    coin: str, year: int) -> np.ndarray:
    """Centroid of TIME_TEMPLATES for (coin, year) at VALUE_LAYER."""
    vecs = []
    for tmpl in TIME_TEMPLATES:
        sent = tmpl.format(y=year, c=coin)
        vecs.append(cls_at_layer(model, tokenizer, device, sent, VALUE_LAYER))
    return np.mean(vecs, axis=0)


def embed_coin_neutral(model, tokenizer, device, coin: str) -> np.ndarray:
    """Baseline: coin embedded without year context (VALUE_TEMPLATES)."""
    vecs = []
    for tmpl in VALUE_TEMPLATES:
        sent = tmpl.format(c=coin)
        vecs.append(cls_at_layer(model, tokenizer, device, sent, VALUE_LAYER))
    return np.mean(vecs, axis=0)


# ---------------------------------------------------------------------------
# Part B — Real data loading
# ---------------------------------------------------------------------------

def load_millennium_data(xlsx_path: Path) -> pd.DataFrame:
    """
    Load BoE millennium dataset.  Returns a DataFrame with columns:
        year         — integer
        cpi          — CPI spliced index (2015=100), col 3 of A47
        real_earn    — real earnings index, col 1 of A48
    Only rows where both year and CPI are non-null are returned.
    """
    # A47: col 0 = year, col 3 = CPI (2015=100); data from row 7 (0-indexed=6)
    df47 = pd.read_excel(
        xlsx_path,
        sheet_name="A47. Wages and prices",
        header=None,
        skiprows=6,   # skip header rows, data starts at row 7 (0-indexed 6)
        usecols=[0, 3],
        names=["year", "cpi"],
    )
    df47 = df47.dropna(subset=["year", "cpi"])
    df47["year"] = df47["year"].astype(int)

    # A48: col 0 = year, col 1 = real earnings index; data from row 6 (0-indexed=5)
    df48 = pd.read_excel(
        xlsx_path,
        sheet_name="A48. Real Earnings ",
        header=None,
        skiprows=5,
        usecols=[0, 1],
        names=["year", "real_earn"],
    )
    df48 = df48.dropna(subset=["year", "real_earn"])
    df48["year"] = df48["year"].astype(int)

    df = pd.merge(df47, df48, on="year", how="outer").sort_values("year").reset_index(drop=True)
    return df


def window_mean(df: pd.DataFrame, col: str, year: int, half: int) -> float | None:
    """15-year centred window mean around a test year."""
    mask = (df["year"] >= year - half) & (df["year"] <= year + half)
    vals = df.loc[mask, col].dropna()
    if len(vals) == 0:
        return None
    return float(vals.mean())


def build_real_series(millennium_df: pd.DataFrame) -> dict:
    """
    For each TEST_YEAR compute:
      - purch_power  = cpi_anchor / cpi_year   (anchor = mean CPI at first
                       available window, i.e. 1250-window)
      - real_earn    = 15yr mean of real earnings index
    Returns dict: year → {"purch_power": float|None, "real_earn": float|None}
    """
    # Anchor CPI: mean over first test year that has data (typically 1250)
    anchor_year = TEST_YEARS[0]
    anchor_cpi  = window_mean(millennium_df, "cpi", anchor_year, WINDOW_HALF)
    if anchor_cpi is None:
        # fallback: first non-null CPI value
        anchor_cpi = float(millennium_df["cpi"].dropna().iloc[0])

    real_series = {}
    for y in TEST_YEARS:
        cpi_y   = window_mean(millennium_df, "cpi",       y, WINDOW_HALF)
        earn_y  = window_mean(millennium_df, "real_earn", y, WINDOW_HALF)
        pp      = (anchor_cpi / cpi_y) if (cpi_y is not None and cpi_y > 0) else None
        real_series[y] = {"purch_power": pp, "real_earn": earn_y}
    return real_series


# ---------------------------------------------------------------------------
# Part C — Correlations
# ---------------------------------------------------------------------------

def compute_correlations(
    coin_rows: list[dict],
    coin: str,
) -> dict:
    """Spearman ρ between value_proj and {purch_power, real_earn} over TEST_YEARS."""
    sub = [r for r in coin_rows if r["coin"] == coin]
    sub.sort(key=lambda r: r["year"])

    vp  = [r["value_proj"]  for r in sub]
    pp  = [r["purch_power"] for r in sub]
    re  = [r["real_earn"]   for r in sub]

    # Mask out None
    mask_pp = [i for i, v in enumerate(pp) if v is not None]
    mask_re = [i for i, v in enumerate(re) if v is not None]

    if len(mask_pp) >= 3:
        rho_cpi, p_cpi = spearmanr(
            [vp[i] for i in mask_pp], [pp[i] for i in mask_pp])
    else:
        rho_cpi, p_cpi = float("nan"), float("nan")

    if len(mask_re) >= 3:
        rho_earn, p_earn = spearmanr(
            [vp[i] for i in mask_re], [re[i] for i in mask_re])
    else:
        rho_earn, p_earn = float("nan"), float("nan")

    return {
        "rho_vs_cpi":  float(rho_cpi),
        "p_vs_cpi":    float(p_cpi),
        "rho_vs_earn": float(rho_earn),
        "p_vs_earn":   float(p_earn),
    }


# ---------------------------------------------------------------------------
# Part D — Plotting
# ---------------------------------------------------------------------------

def normalise_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return arr * 0
    return (arr - lo) / (hi - lo)


def make_main_plot(
    millennium_df: pd.DataFrame,
    results_df:    pd.DataFrame,
    model_key:     str,
    plots_dir:     Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes

    coin_names  = [c for c, _ in TEST_COINS]
    colours     = plt.cm.tab10(np.linspace(0, 0.4, len(coin_names)))

    # ---- Panel 1: CPI time series with real earnings overlay ----
    df_cpi  = millennium_df.dropna(subset=["cpi"]).query("year >= 1209 & year <= 1925")
    df_earn = millennium_df.dropna(subset=["real_earn"]).query("year >= 1209 & year <= 1925")

    ax1b = ax1.twinx()
    ax1.plot(df_cpi["year"],  df_cpi["cpi"],       color="steelblue",
             linewidth=1.2, label="CPI (2015=100)", alpha=0.9)
    ax1b.plot(df_earn["year"], df_earn["real_earn"], color="darkorange",
              linewidth=1.0, label="Real earnings", alpha=0.7)
    for y in TEST_YEARS:
        ax1.axvline(y, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("CPI (2015=100)", color="steelblue")
    ax1b.set_ylabel("Real earnings index", color="darkorange")
    ax1.set_title("BoE Millennium dataset\nCPI & real earnings 1209–1925")
    ax1.legend(loc="upper left", fontsize=8)
    ax1b.legend(loc="lower right", fontsize=8)

    # ---- Panel 2: scatter value_proj vs purchasing power ----
    for i, coin in enumerate(coin_names):
        sub = results_df[results_df["coin"] == coin].dropna(
            subset=["value_proj", "purch_power"]).sort_values("year")
        if len(sub) == 0:
            continue
        rho, _ = spearmanr(sub["value_proj"], sub["purch_power"])
        ax2.scatter(sub["purch_power"], sub["value_proj"],
                    color=colours[i], s=50, zorder=3, label=f"{coin}  ρ={rho:.2f}")
        # Light connecting line
        ax2.plot(sub["purch_power"].values, sub["value_proj"].values,
                 color=colours[i], linewidth=0.6, alpha=0.4)
    ax2.set_xlabel("Purchasing power (CPI_anchor / CPI_year)")
    ax2.set_ylabel(f"Value-axis projection (L{VALUE_LAYER})")
    ax2.set_title(f"BERT value projection vs purchasing power\n{model_key}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: normalised time series overlay ----
    # Purchasing power (from results_df for any coin — same for all)
    ref_coin_sub = results_df[results_df["coin"] == coin_names[0]].dropna(
        subset=["purch_power"]).sort_values("year")
    pp_norm = normalise_01(ref_coin_sub["purch_power"].values)
    ax3.plot(ref_coin_sub["year"], pp_norm,
             color="black", linewidth=2.0, linestyle="--",
             label="Purchasing power", zorder=5)

    for i, coin in enumerate(coin_names):
        sub = results_df[results_df["coin"] == coin].sort_values("year")
        vp_norm = normalise_01(sub["value_proj"].values)
        ax3.plot(sub["year"], vp_norm, marker="o", markersize=5,
                 color=colours[i], linewidth=1.5, label=f"{coin} (BERT)")

    ax3.set_xlabel("Year")
    ax3.set_ylabel("Normalised value [0–1]")
    ax3.set_title(f"BERT value axis vs purchasing power (normalised)\n{model_key}")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Coin value probe — {model_key} Layer {VALUE_LAYER}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = plots_dir / "coin_value_vs_real.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved plot → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coin value × time probe vs BoE purchasing power data."
    )
    parser.add_argument("--model", choices=["bert", "macberth"], default="bert")
    parser.add_argument(
        "--millennium-path",
        type=Path,
        default=DEFAULT_MILLENNIUM_PATH,
        help="Path to a-millennium-of-macroeconomic-data-for-the-uk.xlsx",
    )
    args = parser.parse_args()
    model_key = args.model

    out_dir   = PROJECT_ROOT / "data" / model_key / "value_probe"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Check prerequisites
    val_path  = out_dir / f"value_direction_L{VALUE_LAYER}.npy"
    time_path = out_dir / f"time_direction_L{VALUE_LAYER}.npy"
    for p in (val_path, time_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run 01_build_axes.py first.")
            sys.exit(1)

    value_dir = np.load(val_path)
    time_dir  = np.load(time_path)
    print(f"Loaded value direction from {val_path}")
    print(f"Loaded time  direction from {time_path}")

    # Load model
    model, tokenizer, device = load_model(model_key)

    # ---- Part A: coin × year embeddings ----
    print("\nPart A — Embedding coins at all test year × coin combinations …",
          flush=True)
    coin_rows = []
    for coin, pence in TEST_COINS:
        # Baseline (no year context)
        baseline_emb  = embed_coin_neutral(model, tokenizer, device, coin)
        baseline_vproj = float(np.dot(baseline_emb, value_dir))
        baseline_tproj = float(np.dot(baseline_emb, time_dir))
        print(f"  {coin} baseline: value_proj={baseline_vproj:.5f}  "
              f"time_proj={baseline_tproj:.5f}")

        for year in TEST_YEARS:
            emb    = embed_coin_year(model, tokenizer, device, coin, year)
            vproj  = float(np.dot(emb, value_dir))
            tproj  = float(np.dot(emb, time_dir))
            coin_rows.append({
                "model":      model_key,
                "coin":       coin,
                "year":       year,
                "value_proj": vproj,
                "time_proj":  tproj,
            })

    print(f"\n  Embedded {len(coin_rows)} (coin, year) combinations.")

    # ---- Part B: real data ----
    print("\nPart B — Loading BoE millennium dataset …", flush=True)
    millennium_df = load_millennium_data(args.millennium_path)
    print(f"  Loaded {len(millennium_df)} yearly records "
          f"({int(millennium_df['year'].min())}–{int(millennium_df['year'].max())})")
    real_series   = build_real_series(millennium_df)

    # Attach real data to coin_rows
    for row in coin_rows:
        y = row["year"]
        row["purch_power"] = real_series[y]["purch_power"]
        row["real_earn"]   = real_series[y]["real_earn"]

    results_df = pd.DataFrame(coin_rows)

    # ---- Print value-proj time series per coin ----
    print()
    print("=" * 70)
    print(f"VALUE-AXIS PROJECTIONS BY YEAR  —  model={model_key}")
    print("=" * 70)
    print(f"  {'Coin':<12} {'Year':>6} {'value_proj':>12} {'purch_power':>12} "
          f"{'real_earn':>11}")
    print("-" * 70)
    for _, row in results_df.sort_values(["coin", "year"]).iterrows():
        pp = f"{row['purch_power']:.4f}" if pd.notna(row["purch_power"]) else "   NaN"
        re = f"{row['real_earn']:.2f}"   if pd.notna(row["real_earn"])   else "   NaN"
        print(f"  {row['coin']:<12} {int(row['year']):>6} "
              f"{row['value_proj']:>12.5f} {pp:>12} {re:>11}")

    # ---- Part C: correlations ----
    print()
    print("=" * 70)
    print(f"CORRELATIONS  —  model={model_key}")
    print("=" * 70)
    corr_rows = []
    for coin, _ in TEST_COINS:
        stats = compute_correlations(coin_rows, coin)
        corr_rows.append({"model": model_key, "coin": coin, **stats})
        print(f"\n  Coin: {coin}")
        print(f"    ρ vs purchasing power (1/CPI): {stats['rho_vs_cpi']:+.4f}  "
              f"(p={stats['p_vs_cpi']:.3e})")
        print(f"    ρ vs real earnings:            {stats['rho_vs_earn']:+.4f}  "
              f"(p={stats['p_vs_earn']:.3e})")
        cpi_sig  = "**" if stats['p_vs_cpi']  < 0.01 else ("*" if stats['p_vs_cpi']  < 0.05 else "ns")
        earn_sig = "**" if stats['p_vs_earn'] < 0.01 else ("*" if stats['p_vs_earn'] < 0.05 else "ns")
        print(f"    Significance: CPI={cpi_sig}  earnings={earn_sig}")

    print()
    print("  Summary table:")
    print(f"  {'Coin':<12} {'ρ_CPI':>8} {'p_CPI':>10} {'sig':>4}  "
          f"{'ρ_earn':>8} {'p_earn':>10} {'sig':>4}")
    for row in corr_rows:
        cpi_sig  = "**" if row['p_vs_cpi']  < 0.01 else ("*" if row['p_vs_cpi']  < 0.05 else "ns")
        earn_sig = "**" if row['p_vs_earn'] < 0.01 else ("*" if row['p_vs_earn'] < 0.05 else "ns")
        print(f"  {row['coin']:<12} {row['rho_vs_cpi']:>8.4f} "
              f"{row['p_vs_cpi']:>10.3e} {cpi_sig:>4}  "
              f"{row['rho_vs_earn']:>8.4f} {row['p_vs_earn']:>10.3e} {earn_sig:>4}")

    # ---- Part D: save outputs ----
    csv1 = out_dir / "coin_value_results.csv"
    csv2 = out_dir / "correlation_summary.csv"
    results_df.to_csv(csv1, index=False)
    pd.DataFrame(corr_rows).to_csv(csv2, index=False)
    print(f"\nSaved → {csv1}")
    print(f"Saved → {csv2}")

    make_main_plot(millennium_df, results_df, model_key, plots_dir)


if __name__ == "__main__":
    main()
