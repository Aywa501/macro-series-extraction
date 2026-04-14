"""04_broader_series.py — Test 5 BoE millennium series against BERT semantic axes.

For each series we build a dedicated semantic axis from pole sentences, embed
year-contextualised phrases at all 28 test years, project onto the axis, and
correlate with the real historical data using Spearman ρ.

Series tested
-------------
  wheat_price   Allen basket-of-necessities cost (A47 col 59, 1264–1913)
                Proxy for food/grain prices. Expected POSITIVE.
  coin_supply   Total coin in circulation, spliced (A22 col 10, 1086–2016)
                Tests monetary abundance vs scarcity. UNKNOWN.
  population    Population of England, millions (A2 col 1, 1086–1870)
                Expected NULL — control. Demographic count, not linguistically
                explicit.
  wages         Real consumption earnings (A48 col 1, 1209–2016) + nominal
                wages (A47 col 1) as secondary.  KEY TEST — did the coin axis
                fail because BERT doesn't encode labour value at all, or because
                a coin-derived axis is the wrong probe?
  trade_volume  Composite export volumes (A35 col 6, 1280–2016)
                Trade/commerce vocabulary is historically rich. UNKNOWN.

Outputs (under data/{model}/value_probe/)
------------------------------------------
  broader_series_results.csv       — (model, series, year, bert_proj, real_value)
  broader_series_correlations.csv  — (model, series, rho, p, sig)
  plots/broader_series.png         — 5-panel normalised time-series overlay

Usage
-----
    python3 src/analysis/04_broader_series.py --model bert
    python3 src/analysis/04_broader_series.py --model macberth --layer 12
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
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _common import TEST_YEARS

MODEL_NAMES = {
    "bert":     "bert-base-uncased",
    "macberth": "emanjavacas/MacBERTh",
}

MILLENNIUM_PATH = (
    PROJECT_ROOT / "a-millennium-of-macroeconomic-data-for-the-uk.xlsx"
    if (PROJECT_ROOT / "a-millennium-of-macroeconomic-data-for-the-uk.xlsx").exists()
    else PROJECT_ROOT.parent / "a-millennium-of-macroeconomic-data-for-the-uk.xlsx"
)

WINDOW_HALF = 7   # ±7 years = 15-year centred window

# ---------------------------------------------------------------------------
# Series definitions
# Each entry: axis_low, axis_high (pole sentences), year_templates,
# xlsx_sheet, xlsx_col, skiprows, and optional secondary_data.
# ---------------------------------------------------------------------------

SERIES = {
    "wheat_price": {
        "title": "Food/grain prices\n(Allen basket of necessities)",
        # Dearth/plenty vocabulary — the language contemporaries actually used
        # when grain was cheap vs. dear. No prices stated; categorical contrast.
        "axis_low": [
            "the harvest was plentiful and the granaries were full .",
            "bread was cheap and corn was abundant that season .",
            "the year brought a good harvest and grain sold low .",
            "the barns were well stocked and no man lacked for bread .",
        ],
        "axis_high": [
            "the harvest failed and bread was dear .",
            "the poor went hungry for grain was scarce .",
            "famine gripped the land and the people cried out for bread .",
            "corn fetched a high price for the dearth was great .",
        ],
        "year_templates": [
            "in {y} , wheat was sold at the market .",
            "the bushel of wheat changed hands in {y} .",
            "in {y} , a quarter of grain was traded .",
            "the price of rye was recorded in {y} .",
        ],
        "xlsx_sheet":  "A47. Wages and prices",
        "xlsx_usecols": [0, 59],
        "xlsx_names":   ["year", "value"],
        "skiprows":    6,
        "direction":   "higher = more expensive",
    },
    "coin_supply": {
        "title": "Coin in circulation\n(total, spliced series)",
        # Barter/kind vs. sterling coin — encodes monetary scarcity through the
        # *instrument* of exchange, not volume. Categorical: tally/kind vs. ready money.
        "axis_low": [
            "goods were exchanged for other goods at the fair .",
            "debts were settled in cloth and kind for there was no coin .",
            "the merchant took grain in place of silver .",
            "tallies and notches served where coin was lacking .",
        ],
        "axis_high": [
            "payment was made in good full-weight silver .",
            "coin passed freely and sterling was in every purse .",
            "the merchant counted out silver pennies without hesitation .",
            "ready money was to be had and debts were paid in coin .",
        ],
        "year_templates": [
            "in {y} , a coin changed hands .",
            "money circulated in {y} .",
            "in {y} , coins were used in trade .",
            "silver was exchanged in {y} .",
        ],
        "xlsx_sheet":  "A22. Coin in circulation",
        "xlsx_usecols": [0, 10],
        "xlsx_names":   ["year", "value"],
        "skiprows":    4,
        "direction":   "higher = more coin",
    },
    "population": {
        "title": "Population of England\n(millions, control)",
        # Crowding/density vocabulary — no explicit numbers, to avoid the
        # archaic-register artefact that drove the v1 pole-swap result.
        # If ρ drops to ns, v1 signal was temporal register not demography.
        "axis_low": [
            "the hamlet lay quiet in the valley and few souls dwelt there .",
            "the road was empty and the fields lay without a man in sight .",
            "the countryside was thinly settled and villages far apart .",
            "a man might walk a day and meet no other traveller .",
        ],
        "axis_high": [
            "the city lanes were thronged and the streets never still .",
            "everywhere one turned there were people pressing and jostling .",
            "the town was so crowded that lodgings were hard to find .",
            "folk swarmed in every alley and the market was thick with people .",
        ],
        "year_templates": [
            "in {y} , people lived in the village .",
            "the townspeople gathered in {y} .",
            "in {y} , folk went about their lives .",
            "people worked the land in {y} .",
        ],
        "xlsx_sheet":  "A2. Pop of Eng & GB 1086-1870",
        "xlsx_usecols": [0, 1],
        "xlsx_names":   ["year", "value"],
        "skiprows":    8,
        "direction":   "higher = more people",
    },
    "wages": {
        "title": "Real wages\n(diet-of-labourer axis)",
        # Diet axis — real wages tracked through what workers could afford to eat.
        # Meat-eating vs. bread-eating is a well-attested economic history proxy;
        # vocabulary is concrete and neither pole is numerically or temporally marked.
        "axis_low": [
            "the labourer lived on black bread and thin pottage .",
            "his dinner was a crust of barley bread and a cup of water .",
            "the poor ate oats and pulse and little else .",
            "coarse bread and salt was all the working man could afford .",
        ],
        "axis_high": [
            "the labourer ate beef and white bread for his dinner .",
            "a joint of mutton and good ale was the workman's daily meal .",
            "the working man could afford white wheaten bread and butter .",
            "meat and cheese were common fare at the labourer's table .",
        ],
        "year_templates": [
            "in {y} , a labourer was paid for his work .",
            "wages were given out in {y} .",
            "in {y} , a worker earned his keep .",
            "a craftsman worked for pay in {y} .",
        ],
        "xlsx_sheet":  "A48. Real Earnings ",
        "xlsx_usecols": [0, 1],
        "xlsx_names":   ["year", "value"],
        "skiprows":    4,
        "direction":   "higher = higher real wages",
        # Secondary: also load nominal wages for cross-comparison
        "secondary": {
            "title": "Nominal wages",
            "xlsx_sheet":  "A47. Wages and prices",
            "xlsx_usecols": [0, 1],
            "xlsx_names":   ["year", "value"],
            "skiprows":    6,
        },
    },
    "trade_volume": {
        "title": "Export trade volume\n(composite, 1280–2016)",
        # Port-activity vocabulary — categorical empty/idle vs. thronged/busy quays.
        # Avoids cargo quantities entirely; contrasts port *states* instead.
        "axis_low": [
            "the harbour was empty and the quays lay deserted .",
            "no ships rode at anchor and the wharves were silent .",
            "merchants were few and trade had ceased at the port .",
            "the docks stood idle and no goods were loaded or unloaded .",
        ],
        "axis_high": [
            "ships crowded the harbour and the quays were full of merchants .",
            "bales and barrels covered the wharves and vessels came and went .",
            "the port was loud with the work of lading and unloading .",
            "trade flourished and ships departed daily with their cargoes .",
        ],
        "year_templates": [
            "in {y} , merchants sold goods at the fair .",
            "trade was conducted in {y} .",
            "in {y} , goods were bought and sold .",
            "merchants exchanged wares in {y} .",
        ],
        "xlsx_sheet":  "A35. Trade volumes and prices",
        "xlsx_usecols": [0, 6],
        "xlsx_names":   ["year", "value"],
        "skiprows":    5,
        "direction":   "higher = more trade",
    },
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_key: str, device: torch.device):
    name = MODEL_NAMES[model_key]
    print(f"Loading {name} on {device} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    print("Loaded.\n", flush=True)
    return model, tokenizer


def cls_at_layer(model, tokenizer, device, sentence: str, layer: int) -> np.ndarray:
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.hidden_states[layer][0, 0, :].cpu().numpy()


def embed_sentences(model, tokenizer, device, sentences: list[str],
                    layer: int) -> np.ndarray:
    """Embed a list of sentences and return mean CLS vector."""
    vecs = [cls_at_layer(model, tokenizer, device, s, layer) for s in sentences]
    return np.mean(vecs, axis=0)


# ---------------------------------------------------------------------------
# Axis building
# ---------------------------------------------------------------------------

def build_axis(model, tokenizer, device, low_sents: list[str],
               high_sents: list[str], layer: int) -> np.ndarray:
    """Build semantic axis as mean(high) − mean(low), normalised."""
    low_emb  = embed_sentences(model, tokenizer, device, low_sents,  layer)
    high_emb = embed_sentences(model, tokenizer, device, high_sents, layer)
    direction = high_emb - low_emb
    return direction / (np.linalg.norm(direction) + 1e-12)


# ---------------------------------------------------------------------------
# Year-context embeddings
# ---------------------------------------------------------------------------

def embed_year_series(model, tokenizer, device, year_templates: list[str],
                      layer: int) -> dict[int, np.ndarray]:
    """For each test year, embed all templates and return mean CLS vector."""
    result = {}
    for y in TEST_YEARS:
        sents = [t.format(y=y) for t in year_templates]
        result[y] = embed_sentences(model, tokenizer, device, sents, layer)
    return result


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------

def load_series(cfg: dict) -> pd.DataFrame:
    df = pd.read_excel(
        MILLENNIUM_PATH,
        sheet_name=cfg["xlsx_sheet"],
        header=None,
        skiprows=cfg["skiprows"],
        usecols=cfg["xlsx_usecols"],
        names=cfg["xlsx_names"],
    )
    df = df.dropna(subset=["year"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    return df


def window_mean(df: pd.DataFrame, year: int, half: int = WINDOW_HALF) -> float | None:
    mask = (df["year"] >= year - half) & (df["year"] <= year + half)
    vals = df.loc[mask, "value"].dropna()
    return float(vals.mean()) if len(vals) > 0 else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def normalise_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-12:
        return arr * 0
    return (arr - lo) / (hi - lo)


def make_plot(all_results: dict[str, dict], model_key: str, layer: int,
              plots_dir: Path, tag: str = "") -> None:
    n  = len(SERIES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)

    for ax, (series_key, cfg) in zip(axes, SERIES.items()):
        rows = all_results[series_key]
        years = [r["year"] for r in rows]
        proj  = np.array([r["bert_proj"] for r in rows], dtype=float)
        real  = np.array([r["real_value"] for r in rows], dtype=float)

        proj_n = normalise_01(proj)
        real_n = normalise_01(np.where(np.isnan(real), np.nan, real))

        ax.plot(years, real_n, color="black", linewidth=2.0, linestyle="--",
                label="Real data", zorder=5)
        ax.plot(years, proj_n, color="steelblue", linewidth=1.5, marker="o",
                markersize=3, label="BERT projection", zorder=4)

        # Correlation annotation
        mask = ~np.isnan(real)
        if mask.sum() >= 3:
            rho, p = spearmanr(proj[mask], real[mask])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            ax.set_title(f"{cfg['title']}\nρ={rho:+.3f} {sig}", fontsize=8)
        else:
            ax.set_title(f"{cfg['title']}\nn<3", fontsize=8)

        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Normalised [0–1]", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"BERT semantic axes vs BoE millennium series — {model_key} L{layer}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = plots_dir / f"broader_series{tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved plot → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 5 BoE millennium series against BERT semantic axes."
    )
    parser.add_argument("--model",  choices=["bert", "macberth"], default="bert")
    parser.add_argument("--layer",  type=int, default=4,
                        help="Transformer layer to use (1-indexed, default: 4).")
    parser.add_argument("--tag",    type=str, default="v2",
                        help="Suffix appended to output filenames (default: v2). "
                             "Use empty string to overwrite previous results.")
    args      = parser.parse_args()
    model_key = args.model
    layer     = args.layer
    tag       = f"_{args.tag}" if args.tag else ""

    out_dir   = PROJECT_ROOT / "data" / model_key / "value_probe"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model(model_key, device)

    print(f"Using layer {layer}, device={device}\n")

    # ---- Preload all real data series ----
    print("Loading real data series …", flush=True)
    real_dfs: dict[str, pd.DataFrame] = {}
    secondary_dfs: dict[str, pd.DataFrame] = {}
    for key, cfg in SERIES.items():
        real_dfs[key] = load_series(cfg)
        yr_range = (real_dfs[key]["year"].min(), real_dfs[key]["year"].max())
        print(f"  {key}: {len(real_dfs[key])} rows, {yr_range[0]}–{yr_range[1]}")
        if "secondary" in cfg:
            secondary_dfs[key] = load_series(cfg["secondary"])
            yr2 = (secondary_dfs[key]["year"].min(), secondary_dfs[key]["year"].max())
            print(f"    secondary ({cfg['secondary']['title']}): "
                  f"{len(secondary_dfs[key])} rows, {yr2[0]}–{yr2[1]}")

    # ---- Build axes + embed year series ----
    all_results:  dict[str, list[dict]] = {}
    corr_rows:    list[dict]            = []

    print()
    print("=" * 70)
    print(f"RESULTS — model={model_key}  layer={layer}")
    print("=" * 70)

    for series_key, cfg in SERIES.items():
        print(f"\n--- {series_key}: {cfg['title'].replace(chr(10), ' ')} ---")

        # Build semantic axis
        axis = build_axis(model, tokenizer, device,
                          cfg["axis_low"], cfg["axis_high"], layer)

        # Embed year-context templates
        year_embs = embed_year_series(model, tokenizer, device,
                                      cfg["year_templates"], layer)

        # Project + collect real data
        rows = []
        for y in TEST_YEARS:
            proj      = float(np.dot(year_embs[y], axis))
            real_val  = window_mean(real_dfs[series_key], y)
            rows.append({
                "model":      model_key,
                "series":     series_key,
                "year":       y,
                "bert_proj":  proj,
                "real_value": real_val,
            })
        all_results[series_key] = rows

        # Primary correlation
        proj_arr = np.array([r["bert_proj"]  for r in rows])
        real_arr = np.array([r["real_value"] for r in rows], dtype=float)
        mask     = ~np.isnan(real_arr)
        n_valid  = mask.sum()

        if n_valid >= 3:
            rho, p  = spearmanr(proj_arr[mask], real_arr[mask])
            sig     = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        else:
            rho, p, sig = float("nan"), float("nan"), "n/a"

        corr_rows.append({
            "model": model_key, "series": series_key,
            "label": "primary", "n": int(n_valid),
            "rho": rho, "p": p, "sig": sig,
        })
        print(f"  Primary:   ρ={rho:+.4f}  p={p:.3e}  {sig}  (n={n_valid})")
        print(f"             [{cfg['direction']}]")

        # Secondary correlation (wages series: also test nominal wages)
        if "secondary" in cfg:
            sec_arr = np.array([window_mean(secondary_dfs[series_key], y)
                                for y in TEST_YEARS], dtype=float)
            sec_mask = ~np.isnan(sec_arr)
            n_sec    = sec_mask.sum()
            if n_sec >= 3:
                rho2, p2  = spearmanr(proj_arr[sec_mask], sec_arr[sec_mask])
                sig2      = "**" if p2 < 0.01 else ("*" if p2 < 0.05 else "ns")
            else:
                rho2, p2, sig2 = float("nan"), float("nan"), "n/a"
            corr_rows.append({
                "model": model_key, "series": series_key,
                "label": cfg["secondary"]["title"], "n": int(n_sec),
                "rho": rho2, "p": p2, "sig": sig2,
            })
            print(f"  {cfg['secondary']['title']:<14}: ρ={rho2:+.4f}  "
                  f"p={p2:.3e}  {sig2}  (n={n_sec})")

    # ---- Summary table ----
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Series':<18} {'Label':<18} {'n':>4}  {'rho':>8}  {'p':>10}  sig")
    print("-" * 70)
    for r in corr_rows:
        print(f"  {r['series']:<18} {r['label']:<18} {r['n']:>4}  "
              f"{r['rho']:>+8.4f}  {r['p']:>10.3e}  {r['sig']}")

    # ---- Save CSVs ----
    result_rows = [row for rows in all_results.values() for row in rows]
    csv1 = out_dir / f"broader_series_results{tag}.csv"
    csv2 = out_dir / f"broader_series_correlations{tag}.csv"
    pd.DataFrame(result_rows).to_csv(csv1, index=False)
    pd.DataFrame(corr_rows).to_csv(csv2, index=False)
    print(f"\nSaved → {csv1}")
    print(f"Saved → {csv2}")

    # ---- Plot ----
    make_plot(all_results, model_key, layer, plots_dir, tag=tag)


if __name__ == "__main__":
    main()
