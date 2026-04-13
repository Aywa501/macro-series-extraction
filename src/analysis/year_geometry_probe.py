"""Appendix — Year carrier sentence geometry probe + comparison with monarch probe.

Embeds sentences of the form "the year is 1450 ." across the same year range
as the monarch probe, runs the same layer-by-layer geometry analysis, then
prints a side-by-side comparison.

Two distinct questions:
  A. Explicit year probe  — does BERT have learnt representations of year
     tokens per se?  ("the year is 1450")
  B. Monarch probe (rerun or load from CSV) — does BERT represent historical
     time through factual/contextual knowledge?

Comparing A vs B tells us:
  - If A >> B : BERT's temporal geometry is driven by year-number tokens
  - If B >> A : temporal geometry is driven by historical semantic content
  - If A ≈ B  : both carry similar signal (or neither does)

Usage
-----
    python src/appendix/year_geometry_probe.py
    python src/appendix/year_geometry_probe.py --model macberth
    python src/appendix/year_geometry_probe.py --step 25 --no-rerun-monarch
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR      = PROJECT_ROOT / "data" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Carrier sentence templates — {year} is substituted with the integer year.
# Deliberately content-free: we want to isolate BERT's representation of the
# year token itself, not any associated historical facts.
# ---------------------------------------------------------------------------
CARRIER_TEMPLATES = [
    "the year is {year} .",
    "it is the year {year} .",
    "this took place in the year {year} .",
    "the date is {year} .",
    "this was written in {year} .",
    "{year} was a notable year .",
    "the document is from the year {year} .",
    "events of {year} are described here .",
]


# ---------------------------------------------------------------------------
# Shared geometry analysis (same metrics as monarch probe)
# ---------------------------------------------------------------------------

def analyse_layer(X: np.ndarray, y: np.ndarray, label: str = "") -> dict:
    scaler = StandardScaler(with_std=False)
    X_c = scaler.fit_transform(X)
    n   = len(y)
    cv  = min(5, n)

    r2_ridge = cross_val_score(
        RidgeCV(alphas=np.logspace(-3, 6, 20)), X_c, y, cv=cv, scoring="r2"
    ).mean()

    r2_knn = cross_val_score(
        KNeighborsRegressor(n_neighbors=min(3, n - 1)), X_c, y,
        cv=cv, scoring="r2"
    ).mean()

    pca = PCA(n_components=min(10, n - 1))
    pca.fit(X_c)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_dims_90 = int(np.searchsorted(cum, 0.90)) + 1

    pc1_scores = pca.transform(X_c)[:, 0]
    rho, p = spearmanr(pc1_scores, y)

    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X_c)
    np.fill_diagonal(D, np.inf)
    k = min(5, n - 1)
    nn_coherence = []
    for i in range(n):
        nn_idx  = np.argsort(D[i])[:k]
        within  = np.sum(np.abs(y[nn_idx] - y[i]) <= 100) / k
        nn_coherence.append(within)

    return {
        "r2_ridge":     r2_ridge,
        "r2_knn":       r2_knn,
        "pc1_var":      float(evr[0]),
        "pc2_var":      float(evr[1]) if len(evr) > 1 else 0.0,
        "cum_var_3pc":  float(cum[2]) if len(cum) > 2 else float(cum[-1]),
        "n_dims_90pct": n_dims_90,
        "spearman_rho": float(rho),
        "spearman_p":   float(p),
        "nn_coherence": float(np.mean(nn_coherence)),
        "X_c":          X_c,
        "pca":          pca,
    }


def embed_year_sentences(
    model,
    tokenizer,
    years: np.ndarray,
    device: torch.device,
    n_layers: int,
) -> np.ndarray:
    """Embed carrier sentences for each year; return (n_years, n_layers, hidden)."""
    embs = []
    for i, yr in enumerate(years):
        print(f"  [{i+1:03d}/{len(years)}] year={yr} …", end="\r", flush=True)
        sentences = [t.format(year=int(yr)) for t in CARRIER_TEMPLATES]
        sent_embs = []
        for sent in sentences:
            enc = tokenizer(sent, return_tensors="pt",
                            truncation=True, max_length=32).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            layer_cls = np.stack(
                [hs[0, 0, :].cpu().numpy() for hs in out.hidden_states[1:]],
                axis=0,
            )  # (n_layers, hidden)
            sent_embs.append(layer_cls)
        embs.append(np.mean(sent_embs, axis=0))  # (n_layers, hidden)
    print()
    return np.stack(embs, axis=0)  # (n_years, n_layers, hidden)


def load_or_run_monarch(model_key: str) -> pd.DataFrame | None:
    """Load monarch geometry CSV if it exists."""
    csv = OUT_DIR / f"temporal_geometry_{model_key}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return None


def run(model_key: str, step: int, rerun_monarch: bool) -> None:
    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased" if model_key == "bert" else "emanjavacas/MacBERTh"
    print(f"Loading {model_name} …", flush=True)
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded. {n_layers} layers\n")

    # Year grid: same span as monarchs (1199–1936), evenly spaced
    years = np.arange(1200, 1937, step, dtype=float)
    print(f"Year probe: {len(years)} years, {step}-year step, "
          f"{len(CARRIER_TEMPLATES)} templates each")
    print("Embedding year sentences …")
    year_embs = embed_year_sentences(model, tokenizer, years, device, n_layers)
    print(f"Shape: {year_embs.shape}\n")

    # Analyse every layer for year probe
    print("Analysing year probe layers …")
    year_results = []
    for i in range(n_layers):
        r = analyse_layer(year_embs[:, i, :], years)
        year_results.append(r)
        print(f"  L{i+1:02d}  Ridge_R²={r['r2_ridge']:+.3f}  "
              f"kNN_R²={r['r2_knn']:+.3f}  "
              f"|ρ|={abs(r['spearman_rho']):.3f}  "
              f"NN_coh={r['nn_coherence']:.2f}")

    # Load monarch results
    monarch_df = load_or_run_monarch(model_key)
    if rerun_monarch or monarch_df is None:
        if monarch_df is None:
            print("\nMonarch CSV not found — run temporal_geometry_probe.py first.")
            print("Continuing with year probe only.\n")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"GEOMETRY COMPARISON — {model_name}")
    print(f"{'':5}  {'--- YEAR CARRIER ---':^35}  {'--- MONARCH CONTEXT ---':^35}")
    print(f"{'Layer':>5}  {'Ridge_R²':>9} {'|ρ|':>7} {'NN_coh':>7} {'dims90':>6}  "
          f"{'Ridge_R²':>9} {'|ρ|':>7} {'NN_coh':>7} {'dims90':>6}")
    print("-" * 85)
    for i, yr in enumerate(year_results):
        ystr = (f"  {yr['r2_ridge']:>+9.3f} {abs(yr['spearman_rho']):>7.3f} "
                f"{yr['nn_coherence']:>7.2f} {yr['n_dims_90pct']:>6}")
        if monarch_df is not None and i < len(monarch_df):
            mr = monarch_df.iloc[i]
            mstr = (f"  {mr['r2_ridge']:>+9.3f} {abs(mr['spearman_rho']):>7.3f} "
                    f"{mr['nn_coherence']:>7.2f} {int(mr['n_dims_90pct']):>6}")
        else:
            mstr = "  " + " " * 33
        print(f"  L{i+1:02d}{ystr}{mstr}")
    print("=" * 85)

    # Summary statistics
    y_rho_max = max(abs(r["spearman_rho"]) for r in year_results)
    y_nn_max  = max(r["nn_coherence"] for r in year_results)
    y_r2_max  = max(r["r2_ridge"] for r in year_results)
    print(f"\n  Year probe   — best |ρ|: {y_rho_max:.3f}  "
          f"best NN_coh: {y_nn_max:.2f}  best Ridge_R²: {y_r2_max:+.3f}")
    if monarch_df is not None:
        m_rho_max = monarch_df["spearman_rho"].abs().max()
        m_nn_max  = monarch_df["nn_coherence"].max()
        m_r2_max  = monarch_df["r2_ridge"].max()
        print(f"  Monarch probe — best |ρ|: {m_rho_max:.3f}  "
              f"best NN_coh: {m_nn_max:.2f}  best Ridge_R²: {m_r2_max:+.3f}")
        print()
        print("  Interpretation:")
        if y_rho_max > m_rho_max + 0.1:
            print("    → Year tokens encode time MORE linearly than historical context")
        elif m_rho_max > y_rho_max + 0.1:
            print("    → Historical context encodes time MORE coherently than year tokens")
        else:
            print("    → Both encode temporal ordering similarly well")

        if y_nn_max > m_nn_max + 0.05:
            print("    → Year-sentence embeddings cluster more tightly by era")
        elif m_nn_max > y_nn_max + 0.05:
            print("    → Monarch-context embeddings cluster more tightly by era")
        else:
            print("    → Similar neighbourhood temporal coherence")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    layer_nums = np.arange(1, n_layers + 1)

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.38)

    # --- Row 0: Layer metric profiles ---
    ax_rho = fig.add_subplot(gs[0, 0:2])
    ax_nn  = fig.add_subplot(gs[0, 2:4])

    yr_rhos = [abs(r["spearman_rho"]) for r in year_results]
    yr_nns  = [r["nn_coherence"] for r in year_results]

    ax_rho.plot(layer_nums, yr_rhos, "o-", color="#2c7bb6",
                linewidth=2, label="Year carrier", zorder=3)
    ax_nn.plot(layer_nums,  yr_nns,  "o-", color="#2c7bb6",
               linewidth=2, label="Year carrier", zorder=3)

    if monarch_df is not None:
        mn_rhos = monarch_df["spearman_rho"].abs().values
        mn_nns  = monarch_df["nn_coherence"].values
        ax_rho.plot(layer_nums, mn_rhos, "s--", color="#d7191c",
                    linewidth=2, label="Monarch context", zorder=3)
        ax_nn.plot(layer_nums,  mn_nns,  "s--", color="#d7191c",
                   linewidth=2, label="Monarch context", zorder=3)

    ax_rho.set_xlabel("Layer"); ax_rho.set_ylabel("|Spearman ρ|  (PC1 vs year)")
    ax_rho.set_title("Temporal ordering in PC1")
    ax_rho.set_ylim(0, 1.05); ax_rho.set_xticks(layer_nums)
    ax_rho.axhline(0.5, color="grey", linewidth=0.7, linestyle=":")
    ax_rho.legend(fontsize=9)

    random_nn = min(5, len(years) - 1) / len(years)
    ax_nn.axhline(random_nn, color="grey", linewidth=0.9, linestyle="--",
                  label=f"Random ({random_nn:.0%})")
    ax_nn.set_xlabel("Layer"); ax_nn.set_ylabel("NN temporal coherence")
    ax_nn.set_title("Fraction of 5-NN within ±100 years")
    ax_nn.set_ylim(0, 1.05); ax_nn.set_xticks(layer_nums)
    ax_nn.legend(fontsize=9)

    # --- Rows 1–2: PCA projections for 4 layers (year probe) ---
    show_layers = [0, 3, 7, 11][:n_layers]
    cmap  = plt.cm.plasma
    norm  = plt.Normalize(vmin=years.min(), vmax=years.max())

    for plot_i, layer_idx in enumerate(show_layers):
        row = 1 + plot_i // 2
        col = (plot_i % 2) * 2
        ax  = fig.add_subplot(gs[row, col:col+2])

        r   = year_results[layer_idx]
        X_c = r["X_c"]

        pca2 = PCA(n_components=2).fit(X_c)
        C    = pca2.transform(X_c)
        ev   = pca2.explained_variance_ratio_

        sc = ax.scatter(C[:, 0], C[:, 1], c=years, cmap=cmap,
                        norm=norm, s=25, zorder=3)

        # Connect in year order
        order = np.argsort(years)
        ax.plot(C[order, 0], C[order, 1], "-", color="grey",
                linewidth=0.5, alpha=0.35, zorder=2)

        # Label every 5th year
        for j in range(0, len(years), max(1, len(years) // 12)):
            ax.annotate(str(int(years[j])), (C[j, 0], C[j, 1]),
                        fontsize=5, alpha=0.75,
                        xytext=(2, 2), textcoords="offset points")

        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=8)
        ax.set_title(
            f"Year probe · Layer {layer_idx+1}  "
            f"|ρ|={abs(year_results[layer_idx]['spearman_rho']):.2f}  "
            f"NN={year_results[layer_idx]['nn_coherence']:.2f}",
            fontsize=9,
        )

    fig.suptitle(
        f"{model_name} — temporal geometry: year carrier sentences\n"
        f"('{CARRIER_TEMPLATES[0]}' etc., {step}-year steps, "
        f"{len(CARRIER_TEMPLATES)} templates averaged)",
        fontsize=12, y=1.01,
    )
    out_png = OUT_DIR / f"year_geometry_{model_key}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot → {out_png}")

    # Save metrics CSV
    out_csv = OUT_DIR / f"year_geometry_{model_key}.csv"
    pd.DataFrame([
        {"layer": i+1, **{k: v for k, v in r.items()
                          if k not in ("X_c", "pca")}}
        for i, r in enumerate(year_results)
    ]).to_csv(out_csv, index=False)
    print(f"Metrics → {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Year carrier probe + comparison with monarch geometry."
    )
    parser.add_argument("--model", default="bert",
                        choices=["bert", "macberth"])
    parser.add_argument("--step", type=int, default=25,
                        help="Year step between probe points (default: 25).")
    parser.add_argument("--no-rerun-monarch", action="store_true",
                        help="Skip monarch re-run; use existing CSV if present.")
    args = parser.parse_args()
    run(
        model_key=args.model,
        step=args.step,
        rerun_monarch=not args.no_rerun_monarch,
    )


if __name__ == "__main__":
    main()
