"""Appendix — Temporal geometry probe (standalone).

Does BERT internally represent historical time, and if so what is its geometry?

Method
------
For each monarch, embed a set of factual sentences about their era using
BERT's CLS token at every layer.  Then ask purely geometric questions:

  1. Is temporal order preserved at all in embedding space?
     (Spearman rank correlation: reign-year vs nearest-neighbour order)

  2. What shape does the temporal manifold take?
     (PCA scree plot + 2D projection coloured by year)

  3. Is the structure linear or curved?
     (Compare linear probe R² vs kNN R² on reign-year prediction)

  4. Which layers have the strongest temporal signal?
     (R² profile across all 12 layers)

  5. What is the intrinsic dimensionality?
     (How many PCs needed to explain 90% of variance in the manifold?)

No Penn corpus. No Ridge direction. No injection.
Pure BERT weights — does the model "know" history?

Outputs
-------
  src/appendix/temporal_geometry_{model}.png   — main figure (5 panels)
  src/appendix/temporal_geometry_{model}.csv   — per-layer metrics

Usage
-----
    python src/appendix/temporal_geometry_probe.py
    python src/appendix/temporal_geometry_probe.py --model macberth
    python src/appendix/temporal_geometry_probe.py --layers all
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Monarchs + reign midpoints (used as the target "year" variable)
# ---------------------------------------------------------------------------
MONARCHS = [
    ("John",        1199, 1216),
    ("Henry III",   1216, 1272),
    ("Edward I",    1272, 1307),
    ("Edward II",   1307, 1327),
    ("Edward III",  1327, 1377),
    ("Richard II",  1377, 1399),
    ("Henry IV",    1399, 1413),
    ("Henry V",     1413, 1422),
    ("Henry VI",    1422, 1461),
    ("Edward IV",   1461, 1483),
    ("Richard III", 1483, 1485),
    ("Henry VII",   1485, 1509),
    ("Henry VIII",  1509, 1547),
    ("Edward VI",   1547, 1553),
    ("Mary I",      1553, 1558),
    ("Elizabeth I", 1558, 1603),
    ("James I",     1603, 1625),
    ("Charles I",   1625, 1649),
    ("Cromwell",    1649, 1660),
    ("Charles II",  1660, 1685),
    ("James II",    1685, 1689),
    ("William III", 1689, 1702),
    ("Anne",        1702, 1714),
    ("George I",    1714, 1727),
    ("George II",   1727, 1760),
    ("George III",  1760, 1820),
    ("George IV",   1820, 1830),
    ("William IV",  1830, 1837),
    ("Victoria",    1837, 1901),
    ("Edward VII",  1901, 1910),
    ("George V",    1910, 1936),
]

# For each monarch: multiple sentences about their era.
# Deliberately varied: some name the monarch, some describe events/context.
# This tests whether BERT's representation is about the name or the era.
SENTENCE_TEMPLATES = [
    # Direct name references
    "the reign of {name} was a period of great change in england .",
    "{name} ruled england and shaped its history .",
    "during the reign of {name} , england changed considerably .",
    # Event/era descriptions without naming (filled per monarch below)
    # These are added as per-monarch sentences in MONARCH_SENTENCES
]

# Per-monarch era-describing sentences (no name, just historical context)
# Index matches MONARCHS list order
MONARCH_ERA_SENTENCES = [
    # John 1199-1216
    ["the magna carta limited the power of the english crown .",
     "english barons forced the king to sign a great charter of liberties ."],
    # Henry III 1216-1272
    ["the english king clashed repeatedly with his barons over royal authority .",
     "parliament began to take shape as a governing institution in england ."],
    # Edward I 1272-1307
    ["england conquered wales and attempted to subdue scotland .",
     "the english king expelled the jewish population from england ."],
    # Edward II 1307-1327
    ["the english king suffered a catastrophic defeat at bannockburn .",
     "scottish forces won a decisive victory over the english army ."],
    # Edward III 1327-1377
    ["the hundred years war between england and france began .",
     "the black death devastated the population of england ."],
    # Richard II 1377-1399
    ["the peasants revolt shook the english social order .",
     "the english king was deposed by a rival claimant to the throne ."],
    # Henry IV 1399-1413
    ["the house of lancaster took the english throne by force .",
     "wales rose in rebellion against english rule ."],
    # Henry V 1413-1422
    ["english forces defeated the french at the battle of agincourt .",
     "the treaty of troyes made the english king heir to the french throne ."],
    # Henry VI 1422-1461
    ["england lost most of its french territories .",
     "the wars of the roses began between the houses of york and lancaster ."],
    # Edward IV 1461-1483
    ["the house of york took the english throne from the lancastrians .",
     "the wars of the roses continued between rival claimants ."],
    # Richard III 1483-1485
    ["the young king was confined to the tower of london .",
     "the last plantagenet king died at the battle of bosworth field ."],
    # Henry VII 1485-1509
    ["the tudor dynasty came to power following the wars of the roses .",
     "england was united under a new royal house after decades of civil war ."],
    # Henry VIII 1509-1547
    ["the english church broke from rome and the pope .",
     "the monasteries were dissolved and their lands redistributed ."],
    # Edward VI 1547-1553
    ["protestant reforms transformed the english church .",
     "the book of common prayer was introduced in england ."],
    # Mary I 1553-1558
    ["england returned to the catholic faith under royal decree .",
     "protestants were burned at the stake for heresy in england ."],
    # Elizabeth I 1558-1603
    ["the spanish armada attempted to invade england and failed .",
     "english drama flourished and shakespeare wrote his greatest works ."],
    # James I 1603-1625
    ["the crowns of england and scotland were united under one monarch .",
     "the gunpowder plot attempted to blow up the english parliament ."],
    # Charles I 1625-1649
    ["the english civil war divided the nation between king and parliament .",
     "the king clashed with parliament over taxation and royal prerogative ."],
    # Cromwell 1649-1660
    ["the english king was executed and the country became a republic .",
     "england was governed as a commonwealth without a monarch ."],
    # Charles II 1660-1685
    ["the monarchy was restored after years of republican rule .",
     "the great fire destroyed much of the city of london ."],
    # James II 1685-1689
    ["the glorious revolution replaced the catholic king with a protestant one .",
     "the king fled to france as william of orange invaded england ."],
    # William III 1689-1702
    ["the bill of rights limited the power of the english crown .",
     "the protestant settlement secured english religious independence ."],
    # Anne 1702-1714
    ["the act of union created the kingdom of great britain .",
     "england and scotland were formally merged into a single state ."],
    # George I 1714-1727
    ["the hanoverian dynasty took the british throne from the stuarts .",
     "the first prime minister effectively governed britain in place of the king ."],
    # George II 1727-1760
    ["britain fought france in the war of the austrian succession .",
     "the jacobite rising attempted to restore the stuart line to the throne ."],
    # George III 1760-1820
    ["the american colonies declared independence from britain .",
     "britain fought napoleon and eventually defeated him at waterloo ."],
    # George IV 1820-1830
    ["the prince regent became king following his father s long illness .",
     "the regency era was known for its extravagance and cultural brilliance ."],
    # William IV 1830-1837
    ["the great reform act expanded the british electoral system .",
     "parliament passed legislation abolishing slavery in the british empire ."],
    # Victoria 1837-1901
    ["the british empire reached its greatest extent during this era .",
     "the industrial revolution transformed britain into a manufacturing power ."],
    # Edward VII 1901-1910
    ["the edwardian era followed the long victorian period .",
     "britain formed alliances with france and russia in the early twentieth century ."],
    # George V 1910-1936
    ["britain fought in the first world war from nineteen fourteen .",
     "the royal family changed its name from saxe coburg to windsor ."],
]


def get_sentences_for_monarch(idx: int, name: str) -> list[str]:
    """Return all probe sentences for a monarch (named + era-describing)."""
    named = [t.format(name=name.lower()) for t in SENTENCE_TEMPLATES]
    era   = MONARCH_ERA_SENTENCES[idx]
    return named + era


def embed_all(
    model,
    tokenizer,
    all_sentences: list[list[str]],
    device: torch.device,
    n_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed all sentences.  For each monarch, average CLS across their sentences.

    Returns
    -------
    embeddings : (n_monarchs, n_layers, hidden_dim)
    years      : (n_monarchs,)  — reign midpoint
    """
    monarch_embs = []
    years = []

    for i, ((name, start, end), sentences) in enumerate(zip(MONARCHS, all_sentences)):
        print(f"  [{i+1:02d}/{len(MONARCHS)}] {name} ({len(sentences)} sentences) …",
              end="\r", flush=True)
        mid = (start + end) / 2
        sent_layer_embs = []   # (n_sentences, n_layers, hidden_dim)

        for sent in sentences:
            enc = tokenizer(sent, return_tensors="pt",
                            truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            # hidden_states: tuple of (n_layers+1) tensors each (1, seq, hidden)
            # index 0 = embedding layer, 1..12 = transformer layers
            layer_cls = np.stack(
                [hs[0, 0, :].cpu().numpy() for hs in out.hidden_states[1:]],
                axis=0,
            )  # (n_layers, hidden_dim)
            sent_layer_embs.append(layer_cls)

        # Average over sentences → (n_layers, hidden_dim)
        avg = np.mean(sent_layer_embs, axis=0)
        monarch_embs.append(avg)
        years.append(mid)

    print()
    embeddings = np.stack(monarch_embs, axis=0)   # (n_monarchs, n_layers, hidden_dim)
    return embeddings, np.array(years)


def analyse_layer(
    X: np.ndarray,     # (n_monarchs, hidden_dim)
    y: np.ndarray,     # (n_monarchs,) reign midpoint years
    names: list[str],
    layer_idx: int,
) -> dict:
    """Compute geometry metrics for one layer."""
    scaler = StandardScaler(with_std=False)
    X_c = scaler.fit_transform(X)

    # 1. Linear probe R² (leave-one-out, n=31 so LOO is appropriate)
    ridge = RidgeCV(alphas=np.logspace(-3, 6, 20), fit_intercept=True)
    alphas_cv = np.logspace(-3, 6, 20)
    r2_ridge = cross_val_score(
        RidgeCV(alphas=alphas_cv), X_c, y, cv=min(5, len(y)), scoring="r2"
    ).mean()

    # 2. Non-linear probe R² (kNN)
    r2_knn = cross_val_score(
        KNeighborsRegressor(n_neighbors=3), X_c, y,
        cv=min(5, len(y)), scoring="r2"
    ).mean()

    # 3. PCA
    pca = PCA(n_components=min(10, len(X) - 1))
    pca.fit(X_c)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_dims_90 = int(np.searchsorted(cum, 0.90)) + 1

    # 4. Spearman rank correlation: does PC1 ordering match temporal ordering?
    pc1_scores = pca.transform(X_c)[:, 0]
    rho, p = spearmanr(pc1_scores, y)

    # 5. Nearest-neighbour temporal coherence:
    #    For each monarch, what fraction of their k nearest neighbours
    #    are within ±100 years?
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X_c)
    np.fill_diagonal(D, np.inf)
    k = 5
    nn_coherence = []
    for i in range(len(y)):
        nn_idx = np.argsort(D[i])[:k]
        within = np.sum(np.abs(y[nn_idx] - y[i]) <= 100) / k
        nn_coherence.append(within)

    return {
        "layer":            layer_idx + 1,
        "r2_ridge":         r2_ridge,
        "r2_knn":           r2_knn,
        "linearity_excess": r2_ridge - r2_knn,   # positive → linear, negative → nonlinear
        "pc1_var":          float(evr[0]),
        "pc2_var":          float(evr[1]) if len(evr) > 1 else 0.0,
        "cum_var_3pc":      float(cum[2]) if len(cum) > 2 else float(cum[-1]),
        "n_dims_90pct":     n_dims_90,
        "spearman_rho":     float(rho),
        "spearman_p":       float(p),
        "nn_coherence":     float(np.mean(nn_coherence)),
        "pca":              pca,
        "X_c":              X_c,
    }


def run(model_key: str, layers_to_show: list[int]) -> None:
    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased" if model_key == "bert" else "emanjavacas/MacBERTh"
    print(f"Loading {model_name} …", flush=True)
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name,
                                          output_hidden_states=True).to(device).eval()

    # Infer n_layers
    n_layers  = model.config.num_hidden_layers
    print(f"Loaded. {n_layers} layers, hidden_dim={model.config.hidden_size}\n")

    # Build sentence sets per monarch
    all_sentences = [
        get_sentences_for_monarch(i, name)
        for i, (name, _, _) in enumerate(MONARCHS)
    ]
    n_sents = [len(s) for s in all_sentences]
    print(f"Sentences per monarch: min={min(n_sents)}  max={max(n_sents)}")
    print("Embedding …")
    embeddings, years = embed_all(model, tokenizer, all_sentences, device, n_layers)
    # embeddings: (n_monarchs, n_layers, hidden_dim)
    names = [m[0] for m in MONARCHS]
    print(f"Embedding shape: {embeddings.shape}\n")

    # -----------------------------------------------------------------------
    # Analyse every layer
    # -----------------------------------------------------------------------
    print("Analysing layers …")
    results = []
    for i in range(n_layers):
        r = analyse_layer(embeddings[:, i, :], years, names, i)
        results.append(r)
        print(f"  L{i+1:02d}  ridge_R²={r['r2_ridge']:+.3f}  "
              f"kNN_R²={r['r2_knn']:+.3f}  "
              f"PC1={r['pc1_var']:.2%}  "
              f"Spearman_ρ={r['spearman_rho']:+.3f}  "
              f"NN_coh={r['nn_coherence']:.2f}  "
              f"dims90={r['n_dims_90pct']}")

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("pca", "X_c")} for r in results])
    out_csv = OUT_DIR / f"temporal_geometry_{model_key}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nMetrics → {out_csv}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    n_show = len(layers_to_show)
    fig = plt.figure(figsize=(18, 14))

    # ---- Row 1: Layer profiles (3 panels spanning all columns) ----
    ax_r2    = fig.add_subplot(3, 3, 1)
    ax_pc    = fig.add_subplot(3, 3, 2)
    ax_nn    = fig.add_subplot(3, 3, 3)

    layer_nums = df["layer"].values

    # R² profile
    ax_r2.plot(layer_nums, df["r2_ridge"], "o-", color="#2c7bb6",
               label="Linear (Ridge)", linewidth=1.8)
    ax_r2.plot(layer_nums, df["r2_knn"],   "s--", color="#d7191c",
               label="Non-linear (kNN)", linewidth=1.8)
    ax_r2.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax_r2.set_xlabel("Layer"); ax_r2.set_ylabel("CV R²")
    ax_r2.set_title("Temporal predictability by layer")
    ax_r2.legend(fontsize=8); ax_r2.set_xticks(layer_nums)

    # PC1 variance + Spearman ρ
    ax_pc.bar(layer_nums, df["pc1_var"] * 100, alpha=0.7,
              color="#4dac26", label="PC1 variance %")
    ax_pc2 = ax_pc.twinx()
    ax_pc2.plot(layer_nums, df["spearman_rho"].abs(), "o-",
                color="#b8860b", linewidth=1.8, label="|Spearman ρ| (PC1 vs year)")
    ax_pc.set_xlabel("Layer"); ax_pc.set_ylabel("PC1 variance (%)")
    ax_pc2.set_ylabel("|Spearman ρ|")
    ax_pc.set_title("PC1 structure & temporal ordering")
    ax_pc.set_xticks(layer_nums)
    lines1, labs1 = ax_pc.get_legend_handles_labels()
    lines2, labs2 = ax_pc2.get_legend_handles_labels()
    ax_pc.legend(lines1 + lines2, labs1 + labs2, fontsize=7)

    # Nearest-neighbour temporal coherence
    ax_nn.bar(layer_nums, df["nn_coherence"] * 100, alpha=0.7, color="#7b2d8b")
    ax_nn.set_xlabel("Layer"); ax_nn.set_ylabel("NN temporal coherence (%)")
    ax_nn.set_title("Fraction of 5-NN within ±100 years")
    ax_nn.set_xticks(layer_nums)
    ax_nn.axhline(100 / len(MONARCHS) * 5, color="grey",  # random baseline
                  linewidth=0.9, linestyle="--", label="Random baseline")
    ax_nn.legend(fontsize=8)

    # ---- Rows 2–3: PCA projections for chosen layers ----
    norm = plt.Normalize(vmin=years.min(), vmax=years.max())
    cmap = cm.plasma

    for plot_i, layer_idx in enumerate(layers_to_show):
        row = 1 + plot_i // 3
        col = plot_i % 3
        ax  = fig.add_subplot(3, 3, row * 3 + col + 1)

        r    = results[layer_idx]
        X_c  = r["X_c"]
        pca2 = PCA(n_components=2).fit(X_c)
        C    = pca2.transform(X_c)   # (n_monarchs, 2)
        ev   = pca2.explained_variance_ratio_

        sc = ax.scatter(C[:, 0], C[:, 1], c=years, cmap=cmap,
                        norm=norm, s=60, zorder=3)

        # Connect monarchs in chronological order with a faint line
        order = np.argsort(years)
        ax.plot(C[order, 0], C[order, 1], "-", color="grey",
                linewidth=0.6, alpha=0.4, zorder=2)

        # Label every other monarch
        for j, (name, yr) in enumerate(zip(names, years)):
            if j % 2 == 0:
                ax.annotate(name.split()[0], (C[j, 0], C[j, 1]),
                            fontsize=5.5, alpha=0.8,
                            xytext=(2, 2), textcoords="offset points")

        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=8)
        ax.set_title(
            f"Layer {layer_idx+1}  |  Ridge R²={r['r2_ridge']:.2f}  "
            f"ρ={r['spearman_rho']:+.2f}",
            fontsize=9,
        )
        plt.colorbar(sc, ax=ax, label="Reign midpoint" if col == 2 else "",
                     fraction=0.03, pad=0.02)

    fig.suptitle(
        f"{model_name} — temporal geometry of English monarchs\n"
        f"Embeddings = mean CLS over {sum(n_sents)//len(n_sents)} sentences per monarch "
        f"(named + era-describing)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out_png = OUT_DIR / f"temporal_geometry_{model_key}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {out_png}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    best_r2  = df.loc[df["r2_ridge"].idxmax()]
    best_rho = df.loc[df["spearman_rho"].abs().idxmax()]
    best_nn  = df.loc[df["nn_coherence"].idxmax()]
    print(f"\n=== Summary ===")
    print(f"  Best linear R²:           L{int(best_r2['layer']):02d}  R²={best_r2['r2_ridge']:.3f}")
    print(f"  Best Spearman ρ:          L{int(best_rho['layer']):02d}  ρ={best_rho['spearman_rho']:+.3f}")
    print(f"  Best NN temporal coh.:    L{int(best_nn['layer']):02d}  {best_nn['nn_coherence']:.2%}")
    print(f"  Dims needed for 90% var:  "
          f"{df['n_dims_90pct'].min()}–{df['n_dims_90pct'].max()} across layers")
    print(f"\n  Interpretation:")
    max_r2 = df["r2_ridge"].max()
    max_rho = df["spearman_rho"].abs().max()
    if max_r2 > 0.4:
        print(f"    ✓ Strong linear temporal signal (R²={max_r2:.2f})")
    elif max_r2 > 0.1:
        print(f"    ~ Weak linear temporal signal (R²={max_r2:.2f})")
    else:
        print(f"    ✗ No linear temporal signal (R²={max_r2:.2f})")

    if max_rho > 0.5:
        print(f"    ✓ Temporal ordering partially preserved in PC1 (ρ={max_rho:.2f})")
    else:
        print(f"    ✗ PC1 does not track time (ρ={max_rho:.2f})")

    nn_max = df["nn_coherence"].max()
    random_baseline = min(5, len(MONARCHS)) / len(MONARCHS)
    if nn_max > random_baseline * 1.5:
        print(f"    ✓ Temporally coherent neighbourhoods "
              f"({nn_max:.0%} vs {random_baseline:.0%} random)")
    else:
        print(f"    ✗ No temporal neighbourhood structure "
              f"({nn_max:.0%} vs {random_baseline:.0%} random)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal geometry probe: does BERT know historical time?"
    )
    parser.add_argument("--model", default="bert",
                        choices=["bert", "macberth"],
                        help="Model to probe (default: bert).")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[0, 3, 7, 11],
        help="0-indexed layers to show PCA projections for (default: 0 3 7 11).",
    )
    args = parser.parse_args()
    run(model_key=args.model, layers_to_show=args.layers)


if __name__ == "__main__":
    main()
