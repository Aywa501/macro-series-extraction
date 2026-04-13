"""Appendix — Multi-sequence temporal geometry probe (standalone).

Asks: across multiple historical sequences with *known transition years*,
does BERT's representational geometry encode discrete categorical change?
Which sequences produce the clearest temporal structure?

Method
------
For each sequence we define a set of (label, year_range) pairs and carrier
sentence templates.  We embed each sentence at every BERT layer, then for
each layer compute four complementary metrics:

  1. **Spearman ρ** (PC1 vs mid-year): linear temporal ordering in the
     leading principal component.  Informative only for *monotonic* sequences.
     Non-monotonic sequences (monarchy→commonwealth→monarchy) will show ρ≈0
     even if BERT knows the categories perfectly.

  2. **Between/Within variance ratio (BW)**: ratio of between-cluster
     variance to within-cluster variance.  Measures categorical separability
     independently of temporal order.  High BW = tight clusters regardless
     of ordering.  The right metric for non-monotonic sequences.

  3. **Centroid Kendall τ**: pairwise centroid distances ranked against
     expected temporal distances.  Measures whether *more temporally distant*
     periods are *further apart* in embedding space — a richer test than ρ
     because it captures non-linear ordering.

  4. **Cross-period kNN purity**: for each sentence, is its nearest
     neighbour from a *different period* from the temporally closest period?
     Unlike within-period kNN (trivially 1.0 when same-label sentences share
     identical wording), this asks: do BERT's representations confuse distant
     or adjacent periods?

  Ridge R² is also computed and reported *without clipping* — negative values
  indicate the year cannot be recovered from the full embedding at all, which
  is important diagnostic information.

Sequences included
------------------
  STRICT BOUNDARIES (legislatively precise):
  1. Country name         1707 (Kingdom of Great Britain), 1801 (United Kingdom)
  2. State religion       1534 (Church of England), 1553 (Catholic restoration),
                          1558 (Church of England restored)
  3. Form of government   1649 (Commonwealth), 1660 (Restoration monarchy)
  4. Ruling dynasty       exact accession years
  5. Calendar system      1752 (Julian → Gregorian)

  APPROXIMATE BOUNDARIES (technological/material change):
  6. Primary weapon       longbow, pike, musket, rifle
  7. Ship construction    wood, iron, steel
  8. Primary fuel         wood/peat, coal, steam coal

No Penn corpus. No Ridge direction. Pure pretrained BERT weights.

Outputs
-------
  src/appendix/sequence_geometry_{model}.csv
  src/appendix/sequence_geometry_{model}.png
  Printed comparison table with diagnostic commentary

Usage
-----
    python src/appendix/sequence_geometry_probe.py
    python src/appendix/sequence_geometry_probe.py --model macberth
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Sequence definitions
# ---------------------------------------------------------------------------

class Period(NamedTuple):
    label: str
    start: int
    end:   int


def _sample_years(start: int, end: int, n: int = 6) -> list[int]:
    if end - start < n:
        return list(range(start, end + 1))
    return [int(round(start + i * (end - start) / (n - 1))) for i in range(n)]


SEQUENCES: dict[str, dict] = {

    # ------------------------------------------------------------------
    # STRICT year boundaries
    # ------------------------------------------------------------------

    "country_name": {
        "title": "Country name",
        "subtitle": "strict: 1707, 1801",
        "boundary_type": "strict",
        "transitions": [1707, 1801],
        "monotonic": True,
        "periods": [
            Period("Kingdom of England",        1600, 1706),
            Period("Kingdom of Great Britain",  1707, 1800),
            Period("United Kingdom",            1801, 1900),
        ],
        "templates": [
            "the country was known as {label} .",
            "england was then called {label} .",
            "the nation of {label} was ruled by a monarch .",
            "the official name of the realm was {label} .",
        ],
    },

    "state_religion": {
        "title": "State religion",
        "subtitle": "strict: 1534, 1553, 1558",
        "boundary_type": "strict",
        "monotonic": False,   # Catholic → C.of.E. → Catholic → C.of.E. — repeating
        "note": "non-monotonic (alternates): ρ uninformative; use BW + confusion",
        "transitions": [1534, 1553, 1558],
        "periods": [
            Period("Roman Catholic",    1400, 1533),
            Period("Church of England", 1534, 1552),
            Period("Roman Catholic",    1553, 1557),
            Period("Church of England", 1558, 1700),
        ],
        "templates": [
            "the religion of the land was {label} .",
            "the state religion was {label} .",
            "the church followed {label} teachings .",
            "worship in england followed {label} traditions .",
        ],
    },

    "form_of_government": {
        "title": "Form of government",
        "subtitle": "strict: 1649, 1660",
        "boundary_type": "strict",
        "monotonic": False,   # monarchy → commonwealth → monarchy — repeating
        "note": "non-monotonic (bracket): ρ uninformative; use BW + confusion",
        "transitions": [1649, 1660],
        "periods": [
            Period("monarchy",      1500, 1648),
            Period("commonwealth",  1649, 1659),
            Period("monarchy",      1660, 1800),
        ],
        "templates": [
            "england was governed as a {label} .",
            "the system of rule was a {label} .",
            "the form of government was {label} .",
            "power was held through a {label} .",
        ],
    },

    "ruling_dynasty": {
        "title": "Ruling dynasty",
        "subtitle": "strict: exact accession years",
        "boundary_type": "strict",
        "monotonic": True,
        "transitions": [1399, 1461, 1485, 1603, 1714, 1837, 1901],
        "periods": [
            Period("Plantagenet",  1200, 1398),
            Period("Lancaster",    1399, 1460),
            Period("York",         1461, 1484),
            Period("Tudor",        1485, 1602),
            Period("Stuart",       1603, 1713),
            Period("Hanover",      1714, 1836),
            Period("Windsor",      1837, 1936),
        ],
        "templates": [
            "the {label} dynasty ruled england .",
            "the house of {label} held the throne .",
            "england was ruled by the {label} family .",
            "the {label} monarch sat on the english throne .",
        ],
    },

    "calendar": {
        "title": "Calendar system",
        "subtitle": "strict: 1752",
        "boundary_type": "strict",
        "monotonic": True,
        "transitions": [1752],
        "periods": [
            Period("Julian calendar",     1500, 1751),
            Period("Gregorian calendar",  1752, 1900),
        ],
        "templates": [
            "dates were recorded in the {label} .",
            "the {label} was used to track time .",
            "england used the {label} for official records .",
            "the year was measured by the {label} .",
        ],
    },

    # ------------------------------------------------------------------
    # APPROXIMATE year boundaries (technological change)
    # ------------------------------------------------------------------

    "primary_weapon": {
        "title": "Primary weapon",
        "subtitle": "approx: ~1500, ~1650, ~1850",
        "boundary_type": "approximate",
        "monotonic": True,
        "transitions": [1500, 1650, 1850],
        "periods": [
            Period("longbow",   1200, 1499),
            Period("pike",      1500, 1649),
            Period("musket",    1650, 1849),
            Period("rifle",     1850, 1930),
        ],
        "templates": [
            "the soldier carried a {label} into battle .",
            "troops were armed with the {label} .",
            "the primary weapon of the infantry was the {label} .",
            "soldiers fought with the {label} .",
        ],
    },

    "ship_construction": {
        "title": "Ship construction",
        "subtitle": "approx: ~1820, ~1870",
        "boundary_type": "approximate",
        "monotonic": True,
        "transitions": [1820, 1870],
        "periods": [
            Period("wooden ship",   1200, 1819),
            Period("iron ship",     1820, 1869),
            Period("steel ship",    1870, 1930),
        ],
        "templates": [
            "the navy sailed in a {label} .",
            "the vessel was a {label} .",
            "the fleet consisted of {label}s .",
            "the warship was a {label} .",
        ],
    },

    "primary_fuel": {
        "title": "Primary fuel",
        "subtitle": "approx: ~1700, ~1820",
        "boundary_type": "approximate",
        "monotonic": True,
        "transitions": [1700, 1820],
        "periods": [
            Period("wood and peat",  1200, 1699),
            Period("coal",           1700, 1819),
            Period("steam coal",     1820, 1930),
        ],
        "templates": [
            "homes were heated with {label} .",
            "the furnace burned {label} .",
            "industry relied on {label} for energy .",
            "heat was produced by burning {label} .",
        ],
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_key: str):
    from transformers import AutoModel, AutoTokenizer
    if model_key == "bert":
        name = "bert-base-uncased"
    elif model_key == "macberth":
        name = "emanjavacas/MacBERTh"
    else:
        name = model_key
    print(f"Loading {name} …", flush=True)
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    print("Loaded.\n")
    return model, tokenizer, device, name


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_sentence(model, tokenizer, device, sentence: str) -> np.ndarray:
    """CLS embedding at all layers: (n_layers+1, hidden_dim)."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    hidden = out.hidden_states  # tuple len = 13 for bert-base
    cls    = np.stack([h[0, 0, :].cpu().numpy() for h in hidden], axis=0)
    return cls


def build_dataset(seq_def: dict, model, tokenizer, device,
                  n_years_per_period: int = 5):
    """
    Returns:
        embeddings : (N, n_layers, hidden_dim)
        years      : (N,)  representative mid-year per sample
        labels     : (N,)  label string
        label_ids  : (N,)  integer id (in period order)
        period_ids : (N,)  period index (allows repeated labels like 'monarchy')
        label_set  : list of unique labels (ordered by first occurrence)
        period_list: list of Period objects in order
    """
    periods    = seq_def["periods"]
    templates  = seq_def["templates"]
    label_set  = list(dict.fromkeys(p.label for p in periods))

    all_embs, all_years, all_labels, all_label_ids, all_period_ids = (
        [], [], [], [], []
    )

    for p_idx, period in enumerate(periods):
        years   = _sample_years(period.start, period.end, n_years_per_period)
        label_id = label_set.index(period.label)
        for year in years:
            for tmpl in templates:
                sentence = tmpl.format(label=period.label)
                emb      = embed_sentence(model, tokenizer, device, sentence)
                all_embs.append(emb)
                all_years.append((period.start + period.end) / 2)  # centroid year
                all_labels.append(period.label)
                all_label_ids.append(label_id)
                all_period_ids.append(p_idx)

    embeddings  = np.stack(all_embs, axis=0)
    years       = np.array(all_years, dtype=float)
    labels      = np.array(all_labels)
    label_ids   = np.array(all_label_ids, dtype=int)
    period_ids  = np.array(all_period_ids, dtype=int)

    return embeddings, years, labels, label_ids, period_ids, label_set, periods


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def bw_ratio(X: np.ndarray, period_ids: np.ndarray) -> float:
    """Between/within variance ratio using period (not label) ids.
    Handles repeated labels (monarchy appears twice) correctly."""
    unique = np.unique(period_ids)
    if len(unique) < 2:
        return 0.0
    grand_mean = X.mean(axis=0)
    between    = sum(
        (period_ids == u).sum() *
        np.sum((X[period_ids == u].mean(axis=0) - grand_mean) ** 2)
        for u in unique
    )
    within     = sum(
        np.sum((X[period_ids == u] - X[period_ids == u].mean(axis=0)) ** 2)
        for u in unique
    )
    return float(between / (within + 1e-12))


def centroid_kendall_tau(X: np.ndarray, period_ids: np.ndarray,
                         periods: list) -> float:
    """Kendall τ between pairwise centroid Euclidean distances and
    expected temporal distances (|mid_year_i - mid_year_j|).

    A positive τ means temporally closer periods are also closer in
    embedding space — correct temporal geometry even for non-monotonic
    sequences.
    """
    unique = np.unique(period_ids)
    if len(unique) < 3:
        return float("nan")

    centroids  = np.stack([X[period_ids == u].mean(axis=0) for u in unique])
    mid_years  = np.array(
        [(periods[u].start + periods[u].end) / 2 for u in unique])

    n  = len(unique)
    emb_dists, time_dists = [], []
    for i in range(n):
        for j in range(i + 1, n):
            emb_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
            time_dists.append(abs(mid_years[i] - mid_years[j]))

    if len(emb_dists) < 3:
        return float("nan")
    tau, _ = stats.kendalltau(emb_dists, time_dists)
    return float(tau)


def cross_period_knn_purity(X: np.ndarray, period_ids: np.ndarray,
                             periods: list) -> float:
    """For each sentence, find its nearest neighbour from a DIFFERENT period.
    Is that neighbour's period the temporally adjacent one?

    Returns the fraction of sentences whose closest cross-period neighbour
    comes from a chronologically adjacent period (|Δperiod| = 1).
    """
    n      = len(X)
    n_per  = int(period_ids.max()) + 1
    correct = 0

    for i in range(n):
        own_period = period_ids[i]
        # Mask out own-period samples
        mask  = period_ids != own_period
        if not mask.any():
            continue
        dists = np.linalg.norm(X[mask] - X[i], axis=1)
        nn_period = period_ids[mask][np.argmin(dists)]
        if abs(nn_period - own_period) == 1:
            correct += 1

    return correct / n


def ridge_r2_raw(X: np.ndarray, years: np.ndarray) -> float:
    """Cross-validated Ridge R² for predicting mid-year from full embedding.
    NOT clipped — negative values are meaningful (model fails to predict year).
    """
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    ridge  = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    scores = cross_val_score(ridge, Xs, years, cv=min(5, len(np.unique(years))),
                             scoring="r2")
    return float(np.mean(scores))


def analyse_layer(X: np.ndarray, years: np.ndarray, labels: np.ndarray,
                  period_ids: np.ndarray, periods: list) -> dict:
    """All geometry metrics for one layer's CLS embeddings."""
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # Spearman ρ (PC1 vs year)
    pca    = PCA(n_components=1)
    pc1    = pca.fit_transform(Xs)[:, 0]
    rho, _ = stats.spearmanr(pc1, years)

    # Between/within (period-aware)
    bw     = bw_ratio(Xs, period_ids)

    # Centroid Kendall τ
    tau    = centroid_kendall_tau(Xs, period_ids, periods)

    # Cross-period kNN purity
    knn_xp = cross_period_knn_purity(Xs, period_ids, periods)

    # Ridge R² (raw, not clipped)
    r2     = ridge_r2_raw(X, years)

    # Intrinsic dimensionality (PCs for 90% variance)
    full_pca = PCA().fit(Xs)
    cum_var  = np.cumsum(full_pca.explained_variance_ratio_)
    n_dim_90 = int(np.searchsorted(cum_var, 0.90)) + 1

    return {
        "spearman_rho":    float(rho),
        "bw_ratio":        bw,
        "centroid_tau":    tau,
        "knn_xperiod":     knn_xp,
        "ridge_r2":        r2,
        "n_dim_90":        n_dim_90,
    }


# ---------------------------------------------------------------------------
# Period confusion matrix (best layer)
# ---------------------------------------------------------------------------

def period_confusion(X: np.ndarray, period_ids: np.ndarray,
                     periods: list) -> np.ndarray:
    """N_periods × N_periods confusion matrix.
    Entry (i,j) = fraction of period-i sentences whose nearest cross-period
    neighbour is from period j.
    """
    n_per  = int(period_ids.max()) + 1
    conf   = np.zeros((n_per, n_per))
    for i_sample in range(len(X)):
        own  = period_ids[i_sample]
        mask = period_ids != own
        if not mask.any():
            continue
        dists  = np.linalg.norm(X[mask] - X[i_sample], axis=1)
        nn_per = period_ids[mask][np.argmin(dists)]
        conf[own, nn_per] += 1
    # Normalise each row by row sum
    row_sums = conf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return conf / row_sums


# ---------------------------------------------------------------------------
# Main probe loop
# ---------------------------------------------------------------------------

def run_sequence(seq_name: str, seq_def: dict, model, tokenizer,
                 device) -> tuple[pd.DataFrame, dict]:
    print(f"  {seq_def['title']} ({seq_def['subtitle']}) …", flush=True)

    embs, years, labels, label_ids, period_ids, label_set, periods = \
        build_dataset(seq_def, model, tokenizer, device)

    n_layers = embs.shape[1]
    rows = []
    for layer in range(n_layers):
        X   = embs[:, layer, :]
        met = analyse_layer(X, years, labels, period_ids, periods)
        rows.append({"sequence": seq_name, "layer": layer, **met})

    df = pd.DataFrame(rows)

    # Compute confusion matrix at layer with highest BW ratio
    best_layer = int(df["bw_ratio"].idxmax() - df.index[0])
    X_best     = embs[:, best_layer, :]
    scaler     = StandardScaler()
    X_best_s   = scaler.fit_transform(X_best)
    conf       = period_confusion(X_best_s, period_ids, periods)

    aux = {
        "periods":    periods,
        "label_set":  label_set,
        "embeddings": embs,
        "period_ids": period_ids,
        "years":      years,
        "best_bw_layer": best_layer,
        "confusion": conf,
    }
    return df, aux


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(results: pd.DataFrame, sequences: dict,
                  aux_data: dict, model_name: str) -> None:
    print("\n" + "=" * 90)
    print(f"SEQUENCE GEOMETRY SUMMARY — {model_name}")
    print("=" * 90)

    metric_cols  = ["spearman_rho", "bw_ratio", "centroid_tau", "knn_xperiod", "ridge_r2"]
    metric_names = ["Spearman ρ", "BW ratio", "Kendall τ", "xPeriod kNN", "Ridge R²"]

    # Header
    print(f"\n  {'Sequence':<26} {'Type':>7}  "
          + "  ".join(f"{m:>14}" for m in metric_names))
    print("  " + "-" * 110)

    for seq_name, seq_def in sequences.items():
        sub    = results[results["sequence"] == seq_name]
        btype  = "strict" if seq_def["boundary_type"] == "strict" else "approx"
        title  = seq_def["title"][:25]

        # For each metric: best value and the layer it occurs at.
        # ridge_r2: take max (least negative = best).
        # centroid_tau: show signed (negative = anti-ordered, informative for non-mono).
        # others: take abs-max.
        vals = []
        for col in metric_cols:
            if sub[col].isna().all():
                vals.append("  N/A        ")
                continue
            if col == "ridge_r2":
                best_idx = sub[col].idxmax()
            elif col == "centroid_tau":
                best_idx = sub[col].abs().idxmax()
            else:
                best_idx = sub[col].abs().idxmax()
            best_val = float(sub.loc[best_idx, col])
            best_l   = int(sub.loc[best_idx, "layer"])
            vals.append(f"{best_val:+.3f} (L{best_l:02d})")

        print(f"  {title:<26} {btype:>7}  " + "  ".join(f"{v:>14}" for v in vals))

    # Monotonicity note
    print()
    print("  Note: ρ is uninformative for non-monotonic sequences "
          "(state_religion, form_of_government).")
    print("  For those, use BW ratio (cluster separability) + confusion matrix.")

    # Per-sequence diagnostic notes
    print("\n" + "=" * 90)
    print("PER-SEQUENCE DIAGNOSTICS")
    print("=" * 90)

    for seq_name, seq_def in sequences.items():
        sub       = results[results["sequence"] == seq_name]
        title     = seq_def["title"]
        note      = seq_def.get("note", "")
        monotonic = seq_def.get("monotonic", True)
        aux       = aux_data[seq_name]
        periods   = aux["periods"]
        conf      = aux["confusion"]
        best_l    = aux["best_bw_layer"]

        best_bw   = float(sub["bw_ratio"].max())
        best_bw_l = int(sub["bw_ratio"].idxmax() - sub.index[0])
        best_rho  = float(sub["spearman_rho"].abs().max())
        best_rho_l= int(sub["spearman_rho"].abs().idxmax() - sub.index[0])
        best_tau  = sub["centroid_tau"].dropna()
        best_tau_v= float(best_tau.abs().max()) if len(best_tau) else float("nan")
        best_knn  = float(sub["knn_xperiod"].max())

        print(f"\n  ── {title} ──")
        if note:
            print(f"     ⚠  {note}")
        print(f"     Cluster separability (BW):   {best_bw:.3f} at L{best_bw_l:02d}  "
              f"{'↑ strong' if best_bw > 0.5 else '↓ weak'}")
        if monotonic:
            print(f"     Temporal ordering (ρ):       {best_rho:.3f} at L{best_rho_l:02d}  "
                  f"{'↑ clear' if best_rho > 0.4 else '~ modest' if best_rho > 0.2 else '↓ poor'}")
            # τ: report signed value; negative means centroid distances ANTI-correlate
            # with temporal distance (nearby periods are FURTHER apart than distant ones)
            raw_tau_vals = sub["centroid_tau"].dropna()
            if len(raw_tau_vals):
                best_tau_signed_idx = raw_tau_vals.abs().idxmax()
                best_tau_signed = float(raw_tau_vals.loc[best_tau_signed_idx])
                best_tau_l      = int(sub.loc[best_tau_signed_idx, "layer"])
                tau_interp = ("↑ correct ordering" if best_tau_signed > 0.3
                              else "↓ anti-ordered (close periods are far apart)"
                              if best_tau_signed < -0.3 else "~ weak")
                print(f"     Centroid ordering (τ):       "
                      f"{best_tau_signed:+.3f} at L{best_tau_l:02d}  {tau_interp}")
            else:
                print(f"     Centroid ordering (τ):       N/A (too few periods)")
        print(f"     Cross-period kNN purity:     {best_knn:.3f}  "
              f"(random baseline ≈ {1/max(len(periods)-1,1):.3f})")

        # Confusion matrix (period labels as rows/cols)
        p_labels = [p.label[:14] for p in periods]
        print(f"     Confusion at L{best_l:02d} "
              f"(row=true period, col=nearest cross-period neighbour):")
        header = "       " + " ".join(f"{l:>14}" for l in p_labels)
        print(header)
        for i, row_label in enumerate(p_labels):
            row_str = " ".join(f"{conf[i,j]:>14.2f}" for j in range(len(p_labels)))
            print(f"       {row_label:>14}  {row_str}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(results: pd.DataFrame, sequences: dict,
               aux_data: dict, model_name: str, model_key: str) -> None:

    all_seqs     = list(sequences.keys())
    n_seqs       = len(all_seqs)
    strict_seqs  = [s for s in all_seqs if sequences[s]["boundary_type"] == "strict"]
    approx_seqs  = [s for s in all_seqs if sequences[s]["boundary_type"] == "approximate"]

    metrics       = ["spearman_rho", "bw_ratio", "centroid_tau", "knn_xperiod"]
    metric_labels = [
        "Spearman |ρ|  (temporal ordering via PC1)",
        "BW ratio  (between / within cluster variance)",
        "Centroid Kendall τ  (pairwise temporal distance)",
        "Cross-period kNN purity  (adjacent period = nearest foreign neighbour)",
    ]

    n_layers = int(results["layer"].max()) + 1
    layers   = list(range(n_layers))

    cmap_s = plt.cm.tab10
    cmap_a = plt.cm.Set2

    # ── Main figure: 4 metric panels ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax_i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_i]
        for j, seq in enumerate(strict_seqs):
            sub    = results[results["sequence"] == seq]
            vals   = []
            for l in layers:
                row = sub[sub["layer"] == l]
                v   = float(row[metric].iloc[0]) if len(row) else np.nan
                vals.append(abs(v) if metric in ("spearman_rho", "centroid_tau") else v)
            title  = sequences[seq]["title"]
            mono   = sequences[seq].get("monotonic", True)
            ls     = "-o" if mono else "--^"
            color  = cmap_s(j / max(len(strict_seqs) - 1, 1))
            label  = f"[S] {title}" + ("" if mono else "  ⚠non-mono")
            ax.plot(layers, vals, ls, markersize=4, color=color,
                    label=label, linewidth=1.8)

        for j, seq in enumerate(approx_seqs):
            sub    = results[results["sequence"] == seq]
            vals   = []
            for l in layers:
                row = sub[sub["layer"] == l]
                v   = float(row[metric].iloc[0]) if len(row) else np.nan
                vals.append(abs(v) if metric in ("spearman_rho", "centroid_tau") else v)
            title  = sequences[seq]["title"]
            color  = cmap_a(j / max(len(approx_seqs) - 1, 1))
            ax.plot(layers, vals, ":s", markersize=4, color=color,
                    label=f"[A] {title}", linewidth=1.5, alpha=0.85)

        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_xlabel("BERT layer", fontsize=9)
        ax.set_xticks(layers)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.axvspan(3.5, 4.5, alpha=0.07, color="gold", label="L04 zone")
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")

    fig.suptitle(
        f"{model_name} — temporal geometry across historical sequences\n"
        "[S]=strict legislated boundaries  [A]=approx. technological change  "
        "⚠=non-monotonic (use BW/confusion, not ρ)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    main_path = OUT_DIR / f"sequence_geometry_{model_key}.png"
    fig.savefig(main_path, dpi=150)
    plt.close(fig)
    print(f"Main plot → {main_path}")

    # ── Confusion matrices at best-BW layer ─────────────────────────────────
    # Layout: ruling dynasty gets a double-wide cell; others share a row of 4
    # We render them in two rows: top row = 4 strict seqs, bottom = 4 approx seqs
    # but give ruling_dynasty a bigger subplot.
    _conf_order = [
        ["country_name", "state_religion", "form_of_government", "ruling_dynasty"],
        ["calendar", "primary_weapon", "ship_construction", "primary_fuel"],
    ]
    # Width ratios: ruling_dynasty gets 2x; others 1x
    width_ratios_top = [1, 1, 1, 2]
    width_ratios_bot = [1, 1, 1, 1]

    fig2 = plt.figure(figsize=(26, 11))
    gs_top = fig2.add_gridspec(1, 4, left=0.04, right=0.98, top=0.92, bottom=0.52,
                               width_ratios=width_ratios_top, wspace=0.4)
    gs_bot = fig2.add_gridspec(1, 4, left=0.04, right=0.98, top=0.45, bottom=0.08,
                               width_ratios=width_ratios_bot, wspace=0.4)

    # Interpretation blurb per sequence
    _interp = {
        "country_name":       "GB correctly neighbours\nboth England and UK",
        "state_religion":     "Each faith-period maps to\nits twin, not the adjacent one\n→ BERT knows WHAT, not WHEN",
        "form_of_government": "Both monarchy periods\nmap to each other, not\nto commonwealth\n→ pre/post interlude merged",
        "ruling_dynasty":     "Tudor↔Stuart, Lancaster↔York\n(historical neighbours confused)\nPlantagenet→Stuart: distant\ndynasty confused with prototype",
        "calendar":           "Perfect adjacency\n(only 2 periods)",
        "primary_weapon":     "At L01: pike→rifle (wrong)\nAt L04: ordering improves\n→ word-level ≠ temporal knowledge",
        "ship_construction":  "Near-diagonal: wood↔iron\nand iron↔steel confused\n(correct adjacency pattern)",
        "primary_fuel":       "Clean chain: wood→coal\n→steam coal adjacency\nperfectly preserved",
    }

    all_axes = []
    for col_i, seq_name in enumerate(_conf_order[0]):
        ax = fig2.add_subplot(gs_top[col_i])
        all_axes.append((ax, seq_name))
    for col_i, seq_name in enumerate(_conf_order[1]):
        ax = fig2.add_subplot(gs_bot[col_i])
        all_axes.append((ax, seq_name))

    for ax, seq_name in all_axes:
        aux    = aux_data[seq_name]
        conf   = aux["confusion"]
        per    = aux["periods"]
        bl     = aux["best_bw_layer"]
        # Shorten labels but keep enough to be readable
        labels = []
        for p in per:
            lbl = p.label
            # Abbreviate only very long labels
            if len(lbl) > 14:
                lbl = lbl[:13] + "…"
            labels.append(lbl)

        im = ax.imshow(conf, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right",
                           fontsize=7 if len(per) <= 4 else 6)
        ax.set_yticklabels(labels, fontsize=7 if len(per) <= 4 else 6)
        bw_val = float(results[results["sequence"] == seq_name]["bw_ratio"].max())
        mono   = sequences[seq_name].get("monotonic", True)
        mono_s = "" if mono else " ⚠"
        ax.set_title(
            f"{sequences[seq_name]['title']}{mono_s}\n"
            f"conf @ L{bl:02d}  BW={bw_val:.2f}",
            fontsize=8, fontweight="bold",
        )
        # Annotate cells
        for i in range(len(per)):
            for j in range(len(per)):
                if i == j:
                    continue
                if conf[i, j] > 0.02:
                    ax.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.5 if len(per) <= 5 else 5.5,
                            color="white" if conf[i, j] > 0.55 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Interpretation text below the matrix
        interp_text = _interp.get(seq_name, "")
        ax.set_xlabel(interp_text, fontsize=6, color="#444444", labelpad=8)

    fig2.suptitle(
        f"{model_name} — period confusion matrices at best-BW layer\n"
        "Row = true period; column = nearest cross-period neighbour (fraction)  ·  "
        "Diagonal suppressed  ·  ⚠ = non-monotonic sequence",
        fontsize=10, fontweight="bold",
    )
    conf_path = OUT_DIR / f"sequence_confusion_{model_key}.png"
    fig2.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Confusion plot → {conf_path}")

    # ── PCA scatter at L04 for 4 key sequences ───────────────────────────────
    key_seqs = ["ruling_dynasty", "primary_weapon", "form_of_government", "state_religion"]
    fig3, axes3 = plt.subplots(1, 4, figsize=(22, 5))

    for ax_i, seq_name in enumerate(key_seqs):
        ax      = axes3[ax_i]
        aux     = aux_data[seq_name]
        embs    = aux["embeddings"]   # (N, n_layers, H)
        per_ids = aux["period_ids"]
        periods = aux["periods"]

        layer_idx = 4  # L04 is the key layer
        X = embs[:, layer_idx, :]
        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)
        pca    = PCA(n_components=2)
        coords = pca.fit_transform(Xs)

        cmap_p = plt.cm.viridis
        n_per  = int(per_ids.max()) + 1
        colours = [cmap_p(i / max(n_per - 1, 1)) for i in range(n_per)]

        for p_idx in range(n_per):
            mask = per_ids == p_idx
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=colours[p_idx], alpha=0.7, s=25,
                       label=periods[p_idx].label[:14], edgecolors="white",
                       linewidth=0.3)
            # Centroid marker
            cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
            ax.scatter(cx, cy, color=colours[p_idx], s=120,
                       marker="*", edgecolors="black", linewidth=0.8)

        ax.set_title(
            f"{sequences[seq_name]['title']}\n(PCA @ L04, "
            f"{'monotonic' if sequences[seq_name].get('monotonic') else 'non-monotonic'})",
            fontsize=8,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=7)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=7)
        ax.legend(fontsize=6, loc="best")
        ax.grid(linestyle=":", alpha=0.35)

    fig3.suptitle(
        f"{model_name} — PCA scatter at L04 for selected sequences\n"
        "Stars = period centroids. Colour = temporal order (early→late = dark→light)",
        fontsize=10, fontweight="bold",
    )
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    pca_path = OUT_DIR / f"sequence_pca_{model_key}.png"
    fig3.savefig(pca_path, dpi=150)
    plt.close(fig3)
    print(f"PCA scatter → {pca_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(model_key: str) -> None:
    model, tokenizer, device, model_name = load_model(model_key)

    all_dfs  = []
    aux_data = {}

    for seq_name, seq_def in SEQUENCES.items():
        df, aux = run_sequence(seq_name, seq_def, model, tokenizer, device)
        all_dfs.append(df)
        aux_data[seq_name] = aux

    results = pd.concat(all_dfs, ignore_index=True)

    csv_path = OUT_DIR / f"sequence_geometry_{model_key}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")

    print_summary(results, SEQUENCES, aux_data, model_name)
    make_plots(results, SEQUENCES, aux_data, model_name, model_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-sequence temporal geometry probe."
    )
    parser.add_argument("--model", default="bert",
                        choices=["bert", "macberth"],
                        help="Which model to probe (default: bert).")
    args = parser.parse_args()
    run(model_key=args.model)


if __name__ == "__main__":
    main()
