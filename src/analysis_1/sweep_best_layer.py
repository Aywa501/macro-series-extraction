"""sweep_best_layer.py — Find the layer with the best value-axis quality for a model.

Loads the model once, sweeps all 12 layers, prints a table of Pearson r
(value axis vs log pence) and Spearman ρ (time axis vs year), and prints
the best layer for value encoding.

Usage
-----
    python src/analysis/sweep_best_layer.py --model macberth
    python src/analysis/sweep_best_layer.py --model bert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _common import (
    DENOMINATIONS, VALUE_TEMPLATES, YEAR_TEMPLATES, YEAR_PROBE_YEARS,
)

MODEL_NAMES = {
    "bert":     "bert-base-uncased",
    "macberth": "emanjavacas/MacBERTh",
}


def cls_at_layer(model, tokenizer, device, sentence: str, layer: int) -> np.ndarray:
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.hidden_states[layer][0, 0, :].cpu().numpy()


def embed_all_layers(model, tokenizer, device, sentences: list[str]) -> np.ndarray:
    """Embed a list of sentences, return mean CLS per layer. Shape: (n_layers, 768)."""
    n_layers = model.config.num_hidden_layers
    layer_vecs = [[] for _ in range(n_layers)]
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        for i in range(n_layers):
            layer_vecs[i].append(out.hidden_states[i + 1][0, 0, :].cpu().numpy())
    return np.stack([np.mean(vecs, axis=0) for vecs in layer_vecs])  # (n_layers, 768)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "macberth"], default="macberth")
    args = parser.parse_args()
    model_key = args.model

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    name   = MODEL_NAMES[model_key]
    print(f"Loading {name} on {device} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    n_layers  = model.config.num_hidden_layers
    print(f"Loaded. {n_layers} layers.\n")

    # --- Embed denominations (all templates, all layers) ---
    print("Embedding denominations across all layers …", flush=True)
    # denom_embs[layer] -> dict name -> (768,)
    denom_layer_vecs: list[dict[str, np.ndarray]] = [{} for _ in range(n_layers)]
    for name_c, _ in DENOMINATIONS:
        sents = [tmpl.format(c=name_c) for tmpl in VALUE_TEMPLATES]
        layer_means = embed_all_layers(model, tokenizer, device, sents)  # (n_layers, 768)
        for i in range(n_layers):
            denom_layer_vecs[i][name_c] = layer_means[i]

    # --- Embed year-probe sentences (all templates, all layers) ---
    print("Embedding year-probe sentences across all layers …", flush=True)
    # year_layer_vecs[layer] -> (n_years, 768)
    year_layer_vecs: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    years_arr = np.array(YEAR_PROBE_YEARS, dtype=float)
    for y in YEAR_PROBE_YEARS:
        sents = [tmpl.format(y=y) for tmpl in YEAR_TEMPLATES]
        layer_means = embed_all_layers(model, tokenizer, device, sents)  # (n_layers, 768)
        for i in range(n_layers):
            year_layer_vecs[i].append(layer_means[i])
    year_layer_mats = [np.stack(vecs) for vecs in year_layer_vecs]  # list of (n_years, 768)

    # --- Compute metrics per layer ---
    pences   = np.array([p for _, p in DENOMINATIONS], dtype=float)
    log_p    = np.log(pences)
    names_c  = [n for n, _ in DENOMINATIONS]

    print()
    print("=" * 65)
    print(f"LAYER SWEEP — {MODEL_NAMES[model_key]}")
    print(f"{'Layer':>6}  {'rho_value':>10}  {'p_value':>10}  {'rho_time':>9}  {'p_time':>10}")
    print("-" * 65)

    results = []
    for i in range(n_layers):
        centroids = denom_layer_vecs[i]
        year_mat  = year_layer_mats[i]

        # Value axis — OLS regression onto log(pence), same as 01_build_axes.py
        X   = np.stack([centroids[n] for n in names_c])   # (n_coins, 768)
        X_c = X - X.mean(axis=0)
        y_c = log_p - log_p.mean()
        val_dir, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)
        val_dir /= (np.linalg.norm(val_dir) + 1e-12)
        projs_val = np.array([np.dot(centroids[n], val_dir) for n in names_c])
        rho_val, p_val = spearmanr(projs_val, pences)

        # Time axis — early vs late mean difference
        early_mask = years_arr <= 1450
        late_mask  = years_arr >= 1750
        t_dir      = year_mat[late_mask].mean(0) - year_mat[early_mask].mean(0)
        t_dir     /= (np.linalg.norm(t_dir) + 1e-12)
        projs_t    = year_mat @ t_dir
        rho_t, p_t = spearmanr(projs_t, years_arr)

        results.append({
            "layer":     i + 1,
            "rho_value": float(rho_val),
            "p_value":   float(p_val),
            "rho_time":  float(rho_t),
            "p_time":    float(p_t),
            "val_dir":   val_dir,
            "t_dir":     t_dir,
        })
        print(f"  L{i+1:02d}  {rho_val:>+10.4f}  {p_val:>10.3e}  {rho_t:>+9.4f}  {p_t:>10.3e}")

    print("=" * 65)

    # rho_value is always 1.0 (OLS underdetermined system fits perfectly),
    # so select by rho_time — the genuinely informative metric.
    best = max(results, key=lambda x: x["rho_time"])
    print(f"\n  Best layer (by rho_time): L{best['layer']:02d}  "
          f"rho_value={best['rho_value']:+.4f}  rho_time={best['rho_time']:+.4f}")

    # Save directions for best layer
    out_dir = PROJECT_ROOT / "data" / model_key / "value_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = best["layer"]
    np.save(out_dir / f"value_direction_L{layer}.npy", best["val_dir"])
    np.save(out_dir / f"time_direction_L{layer}.npy",  best["t_dir"])
    print(f"  Saved value_direction_L{layer}.npy")
    print(f"  Saved time_direction_L{layer}.npy")
    print(f"\n  → Now run: python3 src/analysis/03_coin_value_probe.py "
          f"--model {model_key} --layer {layer}")


if __name__ == "__main__":
    main()
