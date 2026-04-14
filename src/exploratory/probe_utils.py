"""probe_utils.py — Shared utilities for temporal probe scripts.

Imported by:
    src/exploratory/sequences/sequence_comprehensive.py
    src/exploratory/temporal/temporal_manifold.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


MODEL_NAMES = {
    "bert":     "bert-base-uncased",
    "macberth": "emanjavacas/MacBERTh",
    "sikubert": "SIKU-BERT/sikubert",
    "openai":   "text-embedding-3-small",
}


def load_model(model_key: str):
    name = MODEL_NAMES[model_key]
    print(f"Loading {name} …", flush=True)
    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModel.from_pretrained(
        name, output_hidden_states=True).to(device).eval()
    n = model.config.num_hidden_layers
    print(f"Loaded. {n+1} layers (0=embed, 1–{n}=transformer).\n")
    return model, tokenizer, device


def embed_all_layers(model, tokenizer, device, text: str) -> np.ndarray:
    """Single forward pass → CLS at every layer. Returns (n_layers, hidden)."""
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=64).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return np.stack([h[0, 0, :].cpu().numpy() for h in out.hidden_states])


def bw_ratio(Xs: np.ndarray, pids: np.ndarray) -> float:
    """Between/within variance ratio."""
    unique = np.unique(pids)
    if len(unique) < 2:
        return 0.0
    grand   = Xs.mean(axis=0)
    between = sum((pids == u).sum() * np.sum((Xs[pids == u].mean(0) - grand) ** 2)
                  for u in unique)
    within  = sum(np.sum((Xs[pids == u] - Xs[pids == u].mean(0)) ** 2)
                  for u in unique)
    return float(between / (within + 1e-12))


def centroid_tau(centroids: np.ndarray, mids: np.ndarray) -> float:
    """Kendall τ between pairwise embedding distance and temporal distance.

    Args:
        centroids: (N, H) — one embedding per period
        mids:      (N,)  — mid-year per period
    """
    n = len(centroids)
    if n < 3:
        return float("nan")
    ed, td = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ed.append(np.linalg.norm(centroids[i] - centroids[j]))
            td.append(abs(float(mids[i] - mids[j])))
    tau, _ = stats.kendalltau(ed, td)
    return float(tau)


def knn_purity(Xs: np.ndarray, pids: np.ndarray) -> float:
    """Fraction of points whose nearest neighbour (excluding same-period)
    is a temporally adjacent period (|pid_nn - pid_own| == 1)."""
    correct = 0
    for i in range(len(Xs)):
        own  = pids[i]
        mask = pids != own
        if not mask.any():
            continue
        nn = pids[mask][np.argmin(np.linalg.norm(Xs[mask] - Xs[i], axis=1))]
        if abs(nn - own) == 1:
            correct += 1
    return correct / len(Xs)


def velocity(embs: np.ndarray, years: np.ndarray, min_dt: float = 2.0) -> np.ndarray:
    """‖Δemb‖ / Δyear for consecutive pairs → (n-1,). NaN where Δt < min_dt."""
    deltas = np.diff(embs, axis=0)
    dt     = np.diff(years)
    norms  = np.linalg.norm(deltas, axis=1)
    return np.where(np.abs(dt) >= min_dt, norms / np.abs(dt), np.nan)


def curvature(embs: np.ndarray) -> np.ndarray:
    """Direction-change angle (degrees) at each interior point → (n-2,)."""
    d1  = np.diff(embs[:-1], axis=0)
    d2  = np.diff(embs[1:],  axis=0)
    cos = np.einsum("ij,ij->i", d1, d2) / (
          np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def local_dim(embs: np.ndarray, window: int = 9) -> np.ndarray:
    """Participation ratio in a sliding window → (n,), NaN at edges."""
    n  = len(embs)
    pr = np.full(n, np.nan)
    hw = window // 2
    for i in range(hw, n - hw):
        X   = embs[i - hw : i + hw + 1]
        Xc  = X - X.mean(0)
        sv  = np.linalg.svd(Xc, compute_uv=False)
        lam = sv ** 2
        lam = lam[lam > 1e-12]
        pr[i] = (lam.sum() ** 2) / (lam ** 2).sum() if len(lam) else 1.0
    return pr


def load_embedder(model_key: str):
    """Load model and return (embed_fn, n_layers).

    embed_fn : str → (n_layers, H)  np.ndarray
    n_layers : int — 13 for BERT-base variants, 1 for API models (OpenAI)

    HuggingFace models use a single forward pass that returns all hidden states.
    OpenAI uses the embeddings API; requires OPENAI_API_KEY in the environment.
    """
    if model_key == "openai":
        from openai import OpenAI  # type: ignore
        client   = OpenAI()
        oai_name = MODEL_NAMES["openai"]
        print(f"Using OpenAI API: {oai_name}")

        def embed_fn(text: str) -> np.ndarray:
            r = client.embeddings.create(input=text, model=oai_name)
            vec = np.array(r.data[0].embedding, dtype=np.float32)
            return vec[np.newaxis, :]   # (1, H)

        return embed_fn, 1

    model, tokenizer, device = load_model(model_key)
    n_layers = model.config.num_hidden_layers + 1

    def embed_fn(text: str) -> np.ndarray:
        return embed_all_layers(model, tokenizer, device, text)

    return embed_fn, n_layers
