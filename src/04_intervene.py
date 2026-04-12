"""Stage 04 — Temporal intervention on idiom phrases (output + activation space).

For each treatment idiom phrase, applies temporal direction shifts across a
range of λ values (in actual years, derived from Stage 03 currency windows)
and records the resulting triviality score.  Two intervention types are run
from the same shared baseline embeddings:

  output     — geometric nudge of the final-layer CLS vector; no re-forward-
                pass through BERT.  Equivalent to asking: if we could directly
                move where this idiom sits in output space along the temporal
                axis, how does its triviality score change?

  activation — injection of λ × temporal_direction_k into the hidden state at
                the output of BERT layer k, which then propagates through the
                remaining layers normally.  Asks: at which layer does the
                temporal signal causally influence the triviality score?

Lambda range
------------
Stage 03 produces a currency_windows.csv with [lambda_min, lambda_max] per
currency.  We take the union across all currencies present in the idioms to get
a single [lam_min, lam_max] span, then step through it at 25-year intervals.
If currency_windows.csv is absent (e.g. Stage 03 not yet run), a fallback
range from the config is used.

Output schema
-------------
data/intervention/intervention_results.parquet
    intervention_type : "output" | "activation"
    idiom             : phrase string
    intervention_layer: int (1–12 for activation; NaN for output)
    lambda_years      : float — temporal shift in years
    score_paraphrase  : float — 2 × cos(embedding, triviality_axis)
    score_baseline    : float — score with no intervention

Usage
-----
    python src/04_intervene.py
    python src/04_intervene.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hooks import batch_inject_at_layer, embed_sentences

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)


def _load_treatment_idioms(idioms_yaml: Path) -> list[dict]:
    with open(idioms_yaml) as fh:
        cfg = yaml.safe_load(fh)
    return [
        e for e in cfg.get("idioms", [])
        if e.get("include", True)
        and e.get("group") == "treatment"
        and e.get("trivial_pole")
        and e.get("significant_pole")
    ]


def _build_lambda_range(cfg: dict, sent_path: Path) -> tuple[np.ndarray, int, int]:
    """Derive λ range from the Penn corpus year span (or config overrides).

    Returns (lambdas, penn_min, penn_max).  Override min/max by setting
    lambda.override_min / lambda.override_max in the config.
    """
    step     = cfg["lambda"]["default_step"]
    lam_cfg  = cfg["lambda"]
    df       = pd.read_parquet(sent_path, columns=["year"])
    penn_min = int(df["year"].min())
    penn_max = int(df["year"].max())
    lam_min  = int(lam_cfg.get("override_min", penn_min))
    lam_max  = int(lam_cfg.get("override_max", penn_max))
    lambdas  = np.arange(lam_min, lam_max + step / 2, step)
    logger.info(
        "λ range: [%d, %d] step %d  (%d points)%s",
        lam_min, lam_max, step, len(lambdas),
        " [OVERRIDE]" if "override_min" in lam_cfg or "override_max" in lam_cfg else "",
    )
    return lambdas, penn_min, penn_max


def _idiom_metadata(
    idiom: dict,
    cfg: dict,
    penn_min: int,
) -> tuple[str, int, int | None]:
    """Return (denominations_str, active_from, first_attested) for an idiom.

    active_from  — earliest year the coin existed, clamped to penn_min.
    first_attested — first documented appearance in writing, or None.
    """
    denom_windows = cfg.get("denomination_windows", {})
    attestations  = cfg.get("idiom_first_attestation", {})

    denoms     = idiom.get("denomination", [])
    denom_str  = ",".join(sorted(denoms))

    intro_years = [
        denom_windows.get(d, {}).get("introduction_year", penn_min)
        for d in denoms
    ]
    # active_from = earliest introduction among relevant denoms, but not
    # before the Penn corpus starts (regression is undefined outside that range)
    if intro_years:
        active_from = max(penn_min, min(intro_years))
    else:
        active_from = penn_min

    first_attested = attestations.get(idiom["phrase"])  # int or None

    return denom_str, active_from, first_attested


def run(cfg: dict, dry_run: bool) -> None:
    import torch
    from transformers import BertModel, BertTokenizerFast

    paths      = cfg["paths"]
    model_name = cfg["model"]["name"]
    batch_size = cfg["model"]["batch_size"]
    n_layers   = cfg["model"]["n_layers"]

    idioms_path    = PROJECT_ROOT / paths["idioms_config"]
    sent_path      = PROJECT_ROOT / paths["penn_sentences"]
    temp_dir_path  = PROJECT_ROOT / paths["temporal_directions"]
    norms_path     = PROJECT_ROOT / paths["temporal_coef_norms"]
    axes_path      = PROJECT_ROOT / paths["triviality_axes"]
    axes_idx_path  = PROJECT_ROOT / paths["triviality_axes_index"]
    pooled_path    = PROJECT_ROOT / paths["triviality_axis_pooled"]
    out_path       = PROJECT_ROOT / paths["intervention_results"]

    # --- Load idioms ---
    idioms = _load_treatment_idioms(idioms_path)
    if dry_run:
        idioms = idioms[:3]
        logger.info("[dry-run] Using first %d idioms", len(idioms))
    logger.info("Loaded %d treatment idioms", len(idioms))

    phrases  = [e["phrase"] for e in idioms]
    n_idioms = len(phrases)

    # --- Build λ range and gather idiom historical metadata ---
    lambdas, penn_min, penn_max = _build_lambda_range(cfg, sent_path)
    logger.info("λ values: %s … %s  (%d steps)", lambdas[0], lambdas[-1], len(lambdas))

    # Per-idiom historical provenance (stored as columns in output parquet)
    idiom_meta: dict[str, tuple[str, int, int | None]] = {
        e["phrase"]: _idiom_metadata(e, cfg, penn_min)
        for e in idioms
    }
    for phrase, (denoms, af, fa) in idiom_meta.items():
        logger.info(
            "  %-45s denoms=%-20s active_from=%d  first_attested=%s",
            phrase, denoms, af, fa,
        )

    # --- Load directions, norms, and triviality axes ---
    temp_dirs  = np.load(temp_dir_path).astype(np.float64)  # (12, 768)
    coef_norms = np.load(norms_path).astype(np.float64)      # (12,)
    axes       = np.load(axes_path).astype(np.float64)       # (n_idioms, 12, 768)
    axes_idx   = pd.read_csv(axes_idx_path)
    pooled     = np.load(pooled_path).astype(np.float64)     # (12, 768)

    idiom_to_idx: dict[str, int] = {
        row["idiom_name"]: int(row["idiom_index"])
        for _, row in axes_idx.iterrows()
    }

    # Final-layer (output-space) direction and norm
    output_dir  = temp_dirs[n_layers - 1]                       # (768,) unit vector
    output_norm = coef_norms[n_layers - 1]                      # scalar
    # Year-scaled: adding λ_years × this shifts predicted year by λ_years
    output_dir_year_scaled = output_dir / output_norm

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s …", model_name, device)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model     = BertModel.from_pretrained(model_name).to(device).eval()

    # --- Shared baseline: embed all phrases at all layers once ---
    logger.info("Computing baseline embeddings (no injection) …")
    base_all = embed_sentences(
        model, tokenizer, phrases, device, batch_size=batch_size
    )  # (n_idioms, 12, 768)

    # Helper: score an embedding against the appropriate triviality axis
    def _score(emb: np.ndarray, phrase: str, layer_idx: int) -> float:
        idx  = idiom_to_idx.get(phrase)
        axis = axes[idx, layer_idx] if idx is not None else pooled[layer_idx]
        emb_n = emb / (np.linalg.norm(emb) + 1e-10)
        return 2.0 * float(emb_n @ axis)

    # Baseline scores per (idiom, layer)
    baseline_scores = np.array([
        [_score(base_all[i, layer_idx], phrases[i], layer_idx) for layer_idx in range(n_layers)]
        for i in range(n_idioms)
    ])  # (n_idioms, n_layers)

    # --- Run interventions ---
    all_rows: list[dict] = []
    t0 = time.time()

    # == OUTPUT-SPACE intervention ==
    # Geometric nudge of the final-layer CLS vector; no BERT re-forward-pass.
    # Uses layer-12 direction and norm; scored against layer-12 triviality axis.
    logger.info("Running output-space interventions …")
    layer_12 = n_layers - 1
    for lam_years in lambdas:
        for i, phrase in enumerate(phrases):
            emb0         = base_all[i, layer_12]
            emb_shifted  = emb0 + lam_years * output_dir_year_scaled
            score        = _score(emb_shifted, phrase, layer_12)
            denoms, active_from, first_attested = idiom_meta[phrase]
            all_rows.append({
                "intervention_type":  "output",
                "idiom":              phrase,
                "denominations":      denoms,
                "active_from":        active_from,
                "first_attested":     first_attested,
                "intervention_layer": float("nan"),
                "lambda_years":       float(lam_years),
                "score_paraphrase":   score,
                "score_baseline":     float(baseline_scores[i, layer_12]),
            })

    logger.info(
        "Output-space done  (%d rows, %.1fs)", len(all_rows), time.time() - t0
    )

    # == ACTIVATION-SPACE intervention ==
    # Hook injects λ × temporal_direction_k into CLS at layer k output;
    # the perturbation propagates through layers k+1 … 12 normally.
    logger.info("Running activation-space interventions …")
    act_t0 = time.time()

    for layer_idx in range(n_layers):
        layer_1indexed = layer_idx + 1
        temp_dir  = temp_dirs[layer_idx]    # (768,) unit vector
        coef_norm = coef_norms[layer_idx]   # scalar: years per unit lambda

        for lam_years in lambdas:
            cls_injected = batch_inject_at_layer(
                model, tokenizer, phrases,
                layer=layer_idx,
                direction=temp_dir,
                lambda_val=float(lam_years) / coef_norm,
                device=device,
            )  # (n_idioms, 768) — final-layer CLS after injection

            for i, phrase in enumerate(phrases):
                score = _score(cls_injected[i], phrase, layer_idx)
                denoms, active_from, first_attested = idiom_meta[phrase]
                all_rows.append({
                    "intervention_type":  "activation",
                    "idiom":              phrase,
                    "denominations":      denoms,
                    "active_from":        active_from,
                    "first_attested":     first_attested,
                    "intervention_layer": layer_1indexed,
                    "lambda_years":       float(lam_years),
                    "score_paraphrase":   score,
                    "score_baseline":     float(baseline_scores[i, layer_idx]),
                })

        logger.info(
            "  Activation layer %2d/%d done  (%.1fs elapsed)",
            layer_1indexed, n_layers, time.time() - act_t0,
        )

    # --- Save ---
    out_df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    total_time = time.time() - t0
    logger.info(
        "Saved %d rows → %s  (%.1f KB)",
        len(out_df), out_path, out_path.stat().st_size / 1e3,
    )

    print("\n=== Intervention summary ===")
    print(f"  Idioms:              {n_idioms}")
    print(f"  λ values:            {len(lambdas)}  ({lambdas[0]:.0f} … {lambdas[-1]:.0f} years)")
    print(f"  Output-space rows:   {(out_df['intervention_type'] == 'output').sum():,}")
    print(f"  Activation rows:     {(out_df['intervention_type'] == 'activation').sum():,}")
    print(f"  Total rows:          {len(out_df):,}")
    print(f"  Total time:          {total_time:.1f}s")


def main() -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    parser = argparse.ArgumentParser(
        description="Stage 04: Temporal intervention on idiom phrases."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(cfg, args.dry_run)


if __name__ == "__main__":
    main()
