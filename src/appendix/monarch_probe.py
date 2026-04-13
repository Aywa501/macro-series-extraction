"""Appendix — Monarch temporal probe (standalone).

Asks: does BERT's factual knowledge alone encode a coherent sense of
historical time?

Method
------
Two complementary analyses:

1. **PLM scoring** (original): for each monarch, score the pseudo-log-
   likelihood (PLM) of their full name appearing in probe sentences.
   PLM masks each token of the name one at a time and sums the log-
   probabilities — this handles multi-token names like "Henry VIII" or
   "Elizabeth I" correctly.

2. **Multi-mask prediction** (new): replace the name slot with N consecutive
   [MASK] tokens (N = 1, 2, 3) and let BERT greedily decode each position.
   We then check what the model freely predicts across different templates
   and multiple mask widths.  This shows what BERT *thinks* is there without
   restricting it to a candidate list.

No Penn corpus. No Ridge regression. No injection. Pure pretrained weights.

Outputs
-------
Printed tables + src/appendix/monarch_scores_{model}.png

Usage
-----
    python src/appendix/monarch_probe.py
    python src/appendix/monarch_probe.py --model macberth
    python src/appendix/monarch_probe.py --top-n 10
    python src/appendix/monarch_probe.py --predict          # also run prediction mode
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Monarch table  (display_name, probe_phrase, reign_start, reign_end)
# probe_phrase is inserted verbatim into the template sentence for scoring.
# ---------------------------------------------------------------------------
MONARCHS: list[tuple[str, str, int, int]] = [
    ("John",        "john",          1199, 1216),
    ("Henry III",   "henry iii",     1216, 1272),
    ("Edward I",    "edward i",      1272, 1307),
    ("Edward II",   "edward ii",     1307, 1327),
    ("Edward III",  "edward iii",    1327, 1377),
    ("Richard II",  "richard ii",    1377, 1399),
    ("Henry IV",    "henry iv",      1399, 1413),
    ("Henry V",     "henry v",       1413, 1422),
    ("Henry VI",    "henry vi",      1422, 1461),
    ("Edward IV",   "edward iv",     1461, 1483),
    ("Richard III", "richard iii",   1483, 1485),
    ("Henry VII",   "henry vii",     1485, 1509),
    ("Henry VIII",  "henry viii",    1509, 1547),
    ("Edward VI",   "edward vi",     1547, 1553),
    ("Mary I",      "mary i",        1553, 1558),
    ("Elizabeth I", "elizabeth i",   1558, 1603),
    ("James I",     "james i",       1603, 1625),
    ("Charles I",   "charles i",     1625, 1649),
    ("Cromwell",    "cromwell",      1649, 1660),
    ("Charles II",  "charles ii",    1660, 1685),
    ("James II",    "james ii",      1685, 1689),
    ("William III", "william iii",   1689, 1702),
    ("Anne",        "anne",          1702, 1714),
    ("George I",    "george i",      1714, 1727),
    ("George II",   "george ii",     1727, 1760),
    ("George III",  "george iii",    1760, 1820),
    ("George IV",   "george iv",     1820, 1830),
    ("William IV",  "william iv",    1830, 1837),
    ("Victoria",    "victoria",      1837, 1901),
    ("Edward VII",  "edward vii",    1901, 1910),
    ("George V",    "george v",      1910, 1936),
]

# Template sentences — {name} is replaced with the monarch's probe_phrase
# for PLM scoring, or with N consecutive [MASK] tokens for prediction mode.
TEMPLATES = [
    "the king of england was {name} .",
    "the queen of england was {name} .",
    "england was ruled by {name} .",
    "the reign of {name} was notable .",
    "{name} was the king of england .",
    "{name} ruled england .",
    "{name} was the monarch of england .",
    "the english throne was held by {name} .",
]

# Templates used for prediction — subject-position ones work better for
# free generation because the first-position token anchors decoding.
PREDICT_TEMPLATES = [
    "the king of england was {name} .",
    "the queen of england was {name} .",
    "england was ruled by {name} .",
    "{name} was the king of england .",
    "{name} ruled england .",
    "{name} was the monarch of england .",
]


def monarch_at(year: int) -> str:
    for name, _, start, end in MONARCHS:
        if start <= year <= end:
            return name
    return "—"


# ---------------------------------------------------------------------------
# PLM scoring
# ---------------------------------------------------------------------------

def pseudo_log_likelihood(
    model,
    tokenizer,
    sentence: str,
    name_phrase: str,
    device: torch.device,
) -> float:
    """Pseudo-log-likelihood of name_phrase tokens in sentence.

    For each token position in name_phrase (as it appears in the full
    sentence), mask that token and record log P(token | rest). Sum over
    all name tokens.  This is the standard PLM score for a span.
    """
    full_ids = tokenizer.encode(sentence, add_special_tokens=True)
    name_ids = tokenizer.encode(name_phrase, add_special_tokens=False)

    if not name_ids:
        return float("-inf")

    # Find contiguous sub-sequence of name_ids inside full_ids
    name_start = None
    for i in range(len(full_ids) - len(name_ids) + 1):
        if full_ids[i : i + len(name_ids)] == name_ids:
            name_start = i
            break

    if name_start is None:
        return float("-inf")

    total_log_prob = 0.0
    for pos in range(name_start, name_start + len(name_ids)):
        masked = full_ids.copy()
        masked[pos] = tokenizer.mask_token_id
        input_tensor = torch.tensor([masked], device=device)
        attention    = torch.ones_like(input_tensor)

        with torch.no_grad():
            logits = model(input_ids=input_tensor,
                           attention_mask=attention).logits

        log_probs     = torch.log_softmax(logits[0, pos], dim=-1)
        total_log_prob += float(log_probs[full_ids[pos]])

    return total_log_prob / len(name_ids)


def score_all_monarchs(model, tokenizer, device) -> dict[str, float]:
    """Return mean PLM score across all templates for each monarch."""
    scores: dict[str, float] = {}
    n = len(MONARCHS)
    for i, (name, phrase, _, _) in enumerate(MONARCHS):
        print(f"  [{i+1:02d}/{n}] {name} …", end="\r", flush=True)
        template_scores = []
        for template in TEMPLATES:
            sentence = template.format(name=phrase)
            plm = pseudo_log_likelihood(model, tokenizer, sentence, phrase, device)
            if plm > float("-inf"):
                template_scores.append(plm)
        scores[name] = float(np.mean(template_scores)) if template_scores else float("-inf")
    print()
    return scores


# ---------------------------------------------------------------------------
# Multi-mask prediction
# ---------------------------------------------------------------------------

def greedy_decode_masks(
    model,
    tokenizer,
    template: str,
    n_masks: int,
    device: torch.device,
) -> str:
    """Insert n_masks consecutive [MASK] tokens and greedily decode each.

    Returns the decoded string (space-joined subword tokens, stripped of ##).
    """
    mask_str  = " ".join([tokenizer.mask_token] * n_masks)
    sentence  = template.format(name=mask_str)
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)

    # Find the first [MASK] position
    mask_id    = tokenizer.mask_token_id
    mask_positions = [i for i, tok in enumerate(input_ids) if tok == mask_id]

    if len(mask_positions) != n_masks:
        return "<tokenization error>"

    ids = input_ids.copy()
    decoded_tokens: list[str] = []

    # Iterative greedy left-to-right
    for pos in mask_positions:
        input_tensor = torch.tensor([ids], device=device)
        attention    = torch.ones_like(input_tensor)
        with torch.no_grad():
            logits = model(input_ids=input_tensor,
                           attention_mask=attention).logits
        pred_id     = int(logits[0, pos].argmax())
        ids[pos]    = pred_id
        decoded_tokens.append(tokenizer.convert_ids_to_tokens([pred_id])[0])

    # Clean up subword markers and join
    clean = " ".join(t.lstrip("#") for t in decoded_tokens)
    return clean


def run_prediction_mode(model, tokenizer, device, max_masks: int = 3) -> None:
    """For each template × mask-width, show what BERT freely predicts."""
    print("\n" + "=" * 72)
    print("MULTI-MASK FREE PREDICTION")
    print("=" * 72)
    print("What does BERT predict when given N consecutive [MASK] tokens?")
    print()

    for n in range(1, max_masks + 1):
        print(f"--- {n} mask(s) ---")
        for tmpl in PREDICT_TEMPLATES:
            pred = greedy_decode_masks(model, tokenizer, tmpl, n, device)
            display_tmpl = tmpl.format(name="[" + " MASK" * n + " ]")
            print(f"  {display_tmpl}")
            print(f"      → predicted name: '{pred}'")
        print()


def predict_vs_plm(model, tokenizer, device, max_masks: int = 3) -> None:
    """Side-by-side: PLM top-5 vs greedy prediction from multiple mask widths."""
    print("\n" + "=" * 72)
    print("TOP-5 PLM MONARCHS  vs  GREEDY MULTI-MASK PREDICTIONS")
    print("=" * 72)

    # Score every monarch
    print("Computing PLM scores …")
    scores = score_all_monarchs(model, tokenizer, device)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\nTop-5 by PLM:")
    for rank, (name, sc) in enumerate(ranked[:5], 1):
        _, _, s, e = next(r for r in MONARCHS if r[0] == name)
        print(f"  {rank}. {name:>14}  PLM={sc:.4f}  ({s}–{e})")

    print()
    # Greedy predictions
    tmpl = "the english throne was held by {name} ."
    for n in range(1, max_masks + 1):
        pred = greedy_decode_masks(model, tokenizer, tmpl, n, device)
        display = tmpl.format(name="[" + " MASK" * n + " ]")
        print(f"  {n} mask(s): '{pred}'   ← from: {display}")

    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(scores: dict[str, float], model_name: str, model_key: str) -> None:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    all_scores  = np.array([scores[n] for n, _, _, _ in MONARCHS], dtype=float)
    shifted     = all_scores - all_scores.mean()
    softmax_all = np.exp(shifted)
    softmax_all /= softmax_all.sum()

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    cmap    = plt.cm.tab20
    n_mon   = len(MONARCHS)
    colours = [cmap(i / n_mon) for i in range(n_mon)]

    # Panel A: PLM score timeline
    ax = axes[0]
    for i, (name, _, start, end) in enumerate(MONARCHS):
        score = scores[name]
        width = max(end - start, 3)
        ax.barh(y=score, left=start, width=width,
                height=abs(all_scores.max() - all_scores.min()) * 0.015,
                color=colours[i], alpha=0.85, edgecolor="white", linewidth=0.3)
        if end - start >= 15:
            ax.text((start + end) / 2, score, name.split()[0],
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold")
    ax.set_ylabel("PLM score (mean log-prob per token)", fontsize=10)
    ax.set_title(
        f"{model_name} — pseudo-log-likelihood of each monarch's FULL name\n"
        f"({len(TEMPLATES)} probe templates, multi-token names scored via PLM)",
        fontsize=11,
    )
    ax.set_xlim(1190, 1945)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Panel B: softmax probability
    ax2 = axes[1]
    for i, (name, _, start, end) in enumerate(MONARCHS):
        prob  = softmax_all[i]
        width = max(end - start, 3)
        ax2.bar(x=(start + end) / 2, height=prob * 100, width=width * 0.85,
                color=colours[i], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.set_ylabel("Softmax share (%)", fontsize=10)
    ax2.set_title("Softmax probability over all monarchs (by reign period)", fontsize=10)
    ax2.set_xlim(1190, 1945)
    ax2.grid(axis="y", linestyle=":", alpha=0.35)

    # Panel C: rank timeline
    ax3 = axes[2]
    rank_lookup = {name: rank for rank, (name, _) in enumerate(ranked, 1)}
    for i, (name, _, start, end) in enumerate(MONARCHS):
        rank  = rank_lookup[name]
        width = max(end - start, 3)
        ax3.bar(x=(start + end) / 2, height=(n_mon + 1 - rank),
                width=width * 0.85,
                color=colours[i], alpha=0.85, edgecolor="white", linewidth=0.3)
        if end - start >= 15:
            ax3.text((start + end) / 2, (n_mon + 1 - rank) / 2, f"#{rank}",
                     ha="center", va="center", fontsize=6.5, color="white",
                     fontweight="bold")
    ax3.set_xlabel("Year", fontsize=11)
    ax3.set_ylabel("Inverted rank\n(higher = BERT knows better)", fontsize=9)
    ax3.set_title(f"BERT familiarity rank by reign period (1 = most familiar of {n_mon})",
                  fontsize=10)
    ax3.set_xlim(1190, 1945)
    ax3.grid(axis="y", linestyle=":", alpha=0.35)

    dynasty_changes = [
        (1399, "Lancaster"), (1461, "York"),    (1485, "Tudor"),
        (1603, "Stuart"),    (1714, "Hanover"), (1901, "Windsor"),
    ]
    for year, label in dynasty_changes:
        for ax_ in axes:
            ax_.axvline(year, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        axes[0].text(year + 2, axes[0].get_ylim()[0], label,
                     rotation=90, fontsize=6.5, color="grey", alpha=0.8, va="bottom")

    fig.tight_layout(pad=2.0)
    out_path = OUT_DIR / f"monarch_scores_{model_key}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model_key: str, top_n: int, predict: bool) -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if model_key == "bert":
        model_name = "bert-base-uncased"
    elif model_key == "macberth":
        model_name = "emanjavacas/MacBERTh"
    else:
        model_name = model_key

    print(f"Loading {model_name} …", flush=True)
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    print("Loaded.\n")

    if predict:
        scores = predict_vs_plm(model, tokenizer, device, max_masks=3)
        print()
        run_prediction_mode(model, tokenizer, device, max_masks=3)
    else:
        print("Computing pseudo-log-likelihood scores for all monarchs …")
        scores = score_all_monarchs(model, tokenizer, device)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Monarch PLM scores (mean normalised log-prob across templates) ===")
    print(f"  {'Rank':>4}  {'Monarch':>16}  {'PLM score':>10}  Reign")
    print(f"  {'----':>4}  {'--------':>16}  {'---------':>10}  -----")
    for rank, (name, score) in enumerate(ranked, 1):
        _, _, start, end = next(r for r in MONARCHS if r[0] == name)
        print(f"  {rank:>4}  {name:>16}  {score:>10.4f}  {start}–{end}")

    # Softmax
    all_scores  = np.array([scores[n] for n, _, _, _ in MONARCHS], dtype=float)
    shifted     = all_scores - all_scores.mean()
    softmax_all = np.exp(shifted)
    softmax_all /= softmax_all.sum()

    print(f"\n=== Softmax distribution (top {top_n}) ===")
    sm_ranked = sorted(zip([n for n, _, _, _ in MONARCHS], softmax_all),
                       key=lambda x: -x[1])
    for name, prob in sm_ranked[:top_n]:
        _, _, start, end = next(r for r in MONARCHS if r[0] == name)
        print(f"  {name:>16}: {prob:.3%}  ({start}–{end})")

    make_plots(scores, model_name, model_key)

    # Template breakdown for top/bottom 5
    top5 = [name for name, _ in ranked[:5]]
    bot5 = [name for name, _ in ranked[-5:]]
    print("\n=== Template breakdown — top 5 most familiar ===")
    _print_template_breakdown(model, tokenizer, device, top5)
    print("\n=== Template breakdown — bottom 5 least familiar ===")
    _print_template_breakdown(model, tokenizer, device, bot5)


def _print_template_breakdown(model, tokenizer, device, names: list[str]) -> None:
    data = [next(r for r in MONARCHS if r[0] == n) for n in names]
    phrases = [r[1] for r in data]
    header = f"  {'Template':>42}" + "".join(f"  {n:>14}" for n in names)
    print(header)
    print("  " + "-" * (42 + 16 * len(names)))
    for template in TEMPLATES:
        row = f"  {template[:41]:>41}"
        for name, phrase in zip(names, phrases):
            sentence = template.format(name=phrase)
            plm = pseudo_log_likelihood(model, tokenizer, sentence, phrase, device)
            row += f"  {plm:>14.4f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monarch PLM probe: extract temporal knowledge from BERT directly."
    )
    parser.add_argument("--model", default="bert",
                        choices=["bert", "macberth"],
                        help="Which model to probe (default: bert).")
    parser.add_argument("--top-n", type=int, default=8,
                        help="Monarchs to highlight in softmax table (default: 8).")
    parser.add_argument("--predict", action="store_true",
                        help="Also run multi-mask free prediction mode.")
    args = parser.parse_args()
    run(model_key=args.model, top_n=args.top_n, predict=args.predict)


if __name__ == "__main__":
    main()
