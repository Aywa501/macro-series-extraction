"""openai_batch_probe.py — temporal_manifold + sequence_comprehensive analysis
for text-embedding-3-small via the OpenAI Batch API.

Collects every text both probing scripts need, submits a single batch job,
then downloads results and runs the full analysis without ever hitting rate
limits.  50 % cheaper than the synchronous API too.

Two-phase workflow
------------------
    # Phase 1 — collect texts, upload file, create batch (~seconds)
    OPENAI_API_KEY=sk-... python3 src/exploratory/openai_batch_probe.py --submit

    # Check progress at any time
    OPENAI_API_KEY=sk-... python3 src/exploratory/openai_batch_probe.py --status

    # Phase 2 — download results, run analysis, write CSVs + figures
    OPENAI_API_KEY=sk-... python3 src/exploratory/openai_batch_probe.py --retrieve
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))               # probe_utils
sys.path.insert(0, str(_HERE / "temporal"))  # temporal_manifold
sys.path.insert(0, str(_HERE / "sequences")) # sequence_comprehensive

from probe_utils import MODEL_NAMES  # noqa: E402
import temporal_manifold as tm        # noqa: E402
import sequence_comprehensive as sc   # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_KEY = "openai"
OAI_MODEL = MODEL_NAMES[MODEL_KEY]   # "text-embedding-3-small"
N_LAYERS  = 1                        # API models have no layer structure

BATCH_DIR  = tm.OUT_DIR.parent / "openai_batch"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = BATCH_DIR / "batch_state.json"


# ---------------------------------------------------------------------------
# Text collection — enumerate every sentence both scripts will embed
# ---------------------------------------------------------------------------

def collect_all_texts() -> list[str]:
    """Return ordered, de-duplicated list of all texts needed by both scripts."""
    seen:  set[str]  = set()
    texts: list[str] = []

    def add(t: str) -> None:
        if t not in seen:
            seen.add(t)
            texts.append(t)

    years = np.arange(tm.YEAR_MIN, tm.YEAR_MAX + tm.YEAR_STEP, tm.YEAR_STEP)

    # ── temporal_manifold: year-carrier trajectory ────────────────────────
    for tmpl in tm.SHARED_TEMPLATES:
        for y in years:
            add(tmpl.format(x=int(y)))

    # ── temporal_manifold: period label centroids ─────────────────────────
    series = tm.load_series()
    for _name, df in series.items():
        for _, row in df.iterrows():
            lbl = str(row["name"])
            sy  = float(row["start_year"])
            ey  = float(row["end_year"]) if pd.notna(row["end_year"]) else sy
            mid = (sy + ey) / 2.0
            if tm.YEAR_MIN <= mid <= tm.YEAR_MAX:
                for tmpl in tm.SHARED_TEMPLATES:
                    add(tmpl.format(x=lbl))

    # ── sequence_comprehensive: year-carrier direction ────────────────────
    for tmpl in sc.YEAR_CARRIER:
        for y in sc.YEAR_RANGE:
            add(tmpl.format(y=int(y)))

    # ── sequence_comprehensive: sequence sentences ────────────────────────
    for seq_def in sc.SEQUENCES.values():
        for p in seq_def["periods"]:
            for tmpl in seq_def["templates"]:
                add(tmpl.format(label=p.label))

    return texts


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

def submit_batch(texts: list[str], client) -> str:
    """Write JSONL, upload, create batch, persist state. Returns batch_id."""
    jsonl_path = BATCH_DIR / "requests.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            record = {
                "custom_id": f"req-{i}",
                "method":    "POST",
                "url":       "/v1/embeddings",
                "body": {
                    "model":           OAI_MODEL,
                    "input":           text,
                    "encoding_format": "float",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(texts)} requests → {jsonl_path.name}")

    print("Uploading …")
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    print(f"  file_id: {batch_file.id}")

    print("Creating batch job …")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
    )
    print(f"  batch_id: {batch.id}")
    print(f"  status:   {batch.status}")

    state = {
        "batch_id": batch.id,
        "file_id":  batch_file.id,
        "n_texts":  len(texts),
        "texts":    texts,
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"\nState saved → {STATE_FILE}")
    print("Run --retrieve once the batch completes (check with --status).")
    return batch.id


def _load_state() -> dict:
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"No batch state at {STATE_FILE}. Run --submit first.")
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


def check_status(client) -> None:
    state = _load_state()
    batch = client.batches.retrieve(state["batch_id"])
    rc    = batch.request_counts
    print(f"Batch ID : {batch.id}")
    print(f"Status   : {batch.status}")
    if rc:
        pct = rc.completed / max(rc.total, 1) * 100
        print(f"Progress : {rc.completed}/{rc.total} completed "
              f"({pct:.0f} %)  |  {rc.failed} failed")
    if batch.output_file_id:
        print(f"Output   : {batch.output_file_id}")


def download_cache(client) -> dict[str, np.ndarray]:
    """Download completed batch results. Returns {text: (1, H)} cache."""
    state = _load_state()
    batch = client.batches.retrieve(state["batch_id"])

    if batch.status != "completed":
        print(f"Batch not complete yet (status: {batch.status}). "
              "Try --retrieve again later.")
        sys.exit(1)

    print(f"Downloading results (output_file: {batch.output_file_id}) …")
    content = client.files.content(batch.output_file_id)
    texts   = state["texts"]

    cache:  dict[str, np.ndarray] = {}
    errors: int = 0
    for line in content.text.splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        if result.get("error"):
            errors += 1
            continue
        idx = int(result["custom_id"].split("-")[1])
        emb = result["response"]["body"]["data"][0]["embedding"]
        cache[texts[idx]] = np.array(emb, dtype=np.float32)[np.newaxis, :]  # (1, H)

    print(f"  {len(cache)} embeddings loaded  ({errors} errors)")
    if errors:
        print(f"  WARNING: {errors} requests failed — some results may be missing")
    return cache


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------

def run_temporal(cache: dict[str, np.ndarray]) -> None:
    """Run temporal_manifold analysis using cached OpenAI embeddings."""
    print("\n── temporal_manifold (openai) ──────────────────────────────────")

    def embed_fn(text: str) -> np.ndarray:
        return cache[text]   # (1, H=1536)

    layer  = 0   # single embedding layer for API models
    series = tm.load_series()

    print("Year trajectory …")
    years_traj, year_embs_full = tm.compute_year_trajectory_all_layers(
        embed_fn, tm.SHARED_TEMPLATES)
    embs_traj      = year_embs_full[:, 0, :]           # (n_years, H)
    all_layer_embs = {0: embs_traj}

    year_dirs = tm.compute_year_dirs(year_embs_full, years_traj)

    print("Period centroids …")
    period_data_all = tm.compute_period_centroids_all_layers(
        series, embed_fn, tm.SHARED_TEMPLATES)
    period_data_all["year_carrier"] = {
        "labels": [str(int(y)) for y in years_traj],
        "mids":   years_traj.copy(),
        "embs":   year_embs_full,              # (n_years, 1, H)
    }
    period_data = {
        name: {
            "labels": d["labels"],
            "mids":   d["mids"],
            "embs":   d["embs"][:, 0, :],     # (N, H)
        }
        for name, d in period_data_all.items()
    }
    n_pts = sum(len(d["mids"]) for d in period_data.values())
    print(f"{n_pts} centroids total.")

    print("Figures …")
    tm.plot_year_manifold_2d(years_traj, embs_traj, period_data, layer, MODEL_KEY)
    tm.plot_velocity_curvature(years_traj, embs_traj, period_data, layer, MODEL_KEY)
    tm.plot_cross_cultural(period_data, years_traj, layer, MODEL_KEY)
    tm.plot_joint_mds(period_data, layer, MODEL_KEY)
    tm.plot_pc1_projection(years_traj, embs_traj, period_data, layer, MODEL_KEY)
    tm.plot_series_velocity(period_data, layer, MODEL_KEY)
    # layer_comparison skipped — no layer sweep for single-layer model

    print("Metrics …")
    metrics_df = tm.compute_series_metrics(period_data_all, year_dirs)
    csv_path   = tm.OUT_DIR / f"manifold_results_{MODEL_KEY}.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"  → {csv_path.name}  ({len(metrics_df)} rows)")


def run_sequences(cache: dict[str, np.ndarray]) -> None:
    """Run sequence_comprehensive analysis using cached OpenAI embeddings."""
    print("\n── sequence_comprehensive (openai) ─────────────────────────────")

    def embed_fn(text: str) -> np.ndarray:
        return cache[text]   # (1, H=1536)

    results, aux, _year_dirs = sc.run_all(embed_fn, N_LAYERS, sc.YEAR_CARRIER)

    csv_path = sc.OUT_DIR / f"comp_results_{MODEL_KEY}.csv"
    results.to_csv(csv_path, index=False)
    print(f"Results → {csv_path}")

    print("Figures …")
    # Layer-sweep figures skipped (N_LAYERS == 1)
    sc.fig_pca_best(aux, results, MODEL_KEY)
    sc.fig_centroid_dist(aux, results, MODEL_KEY)
    sc.fig_confusion(aux, results, MODEL_KEY)
    sc.fig_composite(results, MODEL_KEY)
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--submit",   action="store_true",
                     help="Collect texts, upload, create batch")
    grp.add_argument("--status",   action="store_true",
                     help="Check batch progress")
    grp.add_argument("--retrieve", action="store_true",
                     help="Download results and run full analysis")
    args = parser.parse_args()

    from openai import OpenAI  # type: ignore
    client = OpenAI()          # reads OPENAI_API_KEY from environment

    if args.submit:
        print("Collecting texts …")
        texts = collect_all_texts()
        print(f"  {len(texts)} unique texts to embed")
        submit_batch(texts, client)

    elif args.status:
        check_status(client)

    elif args.retrieve:
        cache = download_cache(client)
        run_temporal(cache)
        run_sequences(cache)
        print("\nAll done.")


if __name__ == "__main__":
    main()
