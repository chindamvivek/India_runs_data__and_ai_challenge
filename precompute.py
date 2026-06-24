"""
precompute.py
~~~~~~~~~~~~~
One-time pre-computation script: encodes all candidates in candidates.jsonl
using sentence-transformers (all-MiniLM-L6-v2) and saves:

  embeddings.npy   — float32 array of shape (N, 384), L2-normalised
  id_index.json    — {"CAND_0000001": 0, "CAND_0000002": 1, ...}

This step runs OUTSIDE the 5-minute ranking budget (per submission spec
Section 10.3). After precompute.py runs once, rank.py loads the .npy file
in ~5 seconds and does a single matrix multiply for semantic scoring.

Usage:
    python precompute.py
    python precompute.py --candidates path/to/candidates.jsonl --out-dir .
    python precompute.py --batch-size 128

Runtime estimate on CPU: ~8-12 minutes for 100K candidates.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

# Default paths relative to the project root
DEFAULT_CANDIDATES = os.path.join(
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
    "candidates.jsonl",
)
DEFAULT_OUT_DIR = "."


def build_candidate_text(candidate: dict) -> str:
    """Assemble a single text representation of a candidate for encoding.
    Must match the same logic used in scorer/semantic_fit.py."""
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])

    headline   = profile.get("headline", "")
    summary    = profile.get("summary", "")
    job_descs  = " ".join(j.get("description", "") for j in career[:3])
    skill_names = " ".join(s.get("name", "") for s in skills[:15])

    return f"{headline} {summary} {job_descs} {skill_names}".strip()


def load_candidates(path: str) -> list[dict]:
    """
    Load candidates from either:
      - A .jsonl file (one JSON object per line), or
      - A .json file containing a JSON array of objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # JSON array format (e.g., sample_candidates.json)
            return json.load(f)

        # JSONL format (candidates.jsonl)
        candidates = []
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping line {line_num} — JSON error: {e}", file=sys.stderr)
        return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings")
    parser.add_argument(
        "--candidates", default=DEFAULT_CANDIDATES,
        help="Path to candidates.jsonl (default: %(default)s)"
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="Directory to save embeddings.npy and id_index.json (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Encoding batch size. Larger = faster but more RAM (default: %(default)s)"
    )
    args = parser.parse_args()

    embeddings_path = os.path.join(args.out_dir, "embeddings.npy")
    index_path      = os.path.join(args.out_dir, "id_index.json")

    # ------------------------------------------------------------------
    print(f"[1/4] Loading candidates from: {args.candidates}")
    t0 = time.time()
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print(f"\n[2/4] Building candidate texts...")
    t1 = time.time()
    candidate_ids: list[str] = []
    texts: list[str] = []
    for c in candidates:
        candidate_ids.append(c["candidate_id"])
        texts.append(build_candidate_text(c))
    print(f"      Built {len(texts):,} text strings in {time.time()-t1:.1f}s")

    # ------------------------------------------------------------------
    print(f"\n[3/4] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    t2 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"      Model loaded in {time.time()-t2:.1f}s")

    print(f"\n      Encoding {len(texts):,} candidates with batch_size={args.batch_size}...")
    t3 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,   # L2-normalise so dot product = cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    encode_time = time.time() - t3
    ms_per = (encode_time / len(texts) * 1000) if texts else 0
    print(f"      Encoded in {encode_time:.1f}s ({ms_per:.1f}ms per candidate)")

    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving outputs to: {args.out_dir}/")
    os.makedirs(args.out_dir, exist_ok=True)

    np.save(embeddings_path, embeddings)
    file_mb = os.path.getsize(embeddings_path) / (1024 * 1024)
    print(f"      embeddings.npy saved — shape={embeddings.shape}, size={file_mb:.1f} MB")

    id_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(id_index, f)
    print(f"      id_index.json saved — {len(id_index):,} entries")

    total_time = time.time() - t0
    print(f"\nDone! Pre-computation complete in {total_time/60:.1f} minutes")
    print(f"   Run rank.py next to use these embeddings.")


if __name__ == "__main__":
    main()
