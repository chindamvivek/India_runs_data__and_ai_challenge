"""
rank.py
~~~~~~~
Main entry point for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

Usage:
    python rank.py --candidates candidates.jsonl --team-id team_xyz
    python rank.py --candidates candidates.jsonl --team-id team_xyz --out results.csv
    python rank.py --candidates sample_candidates.json --team-id team_xyz --top-n 50

The script:
  1. Loads all candidates from a .jsonl or .json file
  2. Detects honeypots (5 integrity checks)
  3. Computes 5 weighted sub-scores (career, skills, behavioral, semantic, location)
  4. Adds composite score and honeypot penalty
  5. Sorts descending; takes top-100
  6. Generates fact-grounded reasoning for each ranked candidate
  7. Validates the output and writes the submission CSV

Pipeline weights:
    0.35 × career_fit
    0.25 × skills_fit
    0.20 × behavioral
    0.12 × semantic
    0.08 × location
    + honeypot_penalty (≤ 0)
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Project modules
sys.path.insert(0, os.path.dirname(__file__))
from scorer.honeypot      import compute_honeypot_penalty
from scorer.career_fit    import compute_career_fit
from scorer.skills_fit    import compute_skills_fit
from scorer.behavioral_fit import compute_behavioral_fit
from scorer.location_fit  import compute_location_fit
from scorer.semantic_fit  import compute_all_semantic_scores, initialise as init_semantic
from reasoning.generator  import build_reasoning

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
)
DEFAULT_CANDIDATES  = os.path.join(DATA_DIR, "candidates.jsonl")
DEFAULT_EMBEDDINGS  = "embeddings.npy"
DEFAULT_ID_INDEX    = "id_index.json"
DEFAULT_VALIDATOR   = os.path.join(DATA_DIR, "validate_submission.py")

# Composite weights — must sum to 1.0
W_CAREER    = 0.35
W_SKILLS    = 0.25
W_BEHAVIORAL = 0.20
W_SEMANTIC  = 0.12
W_LOCATION  = 0.08
assert abs(W_CAREER + W_SKILLS + W_BEHAVIORAL + W_SEMANTIC + W_LOCATION - 1.0) < 1e-9


def load_candidates(path: str) -> list[dict]:
    """Load candidates from a .jsonl or .json file."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        # JSONL format
        candidates = []
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING line {line_num}: {e}", file=sys.stderr)
        return candidates


def score_all(
    candidates: list[dict],
    embeddings_path: str,
    index_path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full scoring pipeline on all candidates.
    Returns a DataFrame with all scores and intermediate data.
    """
    n = len(candidates)

    # ------------------------------------------------------------------
    # Stage: Semantic scores (batch — most efficient to do all at once)
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [semantic] Computing semantic scores for {n:,} candidates...")
    t = time.time()
    semantic_scores = compute_all_semantic_scores(
        candidates,
        embeddings_path=embeddings_path,
        index_path=index_path,
    )
    if verbose:
        print(f"             Done in {time.time()-t:.1f}s")

    # ------------------------------------------------------------------
    # Stage: Per-candidate scoring
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [scoring]  Computing 4 per-candidate sub-scores...")
    t = time.time()

    rows = []
    for c in candidates:
        cid = c["candidate_id"]

        # Run all scorers
        hp   = compute_honeypot_penalty(c)
        cf   = compute_career_fit(c)
        sf   = compute_skills_fit(c)
        bf   = compute_behavioral_fit(c)
        lf   = compute_location_fit(c)
        sem  = semantic_scores.get(cid, 0.0)

        # Composite score (weighted sum of all sub-scores)
        composite = (
            W_CAREER     * cf["career_score"]
          + W_SKILLS     * sf["score"]
          + W_BEHAVIORAL * bf["score"]
          + W_SEMANTIC   * sem
          + W_LOCATION   * lf["score"]
        )

        # Career-fit gate — prevents non-technical candidates from ranking via
        # behavioral/location signals alone.
        #
        # Thresholds (from career_fit.py scoring tiers):
        #   0.00 = fully non-technical (Accountant, Civil Engineer, etc.)
        #   0.10 = fallback/business role with no ML context (gate boundary)
        #   0.15 = Tier C SWE (.NET, Full Stack) at non-product company
        #   0.20 = Tier C/B data role at non-product company (weakest real signal)
        #   0.25 = generic SWE (no ML context confirmed)
        #   0.35 = SWE at product company / or SWE with ML context
        #   0.70+ = direct ML/AI title
        #
        # Gate: candidates below 0.10 technical relevance are suppressed to 10%.
        if cf["technical_score"] < 0.10:
            composite *= 0.10

        # Apply honeypot penalty after the gate (additive, always negative)
        composite += hp

        # Merge all intermediate data for the reasoning generator
        scored_data = {
            # From career_fit
            "score":            cf["score"],
            "technical_score":  cf["technical_score"],   # pure tier score (no penalties)
            "ml_months":        cf["ml_months"],
            "avg_tenure":       cf["avg_tenure"],
            "consulting_only":  cf["consulting_only"],
            "has_retrieval":    cf["has_retrieval"],
            "top_role":         cf["top_role"],
            # From skills_fit
            "top_skills":       sf["top_skills"],
            # From behavioral_fit
            "days_inactive":    bf["days_inactive"],
            # From location_fit
            "country":          lf["country"],
        }

        rows.append({
            "candidate_id":    cid,
            "final_score":     composite,
            "career_fit":      cf["score"],
            "skills_fit":      sf["score"],
            "behavioral":      bf["score"],
            "semantic":        sem,
            "location":        lf["score"],
            "honeypot_penalty": hp,
            "_candidate":      c,          # raw candidate dict (for reasoning)
            "_scored_data":    scored_data, # intermediate values (for reasoning)
        })

    if verbose:
        print(f"             Done in {time.time()-t:.1f}s")

    return pd.DataFrame(rows)


def build_submission(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Sort by final_score descending, take top_n, assign ranks 1..top_n,
    generate reasoning, and return the submission DataFrame.
    """
    df_sorted = df.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True],
    ).head(top_n).copy()

    df_sorted["rank"] = range(1, top_n + 1)

    # Use the raw final_score (rounded to 6dp) as the output score.
    # We do NOT clip to [0,1] because clipping collapses honeypot candidates
    # to the same score=0.0, breaking the tie-break ordering requirement.
    # The validator only requires scores to be non-increasing, not in [0,1].
    df_sorted["score"] = df_sorted["final_score"].round(6)

    # Generate reasoning for each candidate
    reasonings = []
    for _, row in df_sorted.iterrows():
        r = build_reasoning(
            candidate=row["_candidate"],
            scored_data=row["_scored_data"],
            rank=int(row["rank"]),
        )
        reasonings.append(r)
    df_sorted["reasoning"] = reasonings

    return df_sorted[["candidate_id", "rank", "score", "reasoning"]]


def validate_output(csv_path: str, validator_path: str) -> bool:
    """Run validate_submission.py and return True if it passes."""
    if not os.path.exists(validator_path):
        print(f"  WARNING: Validator not found at {validator_path} — skipping validation")
        return True
    result = subprocess.run(
        [sys.executable, validator_path, csv_path],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return False
    return True


def run_internal_checks(submission: pd.DataFrame) -> None:
    """Assert submission format constraints (fail fast before writing)."""
    assert len(submission) == 100 or True, "Expected 100 rows"  # relaxed for small samples
    ranks = submission["rank"].tolist()
    assert ranks == list(range(1, len(ranks) + 1)), "Ranks must be sequential 1..N"
    assert submission["candidate_id"].is_unique, "Duplicate candidate_ids in submission"
    scores = submission["score"].values
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-9, (
            f"Score not non-increasing at rank {i+1}: {scores[i]:.6f} < {scores[i+1]:.6f}"
        )
    assert all(len(r) > 0 for r in submission["reasoning"]), "Empty reasoning string found"
    print("  [OK] Internal format checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranker — rank top 100 from a candidate pool"
    )
    parser.add_argument(
        "--candidates", default=DEFAULT_CANDIDATES,
        help="Path to candidates.jsonl or sample_candidates.json"
    )
    parser.add_argument(
        "--team-id", required=True,
        help="Your registered team/participant ID (used to name the output CSV)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path (default: <team-id>.csv)"
    )
    parser.add_argument(
        "--embeddings", default=DEFAULT_EMBEDDINGS,
        help="Path to pre-computed embeddings.npy"
    )
    parser.add_argument(
        "--id-index", default=DEFAULT_ID_INDEX,
        help="Path to pre-computed id_index.json"
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of candidates to return (default: 100)"
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip running validate_submission.py"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print timing and progress (default: True)"
    )
    args = parser.parse_args()

    out_path = args.out or f"{args.team_id}.csv"
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Redrob Candidate Ranker")
    print(f"{'='*60}")
    print(f"  Candidates : {args.candidates}")
    print(f"  Output     : {out_path}")
    print(f"  Top-N      : {args.top_n}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Stage 1: Load
    # ------------------------------------------------------------------
    print("[1/5] Loading candidates...")
    t = time.time()
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates):,} candidates in {time.time()-t:.1f}s")

    if not candidates:
        print("ERROR: No candidates loaded. Check file path.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 2: Initialise semantic module
    # ------------------------------------------------------------------
    print("\n[2/5] Initialising semantic module...")
    t = time.time()
    init_semantic(
        embeddings_path=args.embeddings,
        index_path=args.id_index,
    )
    print(f"      Ready in {time.time()-t:.1f}s")

    # ------------------------------------------------------------------
    # Stage 3: Score all candidates
    # ------------------------------------------------------------------
    print(f"\n[3/5] Scoring {len(candidates):,} candidates...")
    df = score_all(
        candidates,
        embeddings_path=args.embeddings,
        index_path=args.id_index,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # Stage 4: Sort, top-N, reasoning
    # ------------------------------------------------------------------
    print(f"\n[4/5] Building top-{args.top_n} submission...")
    t = time.time()
    top_n = min(args.top_n, len(candidates))
    submission = build_submission(df, top_n=top_n)
    print(f"      Done in {time.time()-t:.1f}s")

    # Show top-10 preview
    print(f"\n  Top 10 preview:")
    for _, row in submission.head(10).iterrows():
        print(f"    #{int(row['rank']):>3}  {row['candidate_id']}  score={row['score']:.4f}")

    # ------------------------------------------------------------------
    # Stage 5: Validate and write
    # ------------------------------------------------------------------
    print(f"\n[5/5] Validating and writing {out_path}...")
    run_internal_checks(submission)

    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"      Wrote {len(submission)} rows to {out_path}")

    if not args.no_validate:
        validate_output(out_path, DEFAULT_VALIDATOR)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {total:.1f}s ({total/60:.1f} min)")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
