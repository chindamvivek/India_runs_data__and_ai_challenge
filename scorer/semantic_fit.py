"""
scorer/semantic_fit.py
~~~~~~~~~~~~~~~~~~~~~~
Scores a candidate using a weighted combination of section-level semantic
similarities between the candidate profile and the Job Description,
using sentence-transformers (all-MiniLM-L6-v2).

Weight in composite: 12%  (unchanged — controlled in rank.py)

Design (v6 — Weighted Section Semantic Scoring):
  The candidate profile is split into three semantic sections:

    Section 1 — Career History
        Concatenated ``description`` fields from career_history only.
        Excludes: job titles, company names, dates, duration.

    Section 2 — Summary
        The profile ``summary`` field only.

    Section 3 — Skills
        Concatenated skill ``name`` values only.
        Excludes: proficiency, endorsements, duration_months.

  Three independent embeddings are generated per candidate, producing three
  cosine similarities against the same JD embedding:

        career_similarity  — how AI/ML-aligned the actual job descriptions are
        summary_similarity — how AI/ML-aligned the candidate's own pitch is
        skills_similarity  — how AI/ML-aligned the skill names list are

  Final semantic score (the value returned to rank.py):

        semantic_fit = CAREER_WEIGHT  * career_similarity
                     + SUMMARY_WEIGHT * summary_similarity
                     + SKILLS_WEIGHT  * skills_similarity

  Section weights (tunable — defined as named constants below):
        CAREER_WEIGHT  = 0.60
        SUMMARY_WEIGHT = 0.20
        SKILLS_WEIGHT  = 0.20

  All three section similarities AND the final weighted score are stored in
  _section_scores_cache per candidate, retrievable via get_section_scores().

  NOTE: The old single-embedding concatenated approach (v4) has been removed.
  The pre-computed embeddings (.npy + id_index.json) are no longer used.
  All encoding is done on-the-fly in a single batched model.encode() pass.

Public API
----------
compute_all_semantic_scores(candidates, ...)
    → dict[str, float]   — weighted semantic score per candidate; used by rank.py

compute_all_section_scores(candidates)
    → dict[str, dict[str, float]]
      Keys per candidate: career_similarity, summary_similarity, skills_similarity

get_section_scores(candidate_id)
    → dict[str, float] | None
      Keys: career_similarity, summary_similarity, skills_similarity, final_semantic

initialise(...)
    Pre-warms the model and JD embedding at startup (call once from rank.py).
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# JD query text — semantic representation of what the role actually needs.
# Written to capture the *meaning* not just keywords, so sentence-transformers
# can find "built recommendation engine at product company" as a semantic match.
# ---------------------------------------------------------------------------
JD_QUERY = (
    "Senior AI engineer with production experience building and shipping "
    "embedding-based retrieval systems, vector search, ranking and recommendation "
    "systems deployed to real users at product companies. "
    "Experience with hybrid search, FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch. "
    "Evaluation frameworks: NDCG, MRR, MAP, A/B testing. "
    "Python engineer who writes production code, not just research. "
    "LLM fine-tuning, LoRA, retrieval-augmented generation, applied machine learning. "
    "Startup mindset: shipping fast, iterating on real user feedback, "
    "balancing technical depth with pragmatic delivery."
)

# ---------------------------------------------------------------------------
# Section weights — tune these to adjust the contribution of each section
# to the final semantic score. Must sum to 1.0.
# ---------------------------------------------------------------------------
CAREER_WEIGHT:  float = 0.60
SUMMARY_WEIGHT: float = 0.20
SKILLS_WEIGHT:  float = 0.20
assert abs(CAREER_WEIGHT + SUMMARY_WEIGHT + SKILLS_WEIGHT - 1.0) < 1e-9, (
    "Section weights must sum to 1.0"
)

# ---------------------------------------------------------------------------
# Module-level cache — loaded once per process
# ---------------------------------------------------------------------------
_jd_embedding: np.ndarray | None = None  # shape (384,)
_model = None                             # sentence-transformers model (lazy)

# ---------------------------------------------------------------------------
# Section scores inspection cache
# Populated by compute_all_section_scores() (and therefore by
# compute_all_semantic_scores() which calls it internally).
#
# Key:   candidate_id (str)
# Value: {
#     "career_similarity":  float,
#     "summary_similarity": float,
#     "skills_similarity":  float,
#     "final_semantic":     float,   ← weighted combination
# }
# ---------------------------------------------------------------------------
_section_scores_cache: dict[str, dict[str, float]] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model():
    """Lazily load the sentence-transformers model (only when needed)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_jd_embedding() -> np.ndarray:
    """Return the JD embedding, computing and caching it on first call."""
    global _jd_embedding
    if _jd_embedding is None:
        model = _get_model()
        _jd_embedding = model.encode(
            JD_QUERY, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
    return _jd_embedding


def _clamp(v: float) -> float:
    """Clamp a cosine similarity to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Section text builders — each extracts one clean string per section
# ---------------------------------------------------------------------------

def _build_career_text(candidate: dict) -> str:
    """
    Section 1 — Career History.

    Concatenates only the ``description`` field of each career_history entry.
    Excludes: job titles, company names, start/end dates, duration_months.

    Uses all available roles (not capped at 3) so the model sees the full
    career picture, not just the three most recent roles.
    """
    career = candidate.get("career_history") or []
    descs = [(j.get("description") or "") for j in career if (j.get("description") or "").strip()]
    return " ".join(descs).strip()


def _build_summary_text(candidate: dict) -> str:
    """
    Section 2 — Summary.

    Returns the profile ``summary`` field only.
    """
    profile = candidate.get("profile") or {}
    return (profile.get("summary") or "").strip()


def _build_skills_text(candidate: dict) -> str:
    """
    Section 3 — Skills.

    Concatenates only the ``name`` field from each skills entry.
    Excludes: proficiency, endorsements, duration_months.
    """
    skills = candidate.get("skills") or []
    names = [(s.get("name") or "") for s in skills if (s.get("name") or "").strip()]
    return " ".join(names).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialise(
    embeddings_path: str = "",   # kept for call-site backward compat with rank.py
    index_path: str = "",        # kept for call-site backward compat with rank.py
) -> None:
    """
    Pre-load everything at startup so the first score() call is fast.
    Call this once from rank.py before the scoring loop.

    Note: ``embeddings_path`` and ``index_path`` are accepted but ignored.
    Pre-computed embeddings (.npy) from the old pipeline are no longer used;
    all encoding is done on-the-fly using the section-based pipeline.
    """
    _get_jd_embedding()   # pre-warm model + compute JD embedding once


def get_section_scores(candidate_id: str) -> dict[str, float] | None:
    """
    Return the section-level similarities and final weighted score for one
    candidate.

    The cache is populated by compute_all_section_scores() or by
    compute_all_semantic_scores() (which calls it internally).

    Returns:
        dict with keys:
            career_similarity  float — career descriptions vs JD
            summary_similarity float — profile summary vs JD
            skills_similarity  float — skill names vs JD
            final_semantic     float — weighted combination (used in ranking)
        or None if this candidate_id has not been scored yet.
    """
    return _section_scores_cache.get(candidate_id)


def compute_all_section_scores(
    candidates: list[dict],
) -> dict[str, dict[str, float]]:
    """
    Compute three independent section-level cosine similarities for every
    candidate, plus the final weighted semantic score.

    Strategy:
      - Extract three text strings per candidate (career / summary / skills).
      - Encode all 3 × N strings in a single batched model.encode() call.
      - Compute dot products against the shared, pre-cached JD embedding.
      - Compute weighted combination per candidate.
      - Store all four values in _section_scores_cache.

    Returns:
        dict mapping candidate_id → {
            "career_similarity":  float,
            "summary_similarity": float,
            "skills_similarity":  float,
            "final_semantic":     float,
        }
        All values are in [0.0, 1.0].
    """
    jd_emb = _get_jd_embedding()
    model  = _get_model()

    # Build three parallel text lists (same order as candidates)
    career_texts  = [_build_career_text(c)  for c in candidates]
    summary_texts = [_build_summary_text(c) for c in candidates]
    skills_texts  = [_build_skills_text(c)  for c in candidates]

    # Single batched encode — 3 × N strings, one model forward pass
    all_texts = career_texts + summary_texts + skills_texts
    all_embs  = model.encode(
        all_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=len(candidates) > 200,
    ).astype(np.float32)

    n = len(candidates)
    career_embs  = all_embs[:n]         # rows 0 … n-1
    summary_embs = all_embs[n : 2 * n]  # rows n … 2n-1
    skills_embs  = all_embs[2 * n :]    # rows 2n … 3n-1

    # Cosine similarity = dot product (L2-normalised embeddings)
    career_sims  = career_embs  @ jd_emb  # shape (N,)
    summary_sims = summary_embs @ jd_emb  # shape (N,)
    skills_sims  = skills_embs  @ jd_emb  # shape (N,)

    result: dict[str, dict[str, float]] = {}
    for i, cand in enumerate(candidates):
        cid      = cand["candidate_id"]
        c_sim    = _clamp(career_sims[i])
        s_sim    = _clamp(summary_sims[i])
        sk_sim   = _clamp(skills_sims[i])
        weighted = _clamp(
            CAREER_WEIGHT  * c_sim
            + SUMMARY_WEIGHT * s_sim
            + SKILLS_WEIGHT  * sk_sim
        )
        entry = {
            "career_similarity":  c_sim,
            "summary_similarity": s_sim,
            "skills_similarity":  sk_sim,
            "final_semantic":     weighted,
        }
        result[cid]                 = entry
        _section_scores_cache[cid] = entry   # populate inspection cache

    return result


def compute_all_semantic_scores(
    candidates: list[dict],
    embeddings_path: str = "",   # kept for call-site backward compat with rank.py
    index_path: str = "",        # kept for call-site backward compat with rank.py
) -> dict[str, float]:
    """
    Compute the final semantic score for all candidates.

    This is the primary entry point called by rank.py.
    Returns dict[str, float] — one weighted semantic score per candidate.

    Internally calls compute_all_section_scores() which:
      - Generates one batched embedding for all three sections
      - Computes career_similarity, summary_similarity, skills_similarity
      - Computes the weighted combination as the final semantic score
      - Stores all four values in _section_scores_cache for debugging

    Formula:
        semantic_fit = CAREER_WEIGHT  * career_similarity
                     + SUMMARY_WEIGHT * summary_similarity
                     + SKILLS_WEIGHT  * skills_similarity

        Where:  CAREER_WEIGHT={cw}  SUMMARY_WEIGHT={sw}  SKILLS_WEIGHT={skw}

    Note: ``embeddings_path`` and ``index_path`` are accepted for backward
    compatibility with rank.py's call signature but are not used.

    Returns:
        dict mapping candidate_id -> semantic_score (float in [0.0, 1.0])
    """.format(cw=CAREER_WEIGHT, sw=SUMMARY_WEIGHT, skw=SKILLS_WEIGHT)

    section_results = compute_all_section_scores(candidates)

    return {
        cid: scores["final_semantic"]
        for cid, scores in section_results.items()
    }
