"""
scorer/semantic_fit.py
~~~~~~~~~~~~~~~~~~~~~~
Scores a candidate using cosine similarity between a precomputed candidate
embedding and the Job Description embedding.

Weight in composite: 12%  (unchanged — controlled in rank.py)

Design (v7 — Precomputed Embeddings + Sandbox Fallback):

  TWO paths depending on environment:

  ── FAST PATH (rank.py stress test, local) ──────────────────────────────────
  At startup (initialise()), two things happen:
    1. The sentence-transformers model encodes the JD query text ONCE → jd_emb (384,)
    2. The precomputed candidate embeddings are loaded from embeddings.npy
       (shape: N×384, L2-normalised float32, produced by precompute.py)

  During scoring (compute_all_semantic_scores()):
    - A SINGLE matrix multiply:  embeddings @ jd_emb  →  cosine similarities (N,)
    - No per-candidate model.encode() calls.
    - Runtime for 100K candidates: < 1 second.

  ── SLOW PATH (Streamlit sandbox, no embeddings.npy) ─────────────────────────
  When embeddings.npy is NOT present (e.g., deployed to Streamlit Cloud),
  falls back to on-the-fly encoding of each candidate individually.
  Judges typically upload 1-2 candidates to the sandbox, so this is fine.

  This is the correct architecture for both the 5-minute compute budget AND
  the Streamlit sandbox deployment.

  DO NOT revert to v6 (3-section on-the-fly encoding for all candidates) —
  that approach encoded 3×N=300K texts and took ~1.5 HOURS on CPU for 100K
  candidates, completely failing the 5-minute budget requirement.

  Precompute once with:
      python precompute.py
  Then rank as many times as needed with:
      python rank.py ...

Public API
----------
initialise(embeddings_path, index_path)
    Load precomputed embeddings + encode JD once. Call once from rank.py.

compute_all_semantic_scores(candidates, embeddings_path, index_path)
    → dict[str, float]   — cosine similarity score per candidate_id.
    Uses fast matrix multiply if embeddings.npy is available,
    falls back to per-candidate on-the-fly encoding otherwise (sandbox mode).
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
# Module-level state — loaded once per process by initialise()
# ---------------------------------------------------------------------------
_jd_embedding:   np.ndarray | None = None        # shape (384,)
_candidate_embs: np.ndarray | None = None        # shape (N, 384)
_id_to_idx:      dict[str, int] | None = None    # candidate_id → row index
_model = None                                     # sentence-transformers model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model():
    """Lazily load the sentence-transformers model (only when needed)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # local_files_only=True → use cached model, no network call during ranking.
        # Satisfies hackathon constraint: "no network access during ranking".
        # Model must be cached first (done automatically on first-ever run).
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            # First-ever run: model not cached yet — allow download once.
            _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _clamp(v: float) -> float:
    """Clamp a cosine similarity to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(v)))


def _build_candidate_text(candidate: dict) -> str:
    """
    Build a single text string representing the candidate for embedding.
    Matches the logic used in precompute.py so sandbox scores are consistent
    with the precomputed embeddings used during the stress test.
    """
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])

    headline    = profile.get("headline", "")
    summary     = profile.get("summary", "")
    job_descs   = " ".join(j.get("description", "") for j in career[:3])
    skill_names = " ".join(s.get("name", "") for s in skills[:15])

    return f"{headline} {summary} {job_descs} {skill_names}".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialise(
    embeddings_path: str = "embeddings.npy",
    index_path: str = "id_index.json",
) -> None:
    """
    Pre-load everything at startup so scoring is fast.
    Call this once from rank.py before the scoring loop.

    Loads precomputed embeddings if available; otherwise just warms the model
    and JD embedding so the sandbox fallback path is ready.

    Args:
        embeddings_path: Path to embeddings.npy produced by precompute.py.
        index_path:      Path to id_index.json produced by precompute.py.
    """
    global _jd_embedding, _candidate_embs, _id_to_idx

    import json, os

    # Always encode the JD once (fast, single text)
    model = _get_model()
    _jd_embedding = model.encode(
        JD_QUERY,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Load precomputed candidate embeddings if available (fast path)
    if os.path.exists(embeddings_path) and os.path.exists(index_path):
        _candidate_embs = np.load(embeddings_path).astype(np.float32)
        with open(index_path, "r", encoding="utf-8") as f:
            _id_to_idx = json.load(f)
    else:
        # Sandbox mode: no precomputed embeddings, will encode on-the-fly
        _candidate_embs = None
        _id_to_idx = None


def compute_all_semantic_scores(
    candidates: list[dict],
    embeddings_path: str = "embeddings.npy",
    index_path: str = "id_index.json",
) -> dict[str, float]:
    """
    Compute the final semantic score for all candidates.

    FAST PATH (precomputed embeddings available):
        Single matrix multiply — < 1 second for 100K candidates.

    SLOW PATH (sandbox, no embeddings.npy):
        Per-candidate on-the-fly encoding. Acceptable for small uploads
        (judges upload 1-50 candidates to the sandbox).

    Returns:
        dict[candidate_id, float] — cosine similarity in [0.0, 1.0].
    """
    global _jd_embedding, _candidate_embs, _id_to_idx

    # Auto-initialise if not already called
    if _jd_embedding is None:
        initialise(embeddings_path, index_path)

    result: dict[str, float] = {}

    if _candidate_embs is not None and _id_to_idx is not None:
        # ── FAST PATH: single matrix multiply ────────────────────────────────
        # (N, 384) @ (384,) → (N,) cosine similarities in < 1s for 100K
        all_sims = (_candidate_embs @ _jd_embedding).astype(float)
        for cand in candidates:
            cid = cand["candidate_id"]
            idx = _id_to_idx.get(cid)
            result[cid] = _clamp(all_sims[idx]) if idx is not None else 0.0

    else:
        # ── SLOW PATH: on-the-fly per-candidate encoding (sandbox only) ──────
        # Only reached when embeddings.npy is missing (e.g., Streamlit Cloud).
        # Acceptable for small uploads; would be ~1.5hrs for 100K → never use
        # for the ranked stress test (always run precompute.py first).
        model = _get_model()
        for cand in candidates:
            cid  = cand["candidate_id"]
            text = _build_candidate_text(cand)
            emb  = model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            result[cid] = _clamp(float(np.dot(emb, _jd_embedding)))

    return result
