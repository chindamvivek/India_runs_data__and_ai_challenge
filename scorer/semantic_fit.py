"""
scorer/semantic_fit.py
~~~~~~~~~~~~~~~~~~~~~~
Scores a candidate using semantic similarity between their profile text and
the Job Description, using sentence-transformers (all-MiniLM-L6-v2).

Weight in composite: 12%

Design (per implementation plan v4 Fix 3):
  - Pre-computed embeddings (.npy + id_index.json) are loaded at startup
    for the full 100K pool. Cosine similarity is a single matrix multiply (<1s).
  - For any candidate ID NOT found in the pre-computed array (e.g., Stage 3
    sandbox uses a different sample), the model encodes that candidate on-the-fly.
    For ≤100 candidates this takes ~2 seconds.
  - The model is loaded lazily — only when on-the-fly encoding is needed.

JD query text captures the semantic concepts the JD actually cares about,
beyond keywords: "production ranking systems", "shipping under constraints", etc.

Returns:
    float [0.0, 1.0] — cosine similarity to the JD embedding
"""

from __future__ import annotations
import json
import os
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
# File paths for pre-computed embeddings
# ---------------------------------------------------------------------------
_DEFAULT_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "embeddings.npy"
)
_DEFAULT_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "id_index.json"
)

# ---------------------------------------------------------------------------
# Module-level cache — loaded once per process
# ---------------------------------------------------------------------------
_embeddings_matrix: np.ndarray | None = None   # shape (N, 384)
_id_to_row: dict[str, int] | None = None        # candidate_id → row index
_jd_embedding: np.ndarray | None = None         # shape (384,)
_model = None                                    # sentence-transformers model (lazy)


def _load_precomputed(
    embeddings_path: str = _DEFAULT_EMBEDDINGS_PATH,
    index_path: str = _DEFAULT_INDEX_PATH,
) -> bool:
    """Load pre-computed embeddings and id index. Returns True if successful."""
    global _embeddings_matrix, _id_to_row
    if _embeddings_matrix is not None:
        return True   # Already loaded

    if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
        return False  # Pre-computed files not available

    _embeddings_matrix = np.load(embeddings_path).astype(np.float32)
    with open(index_path, "r", encoding="utf-8") as f:
        _id_to_row = json.load(f)
    return True


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


def _build_candidate_text(candidate: dict) -> str:
    """Assemble a single text string representing the candidate's profile."""
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])

    headline = profile.get("headline", "")
    summary  = profile.get("summary", "")

    # Use the 3 most recent job descriptions (most recent = most relevant)
    job_descs = " ".join(
        j.get("description", "") for j in career[:3]
    )

    # Include top skill names to help semantic matching
    skill_names = " ".join(s.get("name", "") for s in skills[:15])

    return f"{headline} {summary} {job_descs} {skill_names}".strip()


def initialise(
    embeddings_path: str = _DEFAULT_EMBEDDINGS_PATH,
    index_path: str = _DEFAULT_INDEX_PATH,
) -> None:
    """
    Pre-load everything at startup so the first score() call is fast.
    Call this once from rank.py before the scoring loop.
    """
    _load_precomputed(embeddings_path, index_path)
    # Always pre-compute the JD embedding (needs the model regardless)
    _get_jd_embedding()


def compute_all_semantic_scores(
    candidates: list[dict],
    embeddings_path: str = _DEFAULT_EMBEDDINGS_PATH,
    index_path: str = _DEFAULT_INDEX_PATH,
) -> dict[str, float]:
    """
    Compute semantic scores for all candidates efficiently.

    Strategy:
      1. For candidates found in the pre-computed array → matrix multiply
      2. For candidates NOT in the array → encode on-the-fly in one batch

    Returns:
        dict mapping candidate_id → semantic_score (float in [0.0, 1.0])
    """
    _load_precomputed(embeddings_path, index_path)
    jd_emb = _get_jd_embedding()

    scores: dict[str, float] = {}
    missing_ids: list[str] = []
    missing_texts: list[str] = []

    for cand in candidates:
        cid = cand["candidate_id"]
        if _id_to_row is not None and cid in _id_to_row:
            # Use pre-computed embedding
            row = _id_to_row[cid]
            emb = _embeddings_matrix[row]  # already L2-normalised
            score = float(np.dot(emb, jd_emb))
            scores[cid] = max(0.0, min(1.0, score))
        else:
            # Will encode on-the-fly in a batch below
            missing_ids.append(cid)
            missing_texts.append(_build_candidate_text(cand))

    if missing_ids:
        model = _get_model()
        batch_embs = model.encode(
            missing_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(missing_ids) > 200,
        ).astype(np.float32)

        batch_scores = batch_embs @ jd_emb  # shape (N,)
        for cid, score in zip(missing_ids, batch_scores.tolist()):
            scores[cid] = max(0.0, min(1.0, float(score)))

    return scores
