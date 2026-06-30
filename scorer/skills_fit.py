"""
scorer/skills_fit.py
~~~~~~~~~~~~~~~~~~~~
Scores a candidate's skills against the JD requirements using a trust-weighted
formula rather than raw keyword counting.

Weight in composite: 25%

Trust formula per skill:
    skill_trust_score = base_weight
                        × proficiency_factor
                        × duration_factor        (capped at 1.5×)
                        × endorsement_factor      (capped at 1.5×)

Normalization (Fix 1 from v4 plan):
    skills_fit_score = min(raw_score / MAX_PLAUSIBLE_SKILL_SCORE, 1.0)

    MAX_PLAUSIBLE_SKILL_SCORE is a hardcoded constant (not min-max normalization)
    so scores are stable and comparable across any subset of candidates — critical
    because Stage 3 sandbox uses a different sample than the full 100K pool.

LangChain penalty (v3 Fix 1):
    Only applied if LangChain is present AND no production retrieval skills exist.
    A senior engineer with LangChain + FAISS gets zero penalty.

Returns:
    dict with keys:
        score       float [0.0, 1.0]
        top_skills  list[str]  — top 2 skill names by trust score (for reasoning)
        raw_score   float      — raw sum before normalization (useful for debugging)
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Normalisation ceiling (Fix 1 — empirically tuned from sample_candidates.json)
# A "perfect" candidate with expert retrieval skills (FAISS, embeddings, ranking,
# Python, NDCG) each at 36+ months and 30 endorsements → raw ~14-16.
# Set ceiling at 15.0 with ~20% headroom above the expected max.
# Adjust this constant by running scorer on sample_candidates.json and inspecting
# raw score distribution before final submission.
# ---------------------------------------------------------------------------
MAX_PLAUSIBLE_SKILL_SCORE: float = 15.0

# ---------------------------------------------------------------------------
# Proficiency factor
# ---------------------------------------------------------------------------
PROFICIENCY_FACTOR: dict[str, float] = {
    "beginner":    0.40,
    "intermediate": 0.70,
    "advanced":    0.90,
    "expert":      1.00,
}

# ---------------------------------------------------------------------------
# Skill weight dictionary
# Key: lowercase skill name (or partial match pattern — see _match_weight)
# Value: base weight [−0.10, 1.00]
# ---------------------------------------------------------------------------
SKILL_WEIGHTS: dict[str, float] = {
    # ── Tier 1: Must-haves (JD: "absolutely need") ──────────────────────────
    "embeddings":              1.00,
    "sentence-transformers":   1.00,
    "sentence transformers":   1.00,
    "vector database":         1.00,
    "vector search":           0.95,
    "hybrid search":           0.95,
    "dense retrieval":         0.95,
    "retrieval":               0.90,
    "ranking":                 0.90,
    "information retrieval":   0.90,
    "re-ranking":              0.90,
    "reranking":               0.90,
    "pinecone":                0.90,
    "qdrant":                  0.90,
    "milvus":                  0.90,
    "weaviate":                0.90,
    "faiss":                   0.90,
    "elasticsearch":           0.85,
    "opensearch":              0.85,
    "bm25":                    0.85,
    "ndcg":                    0.90,
    "mrr":                     0.90,
    "map":                     0.85,
    "mean average precision":  0.85,
    "a/b testing":             0.85,
    "a/b test":                0.85,
    "python":                  0.80,
    "evaluation framework":    0.80,
    "offline evaluation":      0.80,

    # ── Tier 2: Strong positives (JD: "like to have") ───────────────────────
    "lora":                    0.75,
    "qlora":                   0.75,
    "peft":                    0.75,
    "fine-tuning":             0.75,
    "fine-tuning llms":        0.75,
    "fine tuning":             0.70,
    "rag":                     0.75,
    "retrieval augmented":     0.75,
    "learning to rank":        0.80,
    "lambdamart":              0.80,
    "xgboost":                 0.65,
    "transformers":            0.70,
    "huggingface":             0.70,
    "hugging face":            0.70,
    "bert":                    0.65,
    "llm":                     0.65,
    "large language model":    0.65,
    "gpt":                     0.60,
    "mlflow":                  0.55,
    "mlops":                   0.60,

    # ── Tier 3: Adjacent relevant (ML/data infra) ────────────────────────────
    "pytorch":                 0.55,
    "tensorflow":              0.50,
    "scikit-learn":            0.45,
    "sklearn":                 0.45,
    "spark":                   0.40,
    "apache spark":            0.40,
    "pyspark":                 0.40,
    "airflow":                 0.40,
    "kafka":                   0.40,
    "docker":                  0.30,
    "kubernetes":              0.30,
    "feature engineering":     0.45,
    "recommendation":          0.55,
    "recommendation system":   0.60,
    "search":                  0.50,

    # ── Tier 4: Generic / neutral ────────────────────────────────────────────
    "sql":                     0.20,
    "aws":                     0.20,
    "gcp":                     0.20,
    "azure":                   0.20,
    "git":                     0.15,
    "docker":                  0.25,
    "linux":                   0.15,

    # ── Negative signals (JD "DO NOT want") ─────────────────────────────────
    "photoshop":              -0.05,
    "illustrator":            -0.05,
    "content writing":        -0.10,
    "seo":                    -0.08,
    "six sigma":              -0.08,
    "cad":                    -0.05,
    "solidworks":             -0.05,
    "marketing":              -0.05,

    # ── LangChain: contextual (see langchain_penalty below) ─────────────────
    # Handled separately — NOT in this dict
}

# Production retrieval skills — if any of these are present, no LangChain penalty
PRODUCTION_RETRIEVAL_SKILLS: frozenset[str] = frozenset({
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "elasticsearch",
    "opensearch", "embeddings", "sentence-transformers", "sentence transformers",
    "hybrid search", "dense retrieval", "vector database", "vector search",
    "bm25", "information retrieval",
})


def _match_weight(skill_name: str) -> float | None:
    """
    Look up the base weight for a skill name.
    Returns None if the skill has no entry (neutral — contributes 0).
    Does exact lowercase match first, then substring containment.
    """
    name_lower = skill_name.lower().strip()

    # Exact match
    if name_lower in SKILL_WEIGHTS:
        return SKILL_WEIGHTS[name_lower]

    # Substring: check if any known key is a substring of the skill name
    # e.g. "Fine-tuning LLMs" matches "fine-tuning"
    for key, weight in SKILL_WEIGHTS.items():
        if key in name_lower or name_lower in key:
            return weight

    return None  # unknown skill → contributes 0 (not negative)


def _langchain_penalty(skill_names_lower: set[str]) -> float:
    """
    Penalize LangChain-only tutorial profiles.

    JD: "Framework enthusiasts whose GitHub is full of LangChain tutorials
    and blog posts about 'How I used [hot framework] to build [demo]' —
    that's fine but it's not what we need."

    Penalty tiers:
      - LangChain present + no production retrieval skills: -0.20
        (near-disqualifier per JD; doubled from v3's -0.10)
      - LangChain present + production retrieval skills:      0.00
        (legitimate: using LangChain as orchestration over real infra)
    """
    has_langchain = any("langchain" in s for s in skill_names_lower)
    if not has_langchain:
        return 0.0

    has_production_retrieval = bool(skill_names_lower & PRODUCTION_RETRIEVAL_SKILLS)
    if has_production_retrieval:
        return 0.0   # LangChain + real retrieval skills -> no penalty

    return -0.20     # Framework enthusiast with no production retrieval experience


def compute_skills_fit(candidate: dict) -> dict:
    """
    Compute the skills fit score for a candidate.

    Returns:
        dict with keys: score, top_skills, raw_score
    """
    skills = candidate.get("skills", [])

    if not skills:
        return {"score": 0.0, "top_skills": [], "raw_score": 0.0}

    skill_names_lower = {s.get("name", "").lower() for s in skills}

    trust_scores: list[tuple[float, str]] = []   # (trust_score, skill_name)
    raw_total = 0.0

    for skill in skills:
        name        = skill.get("name", "")
        proficiency = skill.get("proficiency", "beginner")
        duration    = int(skill.get("duration_months", 0))
        endorsements = int(skill.get("endorsements", 0))

        base_weight = _match_weight(name)
        if base_weight is None:
            continue   # Unknown skill — ignore (neither positive nor negative)

        if base_weight < 0:
            # Negative skills: apply directly without the trust multipliers
            raw_total += base_weight
            trust_scores.append((base_weight, name))
            continue

        prof_factor = PROFICIENCY_FACTOR.get(proficiency, 0.40)
        dur_factor  = min(duration / 24.0, 1.5) if duration > 0 else 0.1  # tiny credit for 0-month
        end_factor  = min(1.0 + (endorsements / 30.0), 1.5)

        trust = base_weight * prof_factor * dur_factor * end_factor
        raw_total += trust
        trust_scores.append((trust, name))

    # LangChain contextual penalty
    lc_penalty = _langchain_penalty(skill_names_lower)
    raw_total += lc_penalty

    # Fixed-cap normalization (Fix 1)
    normalised = min(raw_total / MAX_PLAUSIBLE_SKILL_SCORE, 1.0)
    normalised = max(normalised, 0.0)   # floor at 0 in case negatives dominate

    # Top-2 skills by trust score (positive only, for reasoning)
    positive_trust = [(t, n) for t, n in trust_scores if t > 0]
    positive_trust.sort(reverse=True)
    top_skills = [n for _, n in positive_trust[:2]]

    return {
        "score":      normalised,
        "top_skills": top_skills,
        "raw_score":  raw_total,
    }
