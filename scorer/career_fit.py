"""
scorer/career_fit.py
~~~~~~~~~~~~~~~~~~~~
Scores a candidate on how well their career trajectory matches the JD for
"Senior AI Engineer — Founding Team" at Redrob AI.

Weight in composite: 35%

Design (v5 — Tiered Role Classification):

Three improvements over v4:

  1. Product company detection (Bug fix)
     The v4 implementation used plain substring matching against PRODUCT_INDUSTRIES,
     which incorrectly classified "Paper Products" as a tech company because
     "product" is a substring of "paper products".
     Replaced with a word-boundary regex (TECH_PRODUCT_INDUSTRIES) against a
     curated list of tech/startup industry terms. The generic word "product"
     has been removed.

  2. Description-based specialisation override (New)
     Generic titles like "Software Engineer" are cross-checked against the job
     description. If the description clearly signals QA, frontend, or mobile
     engineering work, the role is reclassified to Tier D regardless of title.
     QA_DESC_SIGNALS, FRONTEND_DESC_SIGNALS, and MOBILE_DESC_SIGNALS detect
     these specialisations.

  3. Four-tier SWE classification (Improved granularity)
     Tier A — Direct ML/AI/Search/NLP titles         base 0.85 – 0.90
     Tier B — Backend/Platform/Data engineers         base 0.20 – 0.65
     Tier C — Generic SWE/Full-stack/Cloud/DevOps     base 0.15 – 0.50
     Tier D — Frontend/Mobile/QA engineers            base 0.05 – 0.20
     (v4 merged Tier B/C/D into one SWE bucket)

  4. Per-role explainability (New)
     Returns a ``role_breakdown`` list (one dict per career entry) so that
     diagnostics can show exactly which signals fired and why.

Returns dict with keys (unchanged from v4, plus role_breakdown):
    score           float [0.0, 1.0]
    ml_months       int   — total months in ML/AI/Search roles
    avg_tenure      float — avg months per completed role (0 if < 3 roles)
    consulting_only bool  — True if >80% career at consulting firms
    has_retrieval   bool  — True if retrieval/ranking keywords in any job desc
    top_role        str   — best matching role title found in career history
    role_breakdown  list  — per-role debug dicts (new in v5)
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Tier A — Direct ML / AI / Search specialists
# ---------------------------------------------------------------------------
TIER_A_TITLES = re.compile(
    r"\b("
    r"ml engineer|machine learning engineer|ai engineer|applied scientist|"
    r"applied ml|nlp engineer|search engineer|ranking engineer|"
    r"recommendation engineer|retrieval engineer|research engineer|"
    r"ai\/ml engineer|ml scientist|applied researcher|"
    r"computer vision engineer|speech engineer|conversational ai"
    r")\b",
    re.IGNORECASE,
)

# Tier B — Backend / Platform / Data roles (strong SWE depth, ML-adjacent)
TIER_B_TITLES = re.compile(
    r"\b("
    r"backend engineer|back-end engineer|back end engineer|"
    r"platform engineer|distributed systems engineer|"
    r"infrastructure engineer|systems engineer|"
    r"staff engineer|principal engineer|senior engineer|"  # seniority only, not enough alone
    r"data engineer|data scientist|analytics engineer|"
    r"research scientist|ml ops engineer|mlops engineer|"
    r"site reliability engineer|sre|devops engineer|cloud engineer"
    r")\b",
    re.IGNORECASE,
)

# Tier C — Generic SWE / Full-stack / language-specific titles
TIER_C_TITLES = re.compile(
    r"\b("
    r"software engineer|software developer|sde|sde-\d|"
    r"full stack|full-stack|fullstack|"
    r"java developer|python developer|\.net developer|net developer|"
    r"golang developer|ruby developer|c\+\+ developer|"
    r"firmware engineer|embedded engineer"
    r")\b",
    re.IGNORECASE,
)

# Tier D — Frontend / Mobile / QA (lowest relevance to AI engineering)
TIER_D_TITLES = re.compile(
    r"\b("
    r"frontend engineer|front-end engineer|front end engineer|"
    r"ui engineer|ui developer|"
    r"mobile developer|mobile engineer|"
    r"ios developer|ios engineer|android developer|android engineer|"
    r"qa engineer|quality assurance engineer|sdet|test engineer|"
    r"test automation engineer|automation engineer"
    r")\b",
    re.IGNORECASE,
)

# Hard disqualifiers: non-technical career tracks
NON_TECH_TITLES = re.compile(
    r"\b("
    r"marketing manager|operations manager|operation manager|"
    r"hr manager|human resources|accountant|finance|"
    r"civil engineer|mechanical engineer|content writer|"
    r"graphic designer|brand designer|sales|business development|"
    r"customer support|customer success|account manager|"
    r"project manager|scrum master|product manager"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Description-based specialisation signals
# Used to reclassify ambiguous titles (e.g. "Software Engineer") to Tier D
# when the description reveals QA, frontend, or mobile work.
# ---------------------------------------------------------------------------
QA_DESC_SIGNALS = re.compile(
    r"\b("
    r"selenium|cypress|pytest|unittest|jest|mocha|"
    r"test automation|qa engineering|quality assurance|test suite|"
    r"acceptance criteria|testability|locust|load test|load-test|"
    r"end-to-end test|e2e test|regression test|manual test|"
    r"functional test|test coverage|sdet"
    r")\b",
    re.IGNORECASE,
)

FRONTEND_DESC_SIGNALS = re.compile(
    r"\b("
    r"react|vue|angular|angularjs|html|css|webpack|"
    r"design system|animation|accessibility|"
    r"dom|sass|less|responsive design|single.page.app|"
    r"nextjs|next\.js|gatsby|svelte|storybook|"
    r"ui component|browser compatibility|tailwind"
    r")\b",
    re.IGNORECASE,
)

MOBILE_DESC_SIGNALS = re.compile(
    r"\b("
    r"android|ios|swift|kotlin|flutter|react native|"
    r"jetpack|xcode|objective-c|coroutines|hilt|"
    r"app store|play store|push notification|offline.first|"
    r"mobile app|mvvm|viewmodel"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Retrieval / ML keyword signals (job description context)
# ---------------------------------------------------------------------------
RETRIEVAL_KEYWORDS = re.compile(
    r"\b("
    r"retrieval|ranking|recommendation|search|vector|embedding|"
    r"similarity|semantic|faiss|pinecone|qdrant|milvus|weaviate|"
    r"elasticsearch|opensearch|bm25|dense retrieval|hybrid search|"
    r"candidate retrieval|information retrieval|re-rank|rerank|"
    r"ranker|indexing|inverted index"
    r")\b",
    re.IGNORECASE,
)

ML_KEYWORDS_IN_DESC = re.compile(
    r"\b("
    r"machine learning|deep learning|neural network|nlp|llm|"
    r"transformer|bert|gpt|fine-tun|embeddings|model training|"
    r"inference|model serving|mlops|feature engineering|"
    r"a/b test|evaluation|ndcg|mrr|map\b|precision|recall"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------

# Known consulting / services firms to discount
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "mindtree",
}

# Word-boundary regex against curated tech/startup industry terms.
# Deliberately excludes the generic word "product" to avoid matching
# "Paper Products", "Consumer Products", "Food Products", etc.
TECH_PRODUCT_INDUSTRIES = re.compile(
    r"\b("
    r"software|saas|internet|e-commerce|ecommerce|"
    r"fintech|edtech|healthtech|adtech|proptech|insurtech|legaltech|"
    r"agritech|foodtech|logistech|"
    r"artificial intelligence|machine learning|deep learning|"
    r"technology|startup|"
    r"social media|cloud computing|cloud services|gaming|"
    r"cybersecurity|information security|"
    r"media technology|streaming|semiconductor|"
    r"search engine|e-learning"
    r")\b",
    re.IGNORECASE,
)


def _is_consulting_company(company: str, industry: str) -> bool:
    """Return True if the company is a known consulting/services firm."""
    c_lower = company.lower().strip()
    for firm in CONSULTING_FIRMS:
        if firm in c_lower:
            return True
    if "it services" in industry.lower():
        return True
    return False


def _is_product_company(company: str, industry: str) -> bool:
    """
    Return True if the company is a technology product company.

    Uses word-boundary matching against TECH_PRODUCT_INDUSTRIES.
    Consulting firms are always False regardless of industry label.
    """
    if _is_consulting_company(company, industry):
        return False
    return bool(TECH_PRODUCT_INDUSTRIES.search(industry))


def _detect_desc_specialisation(description: str) -> str | None:
    """
    Infer the actual specialisation from job description text.

    Returns one of: "qa", "frontend", "mobile", or None (ambiguous).
    Used to reclassify generic titles to Tier D when the description
    reveals a non-AI-adjacent specialisation.
    """
    signals = {
        "qa":       bool(QA_DESC_SIGNALS.search(description)),
        "frontend": bool(FRONTEND_DESC_SIGNALS.search(description)),
        "mobile":   bool(MOBILE_DESC_SIGNALS.search(description)),
    }
    # Return first detected specialisation (priority: qa > mobile > frontend)
    for spec in ("qa", "mobile", "frontend"):
        if signals[spec]:
            return spec
    return None


def _score_role(
    title: str,
    description: str,
    company: str,
    industry: str,
) -> tuple[float, dict]:
    """
    Score a single career role. Returns (base_score, debug_dict).

    The debug_dict contains:
        tier        str   — A / B / C / D / non-tech / fallback
        title_match str   — which regex matched the title
        desc_spec   str   — description specialisation override (or None)
        is_product  bool
        is_consulting bool
        has_retrieval bool
        has_ml_context bool
        score       float
        reason      str   — human-readable explanation
    """
    is_product    = _is_product_company(company, industry)
    is_consulting = _is_consulting_company(company, industry)
    has_retrieval = bool(RETRIEVAL_KEYWORDS.search(description))
    has_ml_ctx    = bool(ML_KEYWORDS_IN_DESC.search(description))
    desc_spec     = _detect_desc_specialisation(description)

    debug = {
        "is_product":    is_product,
        "is_consulting": is_consulting,
        "has_retrieval": has_retrieval,
        "has_ml_context": has_ml_ctx,
        "desc_spec":     desc_spec,   # "qa", "frontend", "mobile", or None
    }

    # ── Hard disqualifier ────────────────────────────────────────────────
    if NON_TECH_TITLES.search(title):
        m = NON_TECH_TITLES.search(title).group(0)
        debug.update(tier="non-tech", title_match=m,
                     score=0.0, reason=f"Non-technical title: '{m}'")
        return 0.0, debug

    # ── Tier A — Direct ML / AI / Search title ───────────────────────────
    if TIER_A_TITLES.search(title):
        m = TIER_A_TITLES.search(title).group(0)
        score = 0.90 if has_retrieval else 0.85
        reason = (
            f"Tier A: ML/AI/Search title '{m}'"
            + ("; +retrieval keywords in desc" if has_retrieval else "")
        )
        debug.update(tier="A", title_match=m, score=score, reason=reason)
        return score, debug

    # ── Tier B — Backend / Platform / Data ───────────────────────────────
    if TIER_B_TITLES.search(title):
        m = TIER_B_TITLES.search(title).group(0)

        # Description override: title is "Data Scientist" but work is actually QA?
        # (Rare, but defensive.)
        if desc_spec in ("qa", "frontend", "mobile"):
            score = 0.10 if is_product else 0.05
            reason = (
                f"Tier B title '{m}' overridden to Tier D by description "
                f"({desc_spec} signals detected)"
            )
            debug.update(tier="B→D", title_match=m, score=score, reason=reason)
            return score, debug

        if is_product and has_retrieval:
            score, reason = 0.65, f"Tier B: '{m}' at product co with retrieval keywords"
        elif is_product and has_ml_ctx:
            score, reason = 0.55, f"Tier B: '{m}' at product co with ML context"
        elif is_product:
            score, reason = 0.30, f"Tier B: '{m}' at product co (no ML evidence)"
        elif has_retrieval or has_ml_ctx:
            score, reason = 0.35, f"Tier B: '{m}' at non-product co with ML/retrieval"
        else:
            score, reason = 0.20, f"Tier B: '{m}' generic (no product/ML evidence)"

        debug.update(tier="B", title_match=m, score=score, reason=reason)
        return score, debug

    # ── Tier C — Generic SWE / Full-stack / Language-specific ────────────
    if TIER_C_TITLES.search(title):
        m = TIER_C_TITLES.search(title).group(0)

        # Description override: "Software Engineer" doing QA/frontend/mobile work
        if desc_spec in ("qa", "frontend", "mobile"):
            score = 0.10 if is_product else 0.05
            reason = (
                f"Tier C title '{m}' overridden to Tier D by description "
                f"({desc_spec} signals detected)"
            )
            debug.update(tier="C→D", title_match=m, score=score, reason=reason)
            return score, debug

        if is_product and has_retrieval:
            score, reason = 0.50, f"Tier C: '{m}' at product co with retrieval keywords"
        elif is_product and has_ml_ctx:
            score, reason = 0.40, f"Tier C: '{m}' at product co with ML context"
        elif is_product:
            score, reason = 0.25, f"Tier C: '{m}' at product co (no ML evidence)"
        elif has_retrieval or has_ml_ctx:
            score, reason = 0.30, f"Tier C: '{m}' at non-product co with ML/retrieval"
        else:
            score, reason = 0.15, f"Tier C: '{m}' generic SWE"

        debug.update(tier="C", title_match=m, score=score, reason=reason)
        return score, debug

    # ── Tier D — Frontend / Mobile / QA ──────────────────────────────────
    if TIER_D_TITLES.search(title):
        m = TIER_D_TITLES.search(title).group(0)

        if is_product and (has_retrieval or has_ml_ctx):
            score, reason = 0.20, (
                f"Tier D: '{m}' at product co with ML/retrieval (unusual for this tier)"
            )
        elif is_product:
            score, reason = 0.10, f"Tier D: '{m}' at product co"
        else:
            score, reason = 0.05, f"Tier D: '{m}' — low-relevance specialisation"

        debug.update(tier="D", title_match=m, score=score, reason=reason)
        return score, debug

    # ── Fallback: unrecognised technical-sounding title ───────────────────
    if has_retrieval and is_product:
        score, reason = 0.35, "Fallback: unrecognised title with retrieval+product signal"
    elif has_ml_ctx and is_product:
        score, reason = 0.30, "Fallback: unrecognised title with ML+product signal"
    else:
        score, reason = 0.10, "Fallback: unrecognised title, no ML evidence"

    debug.update(tier="fallback", title_match="", score=score, reason=reason)
    return score, debug


def _is_ml_role(title: str, description: str) -> bool:
    """Return True if this role qualifies as an applied ML/AI role."""
    if TIER_A_TITLES.search(title):
        return True
    # Tier B at product company with retrieval/ML keywords counts
    if TIER_B_TITLES.search(title) and (
        RETRIEVAL_KEYWORDS.search(description) or ML_KEYWORDS_IN_DESC.search(description)
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_career_fit(candidate: dict) -> dict:
    """
    Compute the career fit score and intermediate features for a candidate.

    Returns:
        dict with keys: score, ml_months, avg_tenure, consulting_only,
                        has_retrieval, top_role, role_breakdown
    """
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])

    years_exp = float(profile.get("years_of_experience", 0))

    # ------------------------------------------------------------------
    # Hard disqualifier: experience out of valid range
    # ------------------------------------------------------------------
    if years_exp < 2.5 or years_exp > 18.0:
        return {
            "score":          0.0,
            "technical_score": 0.0,
            "career_score":   0.0,
            "ml_months":      0,
            "avg_tenure":     0.0,
            "consulting_only": False,
            "has_retrieval":  False,
            "top_role":       profile.get("current_title", ""),
            "role_breakdown": [],
        }

    # ------------------------------------------------------------------
    # Score each role; track best score and ML months
    # ------------------------------------------------------------------
    best_score      = 0.0
    best_role_title = profile.get("current_title", "")
    ml_months       = 0
    total_months    = 0
    consulting_months = 0
    has_retrieval   = False
    all_roles_non_tech = True  # assume true, disprove below
    role_breakdown  = []

    for job in career:
        title       = job.get("title", "")
        description = job.get("description", "")
        company     = job.get("company", "")
        industry    = job.get("industry", "")
        duration    = int(job.get("duration_months", 0))

        role_score, debug = _score_role(title, description, company, industry)

        debug["title"]    = title
        debug["company"]  = company
        debug["industry"] = industry
        debug["duration_months"] = duration

        role_breakdown.append(debug)

        # If any role has score > 0, not all non-tech
        if role_score > 0:
            all_roles_non_tech = False

        if role_score > best_score:
            best_score = role_score
            best_role_title = title

        # Track ML months
        if _is_ml_role(title, description):
            ml_months += duration

        # Track consulting months
        if _is_consulting_company(company, industry):
            consulting_months += duration

        total_months += duration

        # Track retrieval signal (at any tier)
        if RETRIEVAL_KEYWORDS.search(description) or RETRIEVAL_KEYWORDS.search(title):
            has_retrieval = True

    # ------------------------------------------------------------------
    # Hard disqualifier: entire career is non-technical
    # ------------------------------------------------------------------
    if all_roles_non_tech or best_score == 0.0:
        return {
            "score":          0.0,
            "technical_score": 0.0,
            "career_score":   0.0,
            "ml_months":      ml_months,
            "avg_tenure":     0.0,
            "consulting_only": False,
            "has_retrieval":  has_retrieval,
            "top_role":       best_role_title,
            "role_breakdown": role_breakdown,
        }

    # ------------------------------------------------------------------
    # TECHNICAL SCORE: technical relevance (Base + Depth Bonus)
    # ------------------------------------------------------------------
    technical_score = best_score

    # ML experience depth bonus
    if 48 <= ml_months <= 72:
        technical_score += 0.15
    elif 36 <= ml_months < 48:
        technical_score += 0.10
    elif ml_months > 72:
        technical_score += 0.08
    elif 24 <= ml_months < 36:
        technical_score += 0.05

    # Cap technical score at 1.0
    technical_score = min(1.0, technical_score)

    # ------------------------------------------------------------------
    # CAREER SCORE: Starts from technical_score, applies trajectory penalties
    # ------------------------------------------------------------------
    career_score = technical_score

    # Consulting-only multiplier
    consulting_fraction = (consulting_months / total_months) if total_months > 0 else 0.0
    consulting_only = consulting_fraction > 0.80
    if consulting_only:
        career_score *= 0.5

    # Job-hopping penalty
    completed_roles = [
        j for j in career
        if not j.get("is_current", False) and int(j.get("duration_months", 0)) > 0
    ]
    avg_tenure = 0.0
    if len(completed_roles) >= 3:
        avg_tenure = sum(int(j["duration_months"]) for j in completed_roles) / len(completed_roles)
        if avg_tenure < 12:
            career_score -= 0.20   # Severe: < 1 yr per role
        elif avg_tenure < 18:
            career_score -= 0.12   # JD's explicit 1.5-year threshold

    # Floor career score at 0.0
    career_score = max(0.0, career_score)

    return {
        "score":          career_score,  # Keep for backward compatibility
        "technical_score": technical_score,
        "career_score":   career_score,
        "ml_months":      ml_months,
        "avg_tenure":     avg_tenure,
        "consulting_only": consulting_only,
        "has_retrieval":  has_retrieval,
        "top_role":       best_role_title,
        "role_breakdown": role_breakdown,
    }
