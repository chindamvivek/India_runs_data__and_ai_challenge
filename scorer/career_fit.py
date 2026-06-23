"""
scorer/career_fit.py
~~~~~~~~~~~~~~~~~~~~
Scores a candidate on how well their career trajectory matches the JD for
"Senior AI Engineer — Founding Team" at Redrob AI.

Weight in composite: 35%

Key design decisions (per implementation plan v4):
- Evaluated against full career_history, NOT just current_title
- Plain-language fits counted (e.g., "built recommendation engine" at a product co)
- Pure consulting-only careers penalised (0.5× multiplier)
- career_fit_score CAPPED at 1.0 BEFORE job-hopping penalty
- Job-hopping penalty applied after cap, then floored at 0.0
- Returns a scored_data dict (not just a float) so reasoning/generator.py
  can use the same intermediate values without recomputing them

Returns dict with keys:
    score           float [0.0, 1.0]
    ml_months       int   — total months in ML/AI/Search roles
    avg_tenure      float — avg months per completed role (0 if < 3 roles)
    consulting_only bool  — True if >80% career at consulting firms
    has_retrieval   bool  — True if retrieval/ranking keywords in any job description
    top_role        str   — best matching role title found in career history
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Titles that directly match what the JD is looking for
ML_TITLES = re.compile(
    r"\b("
    r"ml engineer|machine learning engineer|ai engineer|applied scientist|"
    r"applied ml|nlp engineer|search engineer|ranking engineer|"
    r"recommendation engineer|retrieval engineer|research engineer|"
    r"ai\/ml engineer|ml scientist|applied researcher"
    r")\b",
    re.IGNORECASE,
)

# Strong secondary: general SWE/backend/data at product companies with ML context
SWE_TITLES = re.compile(
    r"\b(software engineer|backend engineer|sde|senior engineer|"
    r"staff engineer|principal engineer|platform engineer|"
    r"data engineer|data scientist|research scientist)\b",
    re.IGNORECASE,
)

# Adjacent: data-adjacent roles that can score if descriptions contain ML work
DATA_TITLES = re.compile(
    r"\b(data engineer|analytics engineer|data analyst)\b",
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

# Retrieval/ranking keywords in job descriptions (plain-language fit signal)
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

# Companies explicitly named in JD as consulting/services firms to discount
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "mindtree",  # mindtree merged with LTI but still seen
}

# Industries that are clearly product/tech (not pure services)
PRODUCT_INDUSTRIES = {
    "software", "saas", "internet", "e-commerce", "ecommerce",
    "ai", "artificial intelligence", "fintech", "edtech", "healthtech",
    "technology", "tech", "product", "startup",
}


def _is_consulting_company(company: str, industry: str) -> bool:
    """Return True if the company is a known consulting/services firm."""
    c_lower = company.lower().strip()
    # Exact match on known firms
    for firm in CONSULTING_FIRMS:
        if firm in c_lower:
            return True
    # Industry-level signal: "IT Services" is the canonical consulting industry
    if "it services" in industry.lower():
        return True
    return False


def _is_product_company(company: str, industry: str) -> bool:
    """Return True if the company appears to be a product/tech company."""
    if _is_consulting_company(company, industry):
        return False
    ind_lower = industry.lower()
    for prod_ind in PRODUCT_INDUSTRIES:
        if prod_ind in ind_lower:
            return True
    return False


def _best_role_score(title: str, description: str, company: str, industry: str) -> float:
    """
    Score a single career role based on title + description + company type.
    Returns a base score in [0.0, 0.90].
    """
    # Hard disqualifier: title is clearly non-technical
    if NON_TECH_TITLES.search(title):
        return 0.0

    is_product = _is_product_company(company, industry)
    has_retrieval = bool(RETRIEVAL_KEYWORDS.search(description))
    has_ml_context = bool(ML_KEYWORDS_IN_DESC.search(description))

    # Tier 1: Direct ML/AI/Search title
    if ML_TITLES.search(title):
        return 0.90

    # Tier 2: SWE/Backend/Data Scientist at product company with ML context
    if SWE_TITLES.search(title):
        if is_product and (has_retrieval or has_ml_context):
            return 0.70
        elif is_product:
            return 0.35  # Product company SWE but no ML evidence
        elif has_retrieval or has_ml_context:
            return 0.40  # Consulting SWE but shows ML work
        else:
            return 0.25  # Generic SWE

    # Tier 3: Data Engineer at product company with retrieval keywords
    if DATA_TITLES.search(title):
        if is_product and has_retrieval:
            return 0.45
        elif is_product and has_ml_context:
            return 0.35
        return 0.20

    # Fallback: any other technical-sounding title
    if has_retrieval and is_product:
        return 0.35
    if has_ml_context and is_product:
        return 0.30

    return 0.15  # Any remaining technical role


def _is_ml_role(title: str, description: str) -> bool:
    """Return True if this role qualifies as an applied ML/AI role."""
    if ML_TITLES.search(title):
        return True
    # SWE at product company with retrieval/ML keywords counts
    if SWE_TITLES.search(title) and (
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
                        has_retrieval, top_role
    """
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])

    years_exp = float(profile.get("years_of_experience", 0))

    # ------------------------------------------------------------------
    # Hard disqualifier: experience out of valid range
    # ------------------------------------------------------------------
    if years_exp < 2.5 or years_exp > 18.0:
        return {
            "score": 0.0,
            "ml_months": 0,
            "avg_tenure": 0.0,
            "consulting_only": False,
            "has_retrieval": False,
            "top_role": profile.get("current_title", ""),
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

    # Check all roles including current
    all_roles_non_tech = True  # assume true, disprove below

    for job in career:
        title       = job.get("title", "")
        description = job.get("description", "")
        company     = job.get("company", "")
        industry    = job.get("industry", "")
        duration    = int(job.get("duration_months", 0))

        role_score = _best_role_score(title, description, company, industry)

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

        # Track retrieval signal
        if RETRIEVAL_KEYWORDS.search(description) or RETRIEVAL_KEYWORDS.search(title):
            has_retrieval = True

    # ------------------------------------------------------------------
    # Hard disqualifier: entire career is non-technical
    # ------------------------------------------------------------------
    if all_roles_non_tech or best_score == 0.0:
        return {
            "score": 0.0,
            "ml_months": ml_months,
            "avg_tenure": 0.0,
            "consulting_only": False,
            "has_retrieval": has_retrieval,
            "top_role": best_role_title,
        }

    # ------------------------------------------------------------------
    # Start from best role score
    # ------------------------------------------------------------------
    career_fit_score = best_score

    # ------------------------------------------------------------------
    # ML experience depth bonus
    # JD ideal: 4-6 years in applied ML at product companies
    # ------------------------------------------------------------------
    if 48 <= ml_months <= 72:
        career_fit_score += 0.15
    elif 36 <= ml_months < 48:
        career_fit_score += 0.10
    elif ml_months > 72:
        career_fit_score += 0.08
    elif 24 <= ml_months < 36:
        career_fit_score += 0.05

    # ------------------------------------------------------------------
    # Consulting-only multiplier
    # JD: "people who have only worked at consulting firms" are disqualified
    # ------------------------------------------------------------------
    consulting_fraction = (consulting_months / total_months) if total_months > 0 else 0.0
    consulting_only = consulting_fraction > 0.80
    if consulting_only:
        career_fit_score *= 0.5

    # ------------------------------------------------------------------
    # Fix 2: Cap at 1.0 BEFORE applying job-hopping penalty
    # base (0.90) + depth bonus (0.15) can reach 1.05 → cap first
    # ------------------------------------------------------------------
    career_fit_score = min(1.0, career_fit_score)

    # ------------------------------------------------------------------
    # Job-hopping penalty (v3 Fix 4 / v4 Fix 2 ordering)
    # JD: "switching companies every 1.5 years" is explicit disqualifier
    # ------------------------------------------------------------------
    completed_roles = [
        j for j in career
        if not j.get("is_current", False) and int(j.get("duration_months", 0)) > 0
    ]
    avg_tenure = 0.0
    if len(completed_roles) >= 3:
        avg_tenure = sum(int(j["duration_months"]) for j in completed_roles) / len(completed_roles)
        if avg_tenure < 12:
            career_fit_score -= 0.20   # Severe: < 1 yr per role
        elif avg_tenure < 18:
            career_fit_score -= 0.12   # JD's explicit 1.5-year threshold

    # ------------------------------------------------------------------
    # Fix 2: Floor at 0.0 AFTER penalty
    # ------------------------------------------------------------------
    career_fit_score = max(0.0, career_fit_score)

    return {
        "score":          career_fit_score,
        "ml_months":      ml_months,
        "avg_tenure":     avg_tenure,
        "consulting_only": consulting_only,
        "has_retrieval":  has_retrieval,
        "top_role":       best_role_title,
    }
