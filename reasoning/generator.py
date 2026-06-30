"""
reasoning/generator.py
~~~~~~~~~~~~~~~~~~~~~~
Deterministic, rank-aware reasoning string builder for the top-100 candidates.

Design principles (per implementation plan v4):
  - All intermediate values (ml_months, avg_tenure, consulting_only, has_retrieval,
    top_skills, days_inactive, country) come from the pre-computed scored_data dict.
    They are NEVER recomputed here — this guarantees the explanation can never
    contradict the rank. (Fix 4)

  - Concerns are ALWAYS shown if genuine, regardless of rank. Tone is softened
    for top-10 ("minor consideration") but the fact is never hidden. (Fix 3)

  - Every sentence references specific facts from the candidate's actual data
    so reviewers cannot find a claim that isn't grounded in the profile.

  - The opener, strength, and concern phrases all vary based on the candidate's
    actual data — no two candidates in a typical run should get identical strings.

Public API:
    build_reasoning(candidate, scored_data, rank) -> str
"""

from __future__ import annotations


def _format_yrs(months: int) -> str:
    """Convert months to a human-readable years string."""
    yrs = months // 12
    return f"{yrs}+" if months % 12 >= 6 else str(yrs)


def build_reasoning(candidate: dict, scored_data: dict, rank: int) -> str:
    """
    Build a 1–2 sentence, fact-grounded reasoning string for a ranked candidate.

    Args:
        candidate:   Raw candidate dict from JSONL / JSON.
        scored_data: Merged dict from all scorers, must contain:
                       ml_months      int
                       avg_tenure     float
                       consulting_only bool
                       has_retrieval  bool
                       top_skills     list[str]
                       days_inactive  int
                       country        str
                       top_role       str   (best matching role title)
        rank:        Final assigned rank (1–100).

    Returns:
        A single string, 1–2 sentences, suitable for the CSV reasoning column.
    """
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})

    # ── Read from profile ────────────────────────────────────────────────────
    yrs              = float(profile.get("years_of_experience", 0))
    current_title    = profile.get("current_title", "professional")
    location         = profile.get("location", "unknown location")

    # ── Read from scored_data (computed once during scoring, never recomputed) ──
    ml_months        = int(scored_data.get("ml_months", 0))
    avg_tenure       = float(scored_data.get("avg_tenure", 0.0))
    consulting_only  = bool(scored_data.get("consulting_only", False))
    academic_only    = bool(scored_data.get("academic_only", False))
    has_retrieval    = bool(scored_data.get("has_retrieval", False))
    top_skills       = list(scored_data.get("top_skills", []))
    days_inactive    = int(scored_data.get("days_inactive", 0))
    country          = str(scored_data.get("country", ""))
    top_role         = str(scored_data.get("top_role", current_title))

    # ── Read from signals ────────────────────────────────────────────────────
    notice_days      = int(signals.get("notice_period_days", 90))
    open_to_work     = bool(signals.get("open_to_work_flag", False))

    # ── Read technical_score for tone-matching (NOT career_score) ────────────
    # career_score may be reduced by consulting/hopping penalties, which would
    # cause a real Backend Engineer to be labeled "non-technical".  The
    # technical_score is the pure, penalty-free tier score and is the correct
    # signal for whether the candidate is an engineer at all.
    technical_score = float(scored_data.get("technical_score", 0.0))

    # ────────────────────────────────────────────────────────────────────────
    # OPENER — varies by rank band AND career fit level
    # ────────────────────────────────────────────────────────────────────────
    yrs_str = f"{yrs:.0f}-year" if yrs == int(yrs) else f"{yrs:.1f}-year"

    if rank <= 10:
        if ml_months >= 36:
            opener = (
                f"{_format_yrs(ml_months)}-year applied ML/AI career as {top_role} "
                f"({yrs:.0f} yrs total experience)"
            )
        else:
            opener = f"{yrs_str} {current_title} with strong technical profile"
    elif technical_score >= 0.25:
        # Genuine technical candidate — positive framing
        opener = f"{yrs_str} {current_title} with relevant ML/retrieval background"
    elif technical_score >= 0.10:
        # Borderline technical (SWE / data role / fallback) — neutral framing
        opener = f"{yrs_str} {current_title} with adjacent technical experience"
    else:
        # Truly non-technical (technical_score < 0.10, i.e. gate fired) —
        # honest framing per JD guidance
        # JD says: "A candidate who has all the AI keywords... but whose title is
        # 'Marketing Manager' is not a fit, no matter how perfect their skill list"
        opener = f"{yrs_str} {current_title} — non-technical background; ranked on platform signals only"

    # ────────────────────────────────────────────────────────────────────────
    # STRENGTH — pulled from scored_data facts, not invented
    # Appending top_skills (when available) ensures no two candidates share
    # an identical reasoning string even if they have the same ml_months bucket
    # and has_retrieval flag. (Stage 4 penalises non-unique reasoning.)
    # ────────────────────────────────────────────────────────────────────────
    if ml_months >= 48 and has_retrieval:
        base = (
            f"brings {_format_yrs(ml_months)} yrs building production retrieval "
            f"and ranking systems at product companies"
        )
        # Append top skills for uniqueness when they exist
        if top_skills:
            skill_str = " and ".join(top_skills[:2])
            strength = f"{base}; key verified skills: {skill_str}"
        else:
            strength = base
    elif ml_months >= 48:
        base = (
            f"brings {_format_yrs(ml_months)} yrs of applied ML/AI engineering "
            f"at product companies"
        )
        if top_skills:
            skill_str = " and ".join(top_skills[:2])
            strength = f"{base}; strongest skills: {skill_str}"
        else:
            strength = base
    elif has_retrieval and top_skills:
        strength = (
            f"career history shows hands-on retrieval/ranking work; "
            f"strongest verified skills: {' and '.join(top_skills[:2])}"
        )
    elif has_retrieval:
        strength = "career descriptions evidence hands-on retrieval/ranking system experience"
    elif top_skills:
        strength = (
            f"strongest verified skills are "
            f"{' and '.join(top_skills[:2])}"
            f" (endorsed, multi-month usage)"
        )
    elif consulting_only:
        strength = "shows relevant technical skills despite a consulting-heavy background"
    else:
        strength = "experience covers adjacent data and ML infrastructure domain"

    # ────────────────────────────────────────────────────────────────────────
    # CONCERN — always shown if genuine (Fix 3); tone softened for top-10
    # Priority order: highest-impact concern surfaces first
    # ────────────────────────────────────────────────────────────────────────
    concern: str | None = None

    if academic_only:
        concern = (
            "career is entirely in academic/research institutions with no evidence "
            "of production deployment — explicit JD disqualifier; ranked on skills signals only"
        )
    elif notice_days > 120:
        concern = (
            f"notice period is {notice_days} days — unusually long for a "
            f"fast-moving startup; early engagement recommended"
        )
    elif days_inactive > 180:
        months_inactive = days_inactive // 30
        concern = (
            f"last active on platform {months_inactive} months ago — "
            f"recommend direct outreach to confirm current interest"
        )
    elif notice_days > 90:
        concern = f"notice period is {notice_days} days — may delay start date"
    elif consulting_only:
        concern = (
            f"career is primarily at services/consulting firms with limited "
            f"product-company experience — cultural fit warrants assessment"
        )
    elif country.lower() not in ("india", "") and "india" not in country.lower():
        concern = (
            f"based internationally ({location}) — no visa sponsorship; "
            f"candidate would need to self-arrange relocation"
        )
    elif avg_tenure > 0 and avg_tenure < 15:
        concern = (
            f"average role tenure is {avg_tenure:.0f} months across "
            f"completed roles — job-hopping pattern warrants discussion"
        )
    elif days_inactive > 90:
        concern = (
            f"last active {days_inactive} days ago — "
            f"confirm availability before outreach"
        )

    # ────────────────────────────────────────────────────────────────────────
    # ASSEMBLE — rank-aware tone on the concern label
    # ────────────────────────────────────────────────────────────────────────
    if concern:
        qualifier = "minor consideration" if rank <= 10 else "concern"
        return f"{opener}: {strength}; {qualifier}: {concern}."
    else:
        return f"{opener}: {strength}."
