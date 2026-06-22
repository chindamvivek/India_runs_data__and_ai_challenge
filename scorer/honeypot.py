"""
scorer/honeypot.py
~~~~~~~~~~~~~~~~~~
Detects candidates with subtly impossible profiles ("honeypots") as described
in the submission spec Section 7.

Strategy: never discard — instead return a honeypot_penalty (a negative float).
A total penalty of -0.5 or lower pushes any candidate well below all legitimate
candidates in the composite score, so they will never appear in the top 100.

Five checks are implemented:
  1. Salary impossible  : salary_min > salary_max + 2.0
  2. Skill duration > career : any skill's duration_months > total_exp_months + 24
  3. Expert/Advanced + 0 months : proficiency in (expert, advanced) and duration_months == 0
  4. Career date math mismatch : |actual_months - stated_duration_months| > 6 for any role
  5. Education timeline impossible : end_year < start_year for any degree
"""

from __future__ import annotations
from datetime import datetime


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse an ISO date string (YYYY-MM-DD) into a datetime. Returns None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def compute_honeypot_penalty(candidate: dict) -> float:
    """
    Analyse a candidate profile for impossible / inconsistent data.

    Returns:
        honeypot_penalty (float, <= 0.0)
            0.0  → no anomalies detected
            < 0.0 → anomalies found; value is negative enough to push the
                    candidate out of the top 100 when added to the composite score
    """
    penalty = 0.0
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    education = candidate.get("education", [])

    total_exp_years  = float(profile.get("years_of_experience", 0))
    total_exp_months = total_exp_years * 12.0

    # ------------------------------------------------------------------
    # Check 1: Salary impossible (min > max by more than 2 LPA)
    # Example from real data: CAND with min=16.0, max=7.3
    # ------------------------------------------------------------------
    sal = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = float(sal.get("min", 0))
    sal_max = float(sal.get("max", 0))
    if sal_min > sal_max + 2.0:
        penalty -= 0.50

    # ------------------------------------------------------------------
    # Check 2: Any skill's duration_months exceeds total career experience
    # Allow a 24-month tolerance (someone could have learned a skill during
    # overlapping roles). Cap total penalty from this check at -0.60.
    # ------------------------------------------------------------------
    skill_duration_penalty = 0.0
    for skill in skills:
        s_duration = float(skill.get("duration_months", 0))
        if s_duration > total_exp_months + 24:
            skill_duration_penalty -= 0.30
            if skill_duration_penalty <= -0.60:
                break
    penalty += skill_duration_penalty

    # ------------------------------------------------------------------
    # Check 3: Expert or Advanced proficiency with 0 months of use
    # A genuine expert has used the skill for some time.
    # Cap total penalty from this check at -0.40.
    # ------------------------------------------------------------------
    zero_duration_penalty = 0.0
    for skill in skills:
        if (skill.get("proficiency") in ("expert", "advanced")
                and int(skill.get("duration_months", 1)) == 0):
            zero_duration_penalty -= 0.20
            if zero_duration_penalty <= -0.40:
                break
    penalty += zero_duration_penalty

    # ------------------------------------------------------------------
    # Check 4: Career history date math mismatch
    # The stated duration_months should roughly match (end - start) in months.
    # Allow ±6 months tolerance for rounding/approximation.
    # Cap total penalty from this check at -0.60.
    # ------------------------------------------------------------------
    date_mismatch_penalty = 0.0
    for job in career:
        stated   = int(job.get("duration_months", 0))
        start_dt = _parse_date(job.get("start_date"))
        end_str  = job.get("end_date")

        if end_str is None:
            # Current role: compare against today
            end_dt = datetime.now()
        else:
            end_dt = _parse_date(end_str)

        if start_dt and end_dt:
            actual = (
                (end_dt.year - start_dt.year) * 12
                + (end_dt.month - start_dt.month)
            )
            if abs(actual - stated) > 6:
                date_mismatch_penalty -= 0.30
                if date_mismatch_penalty <= -0.60:
                    break
    penalty += date_mismatch_penalty

    # ------------------------------------------------------------------
    # Check 5: Education timeline impossible (end_year < start_year)
    # ------------------------------------------------------------------
    for edu in education:
        start_yr = edu.get("start_year")
        end_yr   = edu.get("end_year")
        if start_yr and end_yr and end_yr < start_yr:
            penalty -= 0.50
            break  # One confirmed impossible education is enough

    return penalty
