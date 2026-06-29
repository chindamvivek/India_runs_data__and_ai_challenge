"""
scorer/behavioral_fit.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Scores a candidate's availability and engagement based on the 23 Redrob
platform behavioral signals.

Weight in composite: 20%

The JD explicitly states: "a perfect-on-paper candidate who hasn't logged in
for 6 months and has a 5% recruiter response rate is, for hiring purposes,
not actually available. Down-weight them appropriately."

All six component signals are normalised to [0.0, 1.0].
Weights sum exactly to 1.0:
    0.35 × recency          (last_active_date)
    0.20 × response_rate    (recruiter_response_rate)
    0.15 × notice_score     (notice_period_days)
    0.15 × open_to_work     (open_to_work_flag, clean 0/1)
    0.10 × completeness     (profile_completeness_score / 100)
    0.05 × reliability      (interview_completion_rate)
Total: 1.00

Returns:
    dict with keys:
        score           float [0.0, 1.0]
        days_inactive   int   — days since last login (used by reasoning generator)
"""

from __future__ import annotations
from datetime import date

# Use a fixed reference date for reproducibility.
# Set to the competition date; Stage 3 reproduction will use the same file.
REFERENCE_DATE = date(2026, 6, 21)


def _recency_score(last_active_str: str | None) -> float:
    """Score how recently the candidate was active on the platform."""
    if not last_active_str:
        return 0.1   # Unknown — assume low engagement

    try:
        last_active = date.fromisoformat(last_active_str)
    except (ValueError, TypeError):
        return 0.1

    days = (REFERENCE_DATE - last_active).days

    if days <= 0:
        return 1.0   # Active today or in the future (data quirk) → treat as max
    elif days <= 30:
        return 1.0   # Active this month
    elif days <= 90:
        return 0.80  # Active this quarter
    elif days <= 180:
        return 0.50  # Active this half-year
    else:
        return 0.20  # 6+ months inactive → effectively unavailable


def _notice_score(notice_days: int) -> float:
    """Score the notice period. JD: 'we'd love sub-30 day notice.'"""
    if notice_days <= 0:
        return 1.0   # Immediately available
    elif notice_days <= 30:
        return 1.00  # Sub-30: JD's ideal
    elif notice_days <= 60:
        return 0.75
    elif notice_days <= 90:
        return 0.50
    elif notice_days <= 120:
        return 0.30
    else:
        return 0.10  # 150-180 days: nearly disqualifying for a fast-moving startup


def compute_behavioral_fit(candidate: dict) -> dict:
    """
    Compute the behavioral fit score for a candidate.

    Returns:
        dict with keys: score, days_inactive
    """
    signals = candidate.get("redrob_signals", {})

    last_active_str = signals.get("last_active_date")
    open_to_work    = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    response_rate   = float(signals.get("recruiter_response_rate", 0.0))
    notice_days     = int(signals.get("notice_period_days", 90))
    completeness    = float(signals.get("profile_completeness_score", 50.0)) / 100.0
    interview_rate  = float(signals.get("interview_completion_rate", 0.5))

    # Clamp any out-of-range values defensively
    response_rate  = max(0.0, min(1.0, response_rate))
    completeness   = max(0.0, min(1.0, completeness))
    interview_rate = max(0.0, min(1.0, interview_rate))

    recency = _recency_score(last_active_str)
    notice  = _notice_score(notice_days)

    # Compute days_inactive for reasoning generator
    days_inactive = 0
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_inactive = max(0, (REFERENCE_DATE - last_active).days)
        except (ValueError, TypeError):
            days_inactive = 999  # Unknown — treat as very inactive

    # Composite behavioral score — weights sum to 1.0
    behavioral = (
        0.35 * recency        # Most critical: are they actually active?
      + 0.20 * response_rate  # Will they reply to us?
      + 0.15 * notice         # Can we hire them quickly?
      + 0.15 * open_to_work   # Clean 0/1 flag (v4 Fix 2)
      + 0.10 * completeness   # How seriously are they job-hunting?
      + 0.05 * interview_rate # Do they show up when it matters?
    )
    # Verification: 0.35 + 0.20 + 0.15 + 0.15 + 0.10 + 0.05 = 1.00

    return {
        "score":         round(behavioral, 6),
        "days_inactive": days_inactive,
    }
