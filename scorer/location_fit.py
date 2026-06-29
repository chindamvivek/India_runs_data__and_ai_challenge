"""
scorer/location_fit.py
~~~~~~~~~~~~~~~~~~~~~~
Scores a candidate based on their location relative to the JD's preferences.

Weight in composite: 8%

JD location requirements:
  - Preferred: Pune / Noida (India)
  - Acceptable: Hyderabad, Mumbai, Delhi NCR, Bengaluru, Chennai + willing to relocate
  - Outside India: case-by-case, no visa sponsorship

Returns:
    dict with keys:
        score    float [0.0, 1.0]
        country  str   — candidate's country (used by reasoning generator)
"""

from __future__ import annotations

# Cities in JD's explicit "preferred" zone
PUNE_NOIDA_KEYWORDS: frozenset[str] = frozenset({
    "pune", "noida", "delhi", "delhi ncr", "new delhi",
    "gurugram", "gurgaon", "greater noida", "faridabad",
    "ncr",
})

# Tier-1 Indian cities explicitly listed or implied in the JD
TIER1_INDIA_KEYWORDS: frozenset[str] = frozenset({
    "bangalore", "bengaluru", "hyderabad", "mumbai",
    "chennai", "kolkata", "ahmedabad",
})


def _normalise_city(location: str) -> str:
    """Lowercase and strip a location string for keyword matching."""
    return location.lower().strip()


def _city_match(loc_lower: str, city_set: frozenset) -> bool:
    """
    Return True if any city keyword in city_set matches the location string
    as a whole word (or whole phrase), not just as a substring.

    This prevents false positives such as 'philadelphia' matching 'delhi'
    or 'bangalore' matching 'new bangalore township' incorrectly.

    Strategy: tokenise the location on common separators, then check if any
    known city keyword equals one of the tokens or matches a contiguous
    multi-token sequence.
    """
    import re
    # Split on comma, slash, hyphen, or whitespace; drop empty tokens.
    tokens = [t for t in re.split(r'[,/\-\s]+', loc_lower) if t]

    for city in city_set:
        city_tokens = city.split()   # city keywords may be multi-word ("delhi ncr")
        n = len(city_tokens)
        # Check every n-gram of the location tokens against this city keyword.
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == city_tokens:
                return True
    return False


def compute_location_fit(candidate: dict) -> dict:
    """
    Compute the location fit score for a candidate.

    Returns:
        dict with keys: score, country
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = profile.get("location", "")
    country  = profile.get("country", "")
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))

    loc_lower     = _normalise_city(location)
    country_lower = country.lower().strip()

    is_india = "india" in country_lower

    if is_india:
        # Check if location string contains any Pune/Noida/Delhi keywords
        in_preferred = _city_match(loc_lower, PUNE_NOIDA_KEYWORDS)
        in_tier1     = _city_match(loc_lower, TIER1_INDIA_KEYWORDS)

        if in_preferred:
            score = 1.00   # Already in JD's preferred city cluster
        elif in_tier1 and willing_to_relocate:
            score = 0.80   # Tier-1 city, willing to move
        elif in_tier1 and not willing_to_relocate:
            score = 0.55   # Tier-1 but won't relocate — needs remote or WFH
        elif willing_to_relocate:
            score = 0.65   # Anywhere in India, willing to move
        else:
            score = 0.40   # In India but won't relocate and not in preferred city
    else:
        # International candidate
        # JD: "Outside India: case-by-case, but we don't sponsor work visas"
        if willing_to_relocate:
            score = 0.20   # Willing to relocate internationally — possible but uncertain
        else:
            score = 0.05   # International, not willing to relocate — very unlikely fit

    return {
        "score":   score,
        "country": country,
    }
