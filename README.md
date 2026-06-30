# Redrob Intelligent Candidate Discovery & Ranking Challenge

A multi-dimensional candidate ranking system for the **Senior AI Engineer — Founding Team** role at Redrob AI. Scores 100,000 candidates across 5 dimensions and returns the top 100 with fact-grounded reasoning.

---

## Quick Start

### Prerequisites
- Python 3.10+
- 16 GB RAM (CPU-only, no GPU required)
- ~160 MB disk space for pre-computed embeddings

### 1. Set up the environment

```bash
# Clone the repo
git clone https://github.com/chindamvivek/India_runs_data__and_ai_challenge.git
cd India_runs_data__and_ai_challenge

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pre-compute embeddings (one-time, ~10-15 min)

This step encodes all 100K candidates using `sentence-transformers/all-MiniLM-L6-v2` and saves them as a 153 MB `.npy` file. It only needs to run **once**:

```bash
python precompute.py --candidates ./candidates.jsonl
```

You'll see a progress bar. When it prints `Done! Pre-computation complete`, two files appear in your project root:
- `embeddings.npy` (~153 MB) — L2-normalised float32 embeddings
- `id_index.json` — maps candidate IDs to their row index

### 3. Generate the ranking (< 5 minutes)

```bash
python rank.py --candidates ./candidates.jsonl --team-id YOUR_TEAM_ID
```

This produces `YOUR_TEAM_ID.csv` — your submission file. It also runs `validate_submission.py` automatically and prints `Submission is valid.`

### 4. (Optional) Run the Streamlit Sandbox

```bash
streamlit run app.py
```

Upload `sample_candidates.json` via the browser UI to explore rankings interactively.

---

## Project Structure

```
.
├── rank.py                     # Main entry point (CLI)
├── precompute.py               # One-time embedding pre-computation
├── app.py                      # Streamlit sandbox UI
├── submission_metadata.yaml    # Submission metadata
├── requirements.txt
│
├── scorer/
│   ├── honeypot.py             # 5 integrity checks → honeypot penalty
│   ├── career_fit.py           # Career history analysis (35% weight)
│   ├── skills_fit.py           # Trust-weighted skill scoring (25%)
│   ├── behavioral_fit.py       # Platform engagement signals (20%)
│   ├── semantic_fit.py         # Sentence-transformer similarity (12%)
│   └── location_fit.py         # Location & relocation scoring (8%)
│
└── reasoning/
    └── generator.py            # Deterministic reasoning string builder
```

---

## Scoring Architecture

### Composite Score Formula

```
composite = 0.35 × career_fit
          + 0.25 × skills_fit
          + 0.20 × behavioral
          + 0.12 × semantic_fit
          + 0.08 × location_fit
          + honeypot_penalty      ← always ≤ 0
```

### What Each Component Does

| Component | Weight | What it measures |
|---|---|---|
| **Career Fit** | 35% | ML/AI role history, title tier, product vs. consulting, depth (ML months), job-hopping penalty |
| **Skills Fit** | 25% | Trust score = `base_weight × proficiency × duration × endorsements`; fixed-cap normalisation |
| **Behavioral** | 20% | Platform activity: recency, response rate, notice period, open-to-work, completeness, interview rate |
| **Semantic** | 12% | Cosine similarity of candidate text vs. JD embedding (pre-computed, sub-second) |
| **Location** | 8% | Pune/Noida/Delhi NCR > Tier-1 India > India + relocation > international |

### Honeypot Detection

5 integrity checks on candidate data quality:
1. **Salary impossible**: `salary_min > salary_max + 2.0 LPA`
2. **Skill duration > career**: any skill's `duration_months > total_exp_months + 24`
3. **Expert/Advanced at 0 months**: `proficiency ∈ {expert, advanced}` AND `duration_months == 0`
4. **Career date math mismatch**: `|actual_months - stated_duration_months| > 6` for any role
5. **Education timeline impossible**: `end_year < start_year` for any degree

Each check adds a negative penalty. Flagged candidates score far below all legitimate candidates.

### Key Design Decisions

- **Not keyword counting**: `career_fit` evaluates full career history and penalises consulting-only tracks; `skills_fit` uses a trust formula not raw skill count.
- **LangChain penalty is contextual**: only applied if LangChain is listed AND no production retrieval skills (FAISS/Pinecone/Qdrant/etc.) are present.
- **Concerns always shown**: the reasoning string always surfaces genuine concerns. Tone softens for top-10 (`"minor consideration: ..."`) but facts are never hidden.
- **Stable normalisation**: `skills_fit` uses a fixed cap (`MAX_PLAUSIBLE_SKILL_SCORE = 15.0`), not min-max, so scores are identical whether run on 50 or 100K candidates.

---

## Runtime

| Step | Time |
|---|---|
| `precompute.py` (100K candidates, one-time) | ~10-15 min |
| `rank.py` — load embeddings | ~5s |
| `rank.py` — score 100K candidates | ~30-60s |
| `rank.py` — build top-100 reasoning | < 1s |
| **Total `rank.py` wall-clock** | **< 5 min** |

---

## Reproduce Command

```bash
python precompute.py --candidates ./candidates.jsonl  # (if not already done)
python rank.py --candidates ./candidates.jsonl --team-id YOUR_TEAM_ID
```
