"""
app.py
~~~~~~
Streamlit Sandbox for the Redrob Candidate Ranking Challenge.

Allows you (or Stage 3 judges) to:
  - Upload a sample candidates file (JSON or JSONL)
  - Run the full ranking pipeline on it
  - Explore top candidates with their scores and reasoning
  - Download the resulting CSV

To run:
    streamlit run app.py

The app uses pre-computed embeddings (embeddings.npy / id_index.json) if
they exist in the working directory.  For candidates NOT in the pre-computed
array (e.g. fresh sample files) it falls back to on-the-fly encoding.
"""

from __future__ import annotations
import json
import io
import os
import sys
import time
import tempfile

import pandas as pd
import streamlit as st

# Make sure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6C63FF, #3ECFCF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #888;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .score-badge-high   { color: #4CAF50; font-weight: bold; font-size: 1.05rem; }
    .score-badge-mid    { color: #FFC107; font-weight: bold; font-size: 1.05rem; }
    .score-badge-low    { color: #F44336; font-weight: bold; font-size: 1.05rem; }
    .reasoning-box {
        background: #111827;
        border-left: 3px solid #6C63FF;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #ccc;
        margin-top: 0.3rem;
    }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stSidebar"] { background-color: #0f0f1a; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading sentence-transformers model...")
def load_semantic_module():
    """Load semantic scorer once and cache at app level."""
    from scorer.semantic_fit import initialise
    initialise()


def load_candidates_from_bytes(raw: bytes, filename: str) -> list[dict]:
    """Parse uploaded file bytes into a list of candidate dicts."""
    text = raw.decode("utf-8")
    if filename.endswith(".json"):
        return json.loads(text)
    # JSONL
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def run_pipeline(candidates: list[dict], top_n: int) -> tuple[pd.DataFrame, float]:
    """
    Run the full scoring pipeline on a list of candidates.
    Returns (submission_df, elapsed_seconds).
    """
    from scorer.honeypot       import compute_honeypot_penalty
    from scorer.career_fit     import compute_career_fit
    from scorer.skills_fit     import compute_skills_fit
    from scorer.behavioral_fit import compute_behavioral_fit
    from scorer.location_fit   import compute_location_fit
    from scorer.semantic_fit   import compute_all_semantic_scores
    from reasoning.generator   import build_reasoning

    W_CAREER, W_SKILLS, W_BEHAVIORAL, W_SEMANTIC, W_LOCATION = (
        0.35, 0.25, 0.20, 0.12, 0.08
    )

    t0 = time.time()

    semantic_scores = compute_all_semantic_scores(candidates)

    rows = []
    for c in candidates:
        cid = c["candidate_id"]
        hp  = compute_honeypot_penalty(c)
        cf  = compute_career_fit(c)
        sf  = compute_skills_fit(c)
        bf  = compute_behavioral_fit(c)
        lf  = compute_location_fit(c)
        sem = semantic_scores.get(cid, 0.0)

        # Composite score (weighted sum)
        composite = (
            W_CAREER     * cf["score"]
          + W_SKILLS     * sf["score"]
          + W_BEHAVIORAL * bf["score"]
          + W_SEMANTIC   * sem
          + W_LOCATION   * lf["score"]
        )

        # Career-fit gate: candidates with career_fit < 0.20 have zero
        # genuine ML/engineering signal. Suppress their composite to 10%
        # so no Operations Manager / Graphic Designer can beat a real SWE.
        if cf["score"] < 0.20:
            composite *= 0.10

        # Honeypot penalty applied after the gate (additive, always <= 0)
        composite += hp

        scored_data = {
            **cf,
            "top_skills":    sf["top_skills"],
            "days_inactive": bf["days_inactive"],
            "country":       lf["country"],
        }

        rows.append({
            "candidate_id":     cid,
            "final_score":      composite,
            "career_fit":       round(cf["score"], 4),
            "skills_fit":       round(sf["score"], 4),
            "behavioral":       round(bf["score"], 4),
            "semantic":         round(sem, 4),
            "location":         round(lf["score"], 4),
            "honeypot_penalty": round(hp, 4),
            "_candidate":       c,
            "_scored_data":     scored_data,
        })

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True],
    ).head(top_n).copy()

    df_sorted["rank"] = range(1, len(df_sorted) + 1)
    df_sorted["score"] = df_sorted["final_score"].round(6)

    reasonings = []
    for _, row in df_sorted.iterrows():
        r = build_reasoning(
            candidate=row["_candidate"],
            scored_data=row["_scored_data"],
            rank=int(row["rank"]),
        )
        reasonings.append(r)
    df_sorted["reasoning"] = reasonings

    elapsed = time.time() - t0

    submission = df_sorted[[
        "rank", "candidate_id", "score",
        "career_fit", "skills_fit", "behavioral", "semantic", "location",
        "honeypot_penalty", "reasoning",
    ]].reset_index(drop=True)

    return submission, elapsed


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    top_n = st.slider("Top-N candidates to rank", min_value=10, max_value=100, value=100, step=10)
    st.markdown("---")
    st.markdown("### Pipeline Weights")
    st.markdown("""
| Component | Weight |
|---|---|
| Career Fit | **35%** |
| Skills Fit | **25%** |
| Behavioral | **20%** |
| Semantic   | **12%** |
| Location   | **8%**  |
""")
    st.markdown("---")
    st.markdown("### Pre-computed Embeddings")
    emb_exists = os.path.exists("embeddings.npy")
    idx_exists  = os.path.exists("id_index.json")
    if emb_exists and idx_exists:
        emb_size = os.path.getsize("embeddings.npy") / (1024 * 1024)
        st.success(f"embeddings.npy found ({emb_size:.1f} MB)")
    else:
        st.warning("No pre-computed embeddings found.\nOn-the-fly encoding will be used (slower for large files).")
    st.markdown("---")
    st.caption("Redrob Candidate Ranker v1.0")


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Redrob Candidate Ranker</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a candidate file and instantly rank them for the Senior AI Engineer role.</p>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Upload candidates file (.json or .jsonl)",
    type=["json", "jsonl"],
    help="Upload sample_candidates.json or candidates.jsonl",
)

if uploaded_file is None:
    st.info("Upload a candidate file above to get started. The full ranking will run in seconds once embeddings are pre-computed.")
    st.stop()

# Parse uploaded file
try:
    raw_bytes  = uploaded_file.read()
    candidates = load_candidates_from_bytes(raw_bytes, uploaded_file.name)
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

st.success(f"Loaded **{len(candidates):,} candidates** from `{uploaded_file.name}`")

# Run button
if st.button("Run Ranking Pipeline", type="primary", use_container_width=True):

    with st.spinner("Loading semantic model (first run may take ~10s)..."):
        load_semantic_module()

    progress_bar = st.progress(0, text="Scoring candidates...")

    try:
        with st.spinner(f"Running full pipeline on {len(candidates):,} candidates..."):
            progress_bar.progress(20, text="Computing semantic similarity...")
            results, elapsed = run_pipeline(candidates, top_n=min(top_n, len(candidates)))
            progress_bar.progress(100, text="Done!")

        st.session_state["results"] = results
        st.session_state["elapsed"] = elapsed

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Show results if available
if "results" not in st.session_state:
    st.stop()

results: pd.DataFrame = st.session_state["results"]
elapsed: float = st.session_state["elapsed"]

# ── Summary metrics ──────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Candidates Ranked", f"{len(results)}")
with col2:
    top_score = results["score"].max()
    st.metric("Top Score", f"{top_score:.4f}")
with col3:
    flagged = (results["honeypot_penalty"] < -0.1).sum()
    st.metric("Honeypots in Top-N", str(flagged))
with col4:
    st.metric("Pipeline Time", f"{elapsed:.1f}s")

# ── Top-10 deep dive ─────────────────────────────────────────────────────────
st.markdown("### Top 10 Candidates")

for _, row in results.head(10).iterrows():
    score = float(row["score"])
    if score >= 0.6:
        badge_class = "score-badge-high"
    elif score >= 0.3:
        badge_class = "score-badge-mid"
    else:
        badge_class = "score-badge-low"

    cid   = row["candidate_id"]
    rank  = int(row["rank"])

    with st.expander(f"#{rank}  {cid}  —  score: {score:.4f}", expanded=(rank <= 3)):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Career Fit",  f"{row['career_fit']:.3f}")
        c2.metric("Skills Fit",  f"{row['skills_fit']:.3f}")
        c3.metric("Behavioral",  f"{row['behavioral']:.3f}")
        c4.metric("Semantic",    f"{row['semantic']:.3f}")
        c5.metric("Location",    f"{row['location']:.3f}")

        st.markdown(f'<div class="reasoning-box">{row["reasoning"]}</div>', unsafe_allow_html=True)

        if float(row["honeypot_penalty"]) < -0.1:
            st.warning(f"Honeypot flag: penalty = {row['honeypot_penalty']:.2f}")

# ── Full results table ───────────────────────────────────────────────────────
st.markdown("### Full Rankings Table")

display_df = results[[
    "rank", "candidate_id", "score",
    "career_fit", "skills_fit", "behavioral", "semantic", "location",
    "honeypot_penalty"
]].copy()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "rank":            st.column_config.NumberColumn("Rank", width="small"),
        "candidate_id":    st.column_config.TextColumn("Candidate ID"),
        "score":           st.column_config.NumberColumn("Score", format="%.4f"),
        "career_fit":      st.column_config.ProgressColumn("Career", format="%.3f", min_value=0, max_value=1),
        "skills_fit":      st.column_config.ProgressColumn("Skills", format="%.3f", min_value=0, max_value=1),
        "behavioral":      st.column_config.ProgressColumn("Behavioral", format="%.3f", min_value=0, max_value=1),
        "semantic":        st.column_config.ProgressColumn("Semantic", format="%.3f", min_value=0, max_value=1),
        "location":        st.column_config.ProgressColumn("Location", format="%.3f", min_value=0, max_value=1),
        "honeypot_penalty": st.column_config.NumberColumn("Honeypot", format="%.2f"),
    }
)

# ── Download ─────────────────────────────────────────────────────────────────
st.markdown("### Download Submission CSV")

submission_csv = results[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)

st.download_button(
    label="Download submission.csv",
    data=submission_csv.encode("utf-8"),
    file_name="submission.csv",
    mime="text/csv",
    use_container_width=True,
    type="primary",
)

st.caption("The downloaded CSV matches the exact format required by validate_submission.py")
