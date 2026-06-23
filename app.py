"""
app.py — PitchSense AI
IBM SkillsBuild AI Builders Challenge · June 2026

A human-centered, explainable AI companion for World Cup 2026.
Three modes: Match Intelligence, VAR Companion, Tactical Explainer.

Stack:
  - IBM Granite (watsonx.ai) for all LLM generation
  - Docling + sentence-transformers for FIFA Laws RAG
  - scikit-learn Random Forest for match outcome prediction
  - Streamlit for UI
"""
import re
import sys
import os
from pathlib import Path

# Allow src/ imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PitchSense AI · World Cup 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark pitch aesthetic, clean data panels
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Header strip ── */
.pitchsense-header {
    background: linear-gradient(135deg, #0a3d1f 0%, #1a5c33 60%, #0d4a28 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.5rem;
    border-left: 5px solid #4ade80;
}
.pitchsense-header h1 {
    color: #f0fdf4;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.25rem;
}
.pitchsense-header p {
    color: #86efac;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Metric cards ── */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { font-size: 1.75rem; font-weight: 800; color: #0f172a; }
.metric-card.win  { border-top: 4px solid #22c55e; }
.metric-card.draw { border-top: 4px solid #f59e0b; }
.metric-card.loss { border-top: 4px solid #ef4444; }

/* ── Explanation box ── */
.explanation-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 0.97rem;
    line-height: 1.7;
    color: #14532d;
    margin-top: 1rem;
}

/* ── VAR badge ── */
.var-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

/* ── Law citation ── */
.law-citation {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #1e3a5f;
    margin-top: 0.75rem;
    font-style: italic;
}

/* ── Probability bar wrapper ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.prob-label { min-width: 110px; font-size: 0.88rem; font-weight: 600; color: #374151; }
.prob-pct   { min-width: 48px; text-align: right; font-size: 0.88rem; color: #6b7280; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

/* ── Warning / info ── */
.info-pill {
    background: #fef3c7;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-size: 0.84rem;
    color: #92400e;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Resource loading — cached for the app's lifetime
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading prediction model …")
def load_model_artifacts():
    model = joblib.load(Path("models") / "match_predictor.pkl")
    data = joblib.load(Path("models") / "team_data.pkl")
    return model, data["team_stats"], data["feature_cols"]


@st.cache_resource(show_spinner="Loading FIFA Laws knowledge base …")
def load_rag():
    """Load the Docling-powered Laws RAG. Gracefully degrades if unavailable."""
    try:
        from laws_rag import get_rag
        rag = get_rag()
        return rag
    except Exception as e:
        st.warning(f"Laws RAG unavailable: {e}. VAR explanations will run without Law citations.")
        return None


@st.cache_resource(show_spinner=False)
def check_granite():
    """Validate watsonx credentials on startup."""
    try:
        from granite_client import validate_credentials
        ok, msg = validate_credentials()
        return ok, msg
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_match(model, team_stats, feature_cols, team_a, team_b, neutral, major):
    a, b = team_stats[team_a], team_stats[team_b]
    row = pd.DataFrame([{
        "team_a_winrate":    a["winrate"],
        "team_b_winrate":    b["winrate"],
        "team_a_goal_avg":   a["goal_avg"],
        "team_b_goal_avg":   b["goal_avg"],
        "team_a_recent_form": a["recent_form"],
        "team_b_recent_form": b["recent_form"],
        "is_neutral":        int(neutral),
        "is_major_tournament": int(major),
    }])[feature_cols]
    proba = model.predict_proba(row)[0]
    return {
        "team_a_win_prob": float(proba[0]),
        "draw_prob":       float(proba[1]),
        "team_b_win_prob": float(proba[2]),
    }


def prob_bar(label: str, prob: float, color: str):
    """Render a styled probability bar."""
    pct = prob * 100
    st.markdown(f"""
    <div class="prob-row">
      <span class="prob-label">{label}</span>
      <div style="flex:1; background:#e5e7eb; border-radius:6px; height:18px; overflow:hidden;">
        <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:6px;
                    transition:width 0.4s ease;"></div>
      </div>
      <span class="prob-pct">{pct:.1f}%</span>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# World Cup 2026 Group Stage fixtures (hardcoded — live bracket)
# ---------------------------------------------------------------------------

WC2026_FIXTURES = [
    # Group A
    ("USA", "Mexico"), ("Canada", "Morocco"),
    # Group B
    ("Spain", "Brazil"), ("France", "Argentina"),
    # Group C
    ("England", "Germany"), ("Portugal", "Netherlands"),
    # Group D
    ("Japan", "South Korea"), ("Australia", "Senegal"),
    # Group E
    ("Belgium", "Croatia"), ("Switzerland", "Uruguay"),
    # Group F
    ("Ecuador", "Saudi Arabia"), ("Cameroon", "Colombia"),
]

KNOWN_TEAMS_WC26 = list({t for pair in WC2026_FIXTURES for t in pair})


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="pitchsense-header">
  <h1>⚽ PitchSense AI</h1>
  <p>World Cup 2026 · Powered by IBM Granite &amp; Docling · Explainable AI for every fan</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load all resources
# ---------------------------------------------------------------------------

try:
    model, team_stats, feature_cols = load_model_artifacts()
    team_names = sorted(team_stats.keys())
    model_loaded = True
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. "
        "Run the Learning Lab notebook first to generate `models/match_predictor.pkl` "
        "and `models/team_data.pkl`."
    )
    st.stop()

rag = load_rag()
granite_ok, granite_msg = check_granite()

if not granite_ok:
    st.markdown(
        f'<div class="info-pill">⚙️ IBM Granite not configured: {granite_msg} — '
        f'AI explanations will be disabled. Set WATSONX_API_KEY and WATSONX_PROJECT_ID '
        f'in your environment or Streamlit secrets.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔮 Match Intelligence",
    "⚖️ VAR Companion",
    "🧠 Tactical Explainer",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Match Intelligence
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Match Intelligence", divider="green")
    st.caption(
        "Select two teams to get an AI-predicted outcome and a Granite-generated "
        "pre-match tactical briefing grounded in historical stats."
    )

    # Quick-fill from World Cup 2026 fixtures
    with st.expander("⚡ Quick-fill a World Cup 2026 fixture"):
        fixture_labels = [f"{a} vs {b}" for a, b in WC2026_FIXTURES]
        chosen = st.selectbox("Pick a fixture", ["— choose —"] + fixture_labels, key="wc_fixture")
        if chosen != "— choose —":
            idx = fixture_labels.index(chosen)
            st.session_state["t1_team_a"] = WC2026_FIXTURES[idx][0]
            st.session_state["t1_team_b"] = WC2026_FIXTURES[idx][1]

    col1, col2 = st.columns(2)
    with col1:
        default_a = team_names.index("Brazil") if "Brazil" in team_names else 0
        team_a = st.selectbox(
            "Team A",
            team_names,
            index=team_names.index(st.session_state.get("t1_team_a", "Brazil"))
                  if st.session_state.get("t1_team_a") in team_names else default_a,
            key="t1_sel_a",
        )
    with col2:
        default_b = team_names.index("Argentina") if "Argentina" in team_names else 1
        team_b = st.selectbox(
            "Team B",
            team_names,
            index=team_names.index(st.session_state.get("t1_team_b", "Argentina"))
                  if st.session_state.get("t1_team_b") in team_names else default_b,
            key="t1_sel_b",
        )

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        neutral = st.checkbox("Neutral venue", value=True, key="t1_neutral")
    with opt_col2:
        major = st.checkbox("Major tournament", value=True, key="t1_major")

    predict_btn = st.button("Analyse Match →", type="primary", use_container_width=True, key="t1_predict")

    if predict_btn:
        if team_a == team_b:
            st.error("Please select two different teams.")
        else:
            probs = predict_match(model, team_stats, feature_cols, team_a, team_b, neutral, major)
            a_stats, b_stats = team_stats[team_a], team_stats[team_b]

            # Outcome cards
            st.markdown("#### Predicted Outcome")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card win">
                  <div class="label">{team_a} wins</div>
                  <div class="value">{probs['team_a_win_prob']:.0%}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card draw">
                  <div class="label">Draw</div>
                  <div class="value">{probs['draw_prob']:.0%}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card loss">
                  <div class="label">{team_b} wins</div>
                  <div class="value">{probs['team_b_win_prob']:.0%}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            prob_bar(team_a, probs["team_a_win_prob"], "#22c55e")
            prob_bar("Draw",  probs["draw_prob"],       "#f59e0b")
            prob_bar(team_b, probs["team_b_win_prob"], "#ef4444")

            # Team stats table
            st.markdown("#### Historical Stats Used")
            stats_df = pd.DataFrame({
                team_a: {
                    "Win rate":               f"{a_stats['winrate']:.1%}",
                    "Goals per match":         f"{a_stats['goal_avg']:.2f}",
                    "Recent form (last 10)":   f"{a_stats['recent_form']:.1%}",
                    "Matches in dataset":      a_stats["matches_played"],
                },
                team_b: {
                    "Win rate":               f"{b_stats['winrate']:.1%}",
                    "Goals per match":         f"{b_stats['goal_avg']:.2f}",
                    "Recent form (last 10)":   f"{b_stats['recent_form']:.1%}",
                    "Matches in dataset":      b_stats["matches_played"],
                },
            })
            st.table(stats_df)

            # Granite tactical briefing
            st.markdown("#### AI Tactical Briefing")
            if granite_ok:
                with st.spinner("Granite is generating the tactical briefing …"):
                    try:
                        from granite_client import explain_prematch
                        briefing = explain_prematch(
                            team_a, team_b, a_stats, b_stats,
                            neutral, major, probs
                        )
                        st.markdown(
                            f'<div class="explanation-box">{briefing}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Granite error: {e}")
            else:
                st.markdown(
                    '<div class="info-pill">Configure WATSONX_API_KEY to enable AI briefings.</div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VAR Companion
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("VAR Companion", divider="blue")
    st.caption(
        "Describe any match situation and get a plain-English explanation of what "
        "VAR checks, why decisions are made, and what the Laws of the Game actually say. "
        "Powered by IBM Granite + Docling-parsed FIFA Laws."
    )

    # Pre-built scenario templates
    VAR_TEMPLATES = {
        "— enter your own —": ("", ""),
        "Offside goal disallowed": (
            "A striker receives a through-ball and scores, but the flag stays down. "
            "VAR reviews a potential offside. The attacker's armpit is 2cm ahead of the last defender.",
            "Goal disallowed — offside."
        ),
        "Handball penalty": (
            "A defender blocks a shot with their arm inside the box. The arm is slightly away from their body. "
            "The referee waves play on. VAR intervenes.",
            "Penalty awarded after VAR review for handball."
        ),
        "Red card for violent conduct": (
            "During a corner kick, a player elbows an opponent off the ball, away from play. "
            "The referee misses it. VAR checks the incident.",
            "Red card shown for violent conduct after VAR check."
        ),
        "Penalty retake ordered": (
            "The goalkeeper saves a penalty. VAR notices the keeper's feet were off the line "
            "before the ball was kicked.",
            "Penalty retake ordered — goalkeeper encroachment."
        ),
        "Goal ruled out: foul in build-up": (
            "A team scores after a flowing 8-pass move. VAR reviews a possible foul "
            "3 passes earlier in the build-up.",
            "Goal disallowed — foul in the build-up phase."
        ),
    }

    template_choice = st.selectbox("Load a template scenario", list(VAR_TEMPLATES.keys()), key="var_template")
    t_situation, t_decision = VAR_TEMPLATES[template_choice]

    situation = st.text_area(
        "Describe the match situation",
        value=t_situation,
        height=120,
        placeholder="e.g. A striker's shoulder is 3cm offside when the ball is played. The goal stands on the pitch. VAR reviews the call.",
         key=f"var_situation_{template_choice}",
    )
    decision = st.text_input(
        "What decision was made?",
        value=t_decision,
        placeholder="e.g. Goal disallowed — offside.",
        key=f"var_decision_{template_choice}",
    )

    rag_status = "✅ FIFA Laws loaded" if (rag and rag.ready) else "⚠️ Laws RAG not available"
    st.caption(f"Knowledge base: {rag_status}")

    var_btn = st.button("Explain this VAR decision →", type="primary", use_container_width=True, key="var_btn")

    if var_btn:
        if not situation.strip():
            st.error("Please describe the match situation.")
        elif not granite_ok:
            st.error("IBM Granite is not configured. Set WATSONX_API_KEY and WATSONX_PROJECT_ID.")
        else:
            with st.spinner("Retrieving Laws … generating explanation …"):
                # Retrieve relevant Law chunks via Docling RAG
                laws_context = ""
                retrieved_chunks = []
                if rag and rag.ready:
                    query = f"{situation} {decision}"
                    retrieved_chunks = rag.retrieve(query, top_k=4)
                    laws_context = rag.retrieve_as_context(query, top_k=4)

                try:
                    from granite_client import explain_var_decision
                    explanation = explain_var_decision(situation, decision, laws_context)

                    st.markdown('<div class="var-badge">VAR ANALYSIS · IBM GRANITE</div>', unsafe_allow_html=True)
                    explanation_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', explanation)
                    st.markdown(
                    f'<div class="explanation-box">{explanation_html}</div>',
                    unsafe_allow_html=True,
                    )

                    # Show the Law citations retrieved by Docling
                    if retrieved_chunks:
                        with st.expander("📖 Laws of the Game — retrieved context (Docling)"):
                            for chunk in retrieved_chunks:
                                st.markdown(
                                    f'<div class="law-citation"><strong>{chunk["law"]}</strong> '
                                    f'(relevance: {chunk["score"]:.2f})<br>{chunk["text"][:400]}…</div>',
                                    unsafe_allow_html=True,
                                )
                except Exception as e:
                    st.error(f"Granite error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tactical Explainer
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Tactical Explainer", divider="orange")
    st.caption(
        "Two tools in one: explain a formation change mid-match, "
        "or break down a momentum-shifting moment. Powered by IBM Granite."
    )

    tac_mode = st.radio(
        "What do you want to explain?",
        ["Formation / substitution change", "Momentum shift moment"],
        horizontal=True,
        key="tac_mode",
    )

    if tac_mode == "Formation / substitution change":
        st.markdown("---")
        tac_team = st.selectbox("Team", team_names, key="tac_team",
                                index=team_names.index("France") if "France" in team_names else 0)

        col_a, col_b = st.columns(2)
        with col_a:
            before_f = st.text_input("Formation before change", value="4-4-2", key="tac_before")
        with col_b:
            after_f = st.text_input("Formation after change", value="4-3-3", key="tac_after")

        tac_context = st.text_area(
            "Match context",
            value="Trailing 0-1 at half-time. Struggled to create chances centrally. Opponent sitting in a deep 5-4-1 block.",
            height=90,
            key="tac_context",
        )

        tac_btn = st.button("Explain this change →", type="primary", use_container_width=True, key="tac_btn1")

        if tac_btn:
            if not granite_ok:
                st.error("IBM Granite not configured.")
            elif tac_team not in team_stats:
                st.error(f"{tac_team} not found in dataset.")
            else:
                with st.spinner("Generating tactical analysis …"):
                    try:
                        from granite_client import explain_tactical_shift
                        analysis = explain_tactical_shift(
                            tac_team, before_f, after_f,
                            tac_context, team_stats[tac_team]
                        )
                        st.markdown(
                            f'<div class="explanation-box">{analysis}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Granite error: {e}")

    else:  # Momentum shift
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mom_team_a = st.selectbox("Team A", team_names, key="mom_a",
                                     index=team_names.index("England") if "England" in team_names else 0)
        with col_m2:
            mom_team_b = st.selectbox("Team B", team_names, key="mom_b",
                                     index=team_names.index("Germany") if "Germany" in team_names else 1)

        mom_minute = st.slider("Minute of event", 1, 120, 67, key="mom_min")
        mom_event = st.text_input(
            "What happened?",
            value="Red card shown to England's holding midfielder.",
            key="mom_event",
        )
        mom_state = st.text_input(
            "Match state at that moment",
            value="England leading 1-0. Germany had started to build pressure in the previous 10 minutes.",
            key="mom_state",
        )

        mom_btn = st.button("Explain momentum shift →", type="primary", use_container_width=True, key="mom_btn")

        if mom_btn:
            if not granite_ok:
                st.error("IBM Granite not configured.")
            else:
                with st.spinner("Generating momentum analysis …"):
                    try:
                        from granite_client import explain_momentum_shift
                        analysis = explain_momentum_shift(
                            mom_team_a, mom_team_b,
                            mom_event, mom_minute, mom_state
                        )
                        st.markdown(
                            f'<div class="explanation-box">{analysis}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Granite error: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<center style='color:#94a3b8; font-size:0.8rem;'>"
    "PitchSense AI · IBM SkillsBuild AI Builders Challenge · June 2026 · "
    "Built with IBM Granite, Docling, IBM Bob, scikit-learn, Streamlit"
    "</center>",
    unsafe_allow_html=True,
)
