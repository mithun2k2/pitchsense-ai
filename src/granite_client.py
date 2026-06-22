"""
granite_client.py
IBM Granite API integration for MatchMind AI.
Uses ibm-watsonx-ai SDK — handles region routing and auth automatically.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com")

GRANITE_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"

DEFAULT_PARAMS = {
    GenParams.MAX_NEW_TOKENS:      512,
    GenParams.MIN_NEW_TOKENS:      50,
    GenParams.TEMPERATURE:         0.7,
    GenParams.TOP_P:               0.9,
    GenParams.REPETITION_PENALTY:  1.1,
}

# ---------------------------------------------------------------------------
# SDK client — lazy initialisation
# ---------------------------------------------------------------------------

_model: ModelInference | None = None


def _get_model() -> ModelInference:
    global _model
    if _model is not None:
        return _model

    if not WATSONX_API_KEY:
        raise EnvironmentError("WATSONX_API_KEY is not set.")
    if not WATSONX_PROJECT_ID:
        raise EnvironmentError("WATSONX_PROJECT_ID is not set.")

    credentials = Credentials(
        url=WATSONX_URL,
        api_key=WATSONX_API_KEY,
    )

    _model = ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params=DEFAULT_PARAMS,
    )
    return _model


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    model = _get_model()

    if system_prompt:
        full_prompt = (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )
    else:
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

    params = {
        GenParams.MAX_NEW_TOKENS:     max_new_tokens,
        GenParams.TEMPERATURE:        temperature,
        GenParams.MIN_NEW_TOKENS:     1,
        GenParams.REPETITION_PENALTY: 1.1,
    }

    response = model.generate_text(prompt=full_prompt, params=params)
    return response.strip()


# ---------------------------------------------------------------------------
# Domain-specific system prompts
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = (
    "You are MatchMind AI, an expert soccer analyst and tactical commentator. "
    "You explain football matches in clear, engaging language accessible to any fan. "
    "Always ground your analysis in specific stats provided. "
    "Never make up scores or facts not in the prompt. "
    "Write like a Guardian football correspondent: authoritative, warm, precise."
)

_VAR_SYSTEM = (
    "You are a FIFA Laws of the Game expert and VAR specialist. "
    "You explain officiating decisions using the exact language and logic of the Laws. "
    "Be impartial, precise, and educational. "
    "When Laws text is provided as context, cite the relevant article directly. "
    "Keep explanations clear enough for a fan with no officiating background."
)

_TACTICAL_SYSTEM = (
    "You are a UEFA Pro Licence tactical analyst. "
    "You break down formations, pressing triggers, transitions, and set-piece routines "
    "in vivid, specific language. Ground all analysis in the team stats provided. "
    "Never be vague. Every claim should connect to a concrete stat or pattern."
)


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def explain_prematch(team_a, team_b, stats_a, stats_b, is_neutral, is_major, probabilities):
    context = f"""
Match: {team_a} vs {team_b}
Venue: {"Neutral" if is_neutral else f"{team_a} home advantage"}
Tournament: {"Major tournament (World Cup / Euros)" if is_major else "Friendly / qualifier"}

{team_a} stats:
  Win rate:       {stats_a['winrate']:.1%}
  Goals per game: {stats_a['goal_avg']:.2f}
  Recent form:    {stats_a['recent_form']:.1%} (last 10 matches)
  Matches played: {stats_a['matches_played']}

{team_b} stats:
  Win rate:       {stats_b['winrate']:.1%}
  Goals per game: {stats_b['goal_avg']:.2f}
  Recent form:    {stats_b['recent_form']:.1%} (last 10 matches)
  Matches played: {stats_b['matches_played']}

AI predicted probabilities:
  {team_a} win: {probabilities['team_a_win_prob']:.1%}
  Draw:         {probabilities['draw_prob']:.1%}
  {team_b} win: {probabilities['team_b_win_prob']:.1%}
"""
    prompt = (
        f"Write a pre-match tactical briefing for {team_a} vs {team_b}. "
        f"Cover: (1) what the stats reveal about each team's style and current form, "
        f"(2) the key tactical battle that will decide the match, "
        f"(3) which team the AI model favours and why. "
        f"Keep it to 3 focused paragraphs.\n\nContext:\n{context}"
    )
    return generate(prompt, system_prompt=_MATCH_SYSTEM, max_new_tokens=450, temperature=0.72)


def explain_var_decision(situation, decision, laws_context=""):
    context_block = f"\nRelevant Laws of the Game extract:\n{laws_context}\n" if laws_context else ""
    prompt = (
        f"A VAR check has just been completed.\n\n"
        f"Situation: {situation}\n"
        f"Decision: {decision}\n"
        f"{context_block}\n"
        f"Explain: (1) exactly what the VAR team checked and why, "
        f"(2) what the Laws say about this situation, "
        f"(3) whether the decision was correct and why fans find it controversial. "
        f"Be direct. Cite the Law number if provided in context."
    )
    return generate(prompt, system_prompt=_VAR_SYSTEM, max_new_tokens=500, temperature=0.5)


def explain_tactical_shift(team, before_formation, after_formation, match_context, stats):
    prompt = (
        f"{team} changed shape from {before_formation} to {after_formation}.\n\n"
        f"Match context: {match_context}\n"
        f"Team stats: win rate {stats['winrate']:.1%}, "
        f"avg goals {stats['goal_avg']:.2f}, "
        f"recent form {stats['recent_form']:.1%}.\n\n"
        f"Explain: (1) what problem the manager was solving, "
        f"(2) what the new shape offers tactically, "
        f"(3) the risk this introduces. 2-3 paragraphs."
    )
    return generate(prompt, system_prompt=_TACTICAL_SYSTEM, max_new_tokens=420, temperature=0.68)


def explain_momentum_shift(team_a, team_b, event, minute, match_state):
    prompt = (
        f"At minute {minute} of {team_a} vs {team_b}: {event}\n"
        f"Match state: {match_state}\n\n"
        f"Explain how this shifts momentum: what changes for each team "
        f"psychologically and tactically, and what to expect next. 2 paragraphs."
    )
    return generate(prompt, system_prompt=_MATCH_SYSTEM, max_new_tokens=350, temperature=0.75)


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------

def validate_credentials() -> tuple[bool, str]:
    try:
        _get_model()
        # Quick test generation
        result = generate("Say OK", max_new_tokens=20, temperature=0.1)
        return True, f"Credentials valid. Model responded: {result[:20]}"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    ok, msg = validate_credentials()
    print("OK" if ok else "FAIL", msg)