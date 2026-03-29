"""
Inference Script -- Meta-Signal Environment
============================================
MANDATORY env vars:
  API_BASE_URL   The API endpoint for the LLM (default: https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier to use for inference
  HF_TOKEN       Your Hugging Face / API key

Runs the LLM agent across all 3 tasks (seed=42) and prints a results table.
Total runtime: < 5 minutes on 2 vCPU / 8GB RAM.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from openai import OpenAI

# Load .env file if present (local development only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.env import MetaSignalEnv, CAMPAIGN_NAMES
from app.models import Action, AttributionMethod, GraderResult
from app.tasks import TASK_CONFIGS

# ---------------------------------------------------------------------------
# Environment variables (mandatory per competition spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"

# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

TEMPERATURE = 0.2
MAX_TOKENS  = 300

SYSTEM_PROMPT = """You are an expert advertising budget optimisation agent.

Your goal: allocate a budget across three ad campaigns to maximise ROAS
(Return on Ad Spend = revenue / spend) while respecting privacy constraints.

Campaigns:
  camp_feed    - high-converting feed placement
  camp_reels   - medium-converting reels placement
  camp_stories - low-converting stories placement

Each step you receive noisy campaign statistics. Respond with a JSON action
object ONLY -- no explanation, no markdown, just raw JSON.

Action format:
{
  "allocations": {
    "camp_feed":    <dollars>,
    "camp_reels":   <dollars>,
    "camp_stories": <dollars>
  },
  "attribution": "last_click",
  "feature_mask": ["I1"]
}

Rules:
1. allocations must be >= 0 and sum must not exceed the budget shown
2. feature_mask items must be from I1-I13 or C1-C26
3. Respect the max features limit shown each step
4. Spend budget across steps -- you have multiple steps per episode
5. Campaigns with higher estimated_roas deserve more budget

Strategy:
- Task 1: Signal is clean. Identify the best campaign and concentrate spend there.
- Task 2: Noise jumps at step 3. Use early steps to learn, then commit to best campaign.
- Task 3: Only 1 feature allowed. Stay compliant -- violations destroy your score.
"""


def _format_obs(obs, step_n: int, total_steps: int, task_id: int) -> str:
    cfg = TASK_CONFIGS[task_id]
    lines = [
        f"Step {step_n}/{total_steps} | Budget: ${obs.total_budget_remaining:.2f} "
        f"| Epsilon: {obs.epsilon_remaining:.3f} | Regime: {obs.privacy_regime.value}",
        f"Task: {cfg.name} | Target ROAS: {cfg.target_roas} | Max features: {cfg.max_features}",
        "",
        "Campaign stats:",
    ]
    for cs in obs.campaigns:
        lines.append(
            f"  {cs.campaign_id}: noisy_conversions={cs.noisy_conversions:.2f} "
            f"estimated_roas={cs.estimated_roas:.3f} ctr={cs.ctr:.4f}"
        )
    if obs.warning:
        lines.append(f"\nWARNING: {obs.warning}")
    if obs.regulatory_violation:
        lines.append("WARNING: last step violated feature limit -- penalty applied")
    budget_per_step = obs.total_budget_remaining / max(1, total_steps - step_n + 1)
    lines.append(f"\nSuggested spend this step: ${budget_per_step:.2f} total")
    lines.append(f"Max features allowed: {cfg.max_features}")
    lines.append("\nRespond with JSON only:")
    return "\n".join(lines)


def _parse_action(content: str, max_features: int) -> Action:
    try:
        data = json.loads(content)
        mask = data.get("feature_mask", [])[:max_features]
        return Action(
            allocations=data.get("allocations", {c: 0.0 for c in CAMPAIGN_NAMES}),
            attribution=AttributionMethod(data.get("attribution", "last_click")),
            feature_mask=mask,
        )
    except Exception:
        return Action(
            allocations={c: 1.0 for c in CAMPAIGN_NAMES},
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=["I1"][:max_features],
        )


def _run_task(env: MetaSignalEnv, task_id: int, seed: int, client: OpenAI) -> GraderResult:
    cfg      = TASK_CONFIGS[task_id]
    obs      = env.reset(task_id=task_id, seed=seed)
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step_n in range(1, cfg.max_steps + 1):
        prompt = _format_obs(obs, step_n, cfg.max_steps, task_id)
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            reply = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM error step {step_n}] {exc} -- using fallback action")
            reply = ""

        messages.append({"role": "assistant", "content": reply})
        action = _parse_action(reply, cfg.max_features)
        result = env.step(action)
        obs    = result.observation

        alloc_str = " ".join(
            f"{k.replace('camp_','')}=${v:.0f}"
            for k, v in action.allocations.items()
        )
        print(f"    step {step_n:2d}: [{alloc_str}]  "
              f"roas={result.info.step_roas:.3f}  "
              f"reward={result.reward:.3f}  "
              f"regime={obs.privacy_regime.value}")

        if result.done:
            break

    return env.compute_final_score()


def main() -> None:
    if not API_KEY:
        raise EnvironmentError(
            "No API key found. Set HF_TOKEN, GROQ_API_KEY, or OPENAI_API_KEY."
        )

    print(f"Model      : {MODEL_NAME}")
    print(f"API base   : {API_BASE_URL}")
    print(f"Seed       : 42\n")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = MetaSignalEnv()

    results: Dict[str, GraderResult] = {}
    for task_id in [1, 2, 3]:
        cfg = TASK_CONFIGS[task_id]
        print(f"Running Task {task_id}: {cfg.name} ({cfg.max_steps} steps) ...")
        grade = _run_task(env, task_id, seed=42, client=client)
        results[f"task_{task_id}"] = grade
        print(f"  Score: {grade.score:.4f}  |  {grade.breakdown}\n")

    print("=" * 55)
    print(f"{'Task':<32} {'Score':>8}")
    print("-" * 55)
    for key, grade in results.items():
        cfg = TASK_CONFIGS[grade.task_id]
        print(f"Task {grade.task_id} {cfg.name:<27} {grade.score:>8.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
