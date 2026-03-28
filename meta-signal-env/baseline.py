"""
Baseline inference script -- GPT-4o-mini agent.

Runs all three tasks with seed=42 and returns a BaselineResult.
Can be imported by the FastAPI server (/baseline endpoint) or run standalone.

Expected scores (equal-split dummy confirms these gaps are real):
  Task 1: ~0.65   model spots best campaign from clean signals
  Task 2: ~0.40   model struggles when signal degrades mid-episode
  Task 3: ~0.25   model tends to use too many features
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from app.env import MetaSignalEnv, CAMPAIGN_NAMES
from app.models import (
    Action,
    AttributionMethod,
    BaselineResult,
    GraderResult,
    Observation,
)
from app.tasks import TASK_CONFIGS

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert advertising budget optimisation agent.

Your goal: allocate a budget across three ad campaigns to maximise ROAS
(Return on Ad Spend = revenue / spend) while respecting privacy constraints.

Campaigns:
  camp_feed    - high-converting feed placement
  camp_reels   - medium-converting reels placement
  camp_stories - low-converting stories placement

Each step you receive noisy campaign statistics. You must respond with a JSON
action object ONLY -- no explanation, no markdown, just raw JSON.

Action format:
{
  "allocations": {
    "camp_feed":    <dollars, float>,
    "camp_reels":   <dollars, float>,
    "camp_stories": <dollars, float>
  },
  "attribution": "last_click",
  "feature_mask": ["I1"]
}

Rules:
1. allocations must be >= 0 and their sum must not exceed the budget shown
2. feature_mask items must be from I1-I13 or C1-C26
3. You will be told the max features allowed per step -- respect this limit
4. Spend budget wisely across steps -- you have multiple steps per episode
5. Campaigns with higher estimated_roas deserve more budget

Strategy tips:
- In Task 1: signal is clean, identify the best campaign quickly and concentrate spend
- In Task 2: signal degrades at step 3 -- use early steps to learn, then commit
- In Task 3: only 1 feature allowed per step -- choose it carefully, stay compliant
"""


def _format_observation(obs: Observation, step_n: int, total_steps: int, task_id: int) -> str:
    """Convert an Observation into a clear text prompt for the GPT agent."""
    cfg = TASK_CONFIGS[task_id]
    lines = [
        f"Step {step_n}/{total_steps} | Budget remaining: ${obs.total_budget_remaining:.2f} "
        f"| Epsilon: {obs.epsilon_remaining:.3f} | Regime: {obs.privacy_regime.value}",
        f"Task: {cfg.name} | Target ROAS: {cfg.target_roas} | Max features per step: {cfg.max_features}",
        "",
        "Campaign stats this step:",
    ]
    for cs in obs.campaigns:
        lines.append(
            f"  {cs.campaign_id} ({cs.placement}): "
            f"noisy_conversions={cs.noisy_conversions:.2f}, "
            f"estimated_roas={cs.estimated_roas:.3f}, "
            f"ctr={cs.ctr:.4f}, "
            f"impressions={cs.impressions}"
        )

    if obs.warning:
        lines.append(f"\nWARNING: {obs.warning}")
    if obs.regulatory_violation:
        lines.append("WARNING: last action violated the feature limit -- penalty applied")

    budget_per_step = obs.total_budget_remaining / max(1, total_steps - step_n + 1)
    lines.append(
        f"\nSuggested spend this step (to spread evenly): ${budget_per_step:.2f} total"
    )
    lines.append(f"Allowed features: {obs.available_features[:5]} ... "
                 f"(max {cfg.max_features} per step)")
    lines.append("\nRespond with JSON action only:")
    return "\n".join(lines)


def _parse_action(content: str, max_features: int) -> Action:
    """
    Parse GPT response into an Action. Falls back to a safe equal-split action
    if the response is malformed.
    """
    try:
        data = json.loads(content)
        # Clip feature_mask to allowed limit
        mask = data.get("feature_mask", [])[:max_features]
        return Action(
            allocations=data.get("allocations", {c: 0.0 for c in CAMPAIGN_NAMES}),
            attribution=AttributionMethod(data.get("attribution", "last_click")),
            feature_mask=mask,
        )
    except Exception:
        # Fallback: equal split, single feature
        return Action(
            allocations={c: 1.0 for c in CAMPAIGN_NAMES},
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=["I1"][:max_features],
        )


def _run_task(
    env: MetaSignalEnv,
    task_id: int,
    seed: int,
    client,
) -> GraderResult:
    """Run one full episode for a task using the GPT agent."""
    cfg    = TASK_CONFIGS[task_id]
    obs    = env.reset(task_id=task_id, seed=seed)
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step_n in range(1, cfg.max_steps + 1):
        prompt = _format_observation(obs, step_n, cfg.max_steps, task_id)
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        action = _parse_action(reply, cfg.max_features)
        result = env.step(action)
        obs    = result.observation

        if result.done:
            break

    return env.compute_final_score()


def run_baseline(seed: int = 42) -> BaselineResult:
    """
    Run the GPT-4o-mini baseline across all three tasks.
    Returns a BaselineResult with per-task scores and breakdowns.

    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it before running the baseline."
        )

    from openai import OpenAI
    client  = OpenAI(api_key=api_key)
    env     = MetaSignalEnv()   # fresh instance, does not touch the server's shared env
    scores:  Dict[str, float]      = {}
    details: Dict[str, GraderResult] = {}

    for task_id in [1, 2, 3]:
        grade = _run_task(env, task_id, seed, client)
        key   = f"task_{task_id}"
        scores[key]  = grade.score
        details[key] = grade

    return BaselineResult(
        model="gpt-4o-mini",
        seed=seed,
        scores=scores,
        details=details,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running GPT-4o-mini baseline (seed=42) ...\n")
    try:
        result = run_baseline(seed=42)
        print(f"Model : {result.model}")
        print(f"Seed  : {result.seed}\n")
        print(f"{'Task':<30} {'Score':>8}  Breakdown")
        print("-" * 70)
        for key, grade in result.details.items():
            bd = "  ".join(f"{k}={v:.3f}" for k, v in grade.breakdown.items())
            print(f"Task {grade.task_id} {TASK_CONFIGS[grade.task_id].name:<25} "
                  f"{grade.score:>8.4f}  {bd}")
    except EnvironmentError as e:
        print(f"ERROR: {e}")
