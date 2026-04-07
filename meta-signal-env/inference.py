"""
inference.py -- COMPETITION SUBMISSION SCRIPT (mandatory name per hackathon rules)
====================================================================================
Mandatory env vars (set in HF Space Secrets or .env for local testing):
  API_BASE_URL   The LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier  (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN       Your Hugging Face / API key

STDOUT FORMAT (mandatory -- any deviation fails evaluation):
  [START] task=<name> env=meta-signal model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Runtime: < 5 min on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.env import MetaSignalEnv, CAMPAIGN_NAMES
from app.models import Action, AttributionMethod, GraderResult
from app.tasks import TASK_CONFIGS

# ---------------------------------------------------------------------------
# Mandatory environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Mandatory stdout log helpers — exact format required by validator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


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
4. Spread budget across steps -- you have multiple steps per episode
5. Campaigns with higher estimated_roas deserve more budget

Strategy:
- Task 1: Signal is clean. Explore early (3-way split), learn mid, exploit best campaign late.
- Task 2: Heavy noise throughout. At step 9 Reels CVR doubles -- watch for a sudden ROAS jump.
- Task 3: Only 1 feature allowed. Every step costs epsilon. Be conservative to avoid depletion.
- Task 4: Audit fires at step 5. When audit_active=true set flagged campaign to 0,
  include halted_campaigns and legal_reason_code: GDPR_ART17.

Task 4 action when audit is active:
{
  "allocations": {"camp_feed": 500, "camp_reels": 300, "camp_stories": 0},
  "attribution": "last_click",
  "feature_mask": ["I1"],
  "halted_campaigns": ["camp_stories"],
  "legal_reason_code": "GDPR_ART17"
}
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
    if obs.audit_active and obs.flagged_campaign:
        lines.append(
            f"\nREGULATORY AUDIT ACTIVE: '{obs.flagged_campaign}' is SUSPENDED."
            " Set its allocation to 0 and include legal_reason_code: GDPR_ART17"
        )
    budget_per_step = obs.total_budget_remaining / max(1, total_steps - step_n + 1)
    lines.append(f"\nSuggested spend this step: ${budget_per_step:.2f} total")
    lines.append(f"Max features allowed: {cfg.max_features}")
    lines.append("\nRespond with JSON only:")
    return "\n".join(lines)


def _parse_action(content: str, max_features: int, flagged: str | None = None) -> Action:
    try:
        data   = json.loads(content)
        allocs = data.get("allocations", {c: 0.0 for c in CAMPAIGN_NAMES})
        if flagged:
            allocs[flagged] = 0.0
        mask = data.get("feature_mask", [])[:max_features]
        return Action(
            allocations=allocs,
            attribution=AttributionMethod(data.get("attribution", "last_click")),
            feature_mask=mask,
            halted_campaigns=data.get("halted_campaigns", []),
            legal_reason_code=data.get("legal_reason_code"),
        )
    except Exception:
        allocs = {c: 1.0 for c in CAMPAIGN_NAMES}
        if flagged:
            allocs[flagged] = 0.0
        return Action(
            allocations=allocs,
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=["I1"][:max_features],
        )


def _run_task(env: MetaSignalEnv, task_id: int, seed: int, client: OpenAI) -> GraderResult:
    cfg      = TASK_CONFIGS[task_id]
    task_slug = cfg.name.lower().replace(" ", "-")
    obs      = env.reset(task_id=task_id, seed=seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    flagged: str | None = None
    rewards: List[float] = []
    steps_taken = 0
    last_error: Optional[str] = None

    log_start(task=task_slug, env="meta-signal", model=MODEL_NAME)

    try:
        for step_n in range(1, cfg.max_steps + 1):
            if obs.audit_active and obs.flagged_campaign:
                flagged = obs.flagged_campaign

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
                last_error = None
            except Exception as exc:
                reply = ""
                last_error = str(exc)[:120]
                print(f"[LLM error step {step_n}] {exc}", file=sys.stderr, flush=True)

            messages.append({"role": "assistant", "content": reply})
            action = _parse_action(reply, cfg.max_features, flagged)
            result = env.step(action)
            obs    = result.observation
            steps_taken = step_n

            rewards.append(result.reward)

            # compact action string for log
            alloc_str = json.dumps(action.allocations, separators=(",", ":"))
            log_step(
                step=step_n,
                action=alloc_str,
                reward=result.reward,
                done=result.done,
                error=last_error,
            )

            if result.done:
                break

    except Exception as exc:
        last_error = str(exc)
        print(f"[episode error] {exc}", file=sys.stderr, flush=True)

    grade   = env.compute_final_score()
    score   = grade.score
    success = score > 0.0

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return grade


def main() -> None:
    if not HF_TOKEN:
        print("ERROR: No API key. Set HF_TOKEN or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    print(f"Model    : {MODEL_NAME}",   file=sys.stderr, flush=True)
    print(f"API base : {API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"Seed     : 42\n",           file=sys.stderr, flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env    = MetaSignalEnv()

    results = {}
    for task_id in [1, 2, 3, 4]:
        grade = _run_task(env, task_id, seed=42, client=client)
        results[task_id] = grade

    # Human-readable summary — stderr only so validator stdout stays clean
    print("\n" + "=" * 55, file=sys.stderr)
    print(f"{'Task':<35} {'Score':>8}", file=sys.stderr)
    print("-" * 55, file=sys.stderr)
    for tid, grade in results.items():
        cfg = TASK_CONFIGS[tid]
        print(f"Task {tid} {cfg.name:<30} {grade.score:>8.4f}", file=sys.stderr)
    avg = sum(g.score for g in results.values()) / len(results)
    print(f"{'Average':<35} {avg:>8.4f}", file=sys.stderr)
    print("=" * 55, file=sys.stderr)


if __name__ == "__main__":
    main()
