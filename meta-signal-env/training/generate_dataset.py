"""
Dataset Generator — runs the ExpertBot across many seeds and writes a JSONL
training file with one record per step.

Each record is a JSON object with:
  - "instruction": natural-language description of the situation
  - "input":       the observation (serialised) + current phase hint
  - "output":      the expert action (serialised)
  - "metadata":    step/task/seed/score for filtering

This format is compatible with Alpaca-style fine-tuning (Unsloth, Axolotl, etc.)

Usage:
    python -m training.generate_dataset --tasks 5 6 7 --episodes 200 --out data/expert_demos.jsonl
    python -m training.generate_dataset --tasks 7 --episodes 50 --out data/task7_only.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.env import MetaSignalEnv
from app.models import (
    Action,
    LearningStatus,
    Observation,
    PlatformHealth,
    StepResult,
)
from app.tasks import get_task_config
from training.expert_bot import ExpertBot

# ---------------------------------------------------------------------------
# Instruction templates per phase
# ---------------------------------------------------------------------------

_PHASE_INSTRUCTIONS: Dict[str, str] = {
    "Nominal": (
        "You are an advertising budget optimisation agent in Phase 1 (days 1–20) of a "
        "100-day campaign. Signal quality is clean. Your goal is to identify which campaign "
        "has the best conversion rate (CVR) and progressively concentrate budget there "
        "while staying below the 70% concentration limit to avoid the correlation penalty. "
        "Output a JSON action with allocations, attribution, and feature_mask."
    ),
    "Signal_Loss": (
        "You are an advertising budget optimisation agent in Phase 2 (days 21–50). "
        "iOS App Tracking Transparency has caused a 3x noise spike — your noisy_conversions "
        "signal is unreliable. You can set use_capi=true to pay 2.0 epsilon and get clean "
        "conversion counts, but ration carefully. Hold your Phase 1 allocation steady between "
        "CAPI calls. Do NOT chase the noisy signal. "
        "Output a JSON action with allocations, attribution, feature_mask, and use_capi."
    ),
    "Andromeda_Glitched": (
        "You are an advertising budget optimisation agent in Phase 3 (days 51–80). "
        "The Andromeda algorithm update is live. Any allocation change exceeding 20% of total "
        "budget triggers a 7-day learning reset — CVR drops to 30% of normal. "
        "Freeze your Phase 2 allocation and scale it linearly to the per-step budget. "
        "If learning_status is Reset, do not change allocations at all for 7 steps. "
        "Output a JSON action with allocations, attribution, and feature_mask."
    ),
    "Peak_Load": (
        "You are an advertising budget optimisation agent in Phase 4 (days 81–100). "
        "Black Friday peak load has doubled noise volatility. Setting pacing_speed above 1.5 "
        "carries a 30% chance of a catastrophic budget dump per step. "
        "Hold your Phase 3 allocation, set pacing_speed=1.0, and prioritise surviving "
        "to the end of the episode over maximising ROAS. "
        "Output a JSON action with allocations, attribution, feature_mask, and pacing_speed."
    ),
}


def _obs_to_text(obs: Observation) -> str:
    """Serialise observation to a compact but readable string for the 'input' field."""
    camps = []
    for cs in obs.campaigns:
        camps.append(
            f"{cs.campaign_id}: spend=${cs.spend:.1f}, "
            f"noisy_conv={cs.noisy_conversions:.2f}, "
            f"est_roas={cs.estimated_roas:.3f}, "
            f"ctr={cs.ctr:.4f}, "
            f"ci=[{cs.confidence_interval[0]:.2f},{cs.confidence_interval[1]:.2f}]"
        )
    camp_str = " | ".join(camps)

    lines = [
        f"step={obs.step} day={obs.day} phase={obs.platform_health.value}",
        f"budget_remaining=${obs.total_budget_remaining:.2f} epsilon={obs.epsilon_remaining:.3f}",
        f"privacy_regime={obs.privacy_regime.value} learning_status={obs.learning_status.value}",
        f"market_trend={obs.market_trend.value} regulatory_violation={obs.regulatory_violation}",
        f"campaigns: {camp_str}",
    ]
    if obs.warning:
        lines.append(f"warning: {obs.warning}")
    return "\n".join(lines)


def _action_to_text(action: Action) -> str:
    """Serialise action to JSON string."""
    d = {
        "allocations": {k: round(v, 2) for k, v in action.allocations.items()},
        "attribution":  action.attribution.value,
        "feature_mask": action.feature_mask,
        "use_capi":     action.use_capi,
        "pacing_speed": action.pacing_speed,
        "apply_safety_cap": action.apply_safety_cap,
    }
    return json.dumps(d)


def _make_record(
    obs: Observation,
    action: Action,
    result: StepResult,
    task_id: int,
    seed: int,
    episode_score: float,
) -> dict:
    phase_key = obs.platform_health.value  # e.g. "Nominal"
    instruction = _PHASE_INSTRUCTIONS.get(phase_key, _PHASE_INSTRUCTIONS["Nominal"])
    input_text  = _obs_to_text(obs)
    output_text = _action_to_text(action)

    return {
        "instruction": instruction,
        "input":       input_text,
        "output":      output_text,
        "metadata": {
            "task_id":       task_id,
            "seed":          seed,
            "step":          obs.step,
            "day":           obs.day,
            "phase":         phase_key,
            "step_roas":     round(result.info.step_roas, 4),
            "reward":        round(result.reward, 4),
            "episode_score": round(episode_score, 4),
            "use_capi":      action.use_capi,
        },
    }


# ---------------------------------------------------------------------------
# Episode runner that collects records
# ---------------------------------------------------------------------------


def generate_episode(
    task_id: int,
    seed: int,
) -> List[dict]:
    """
    Run one expert episode and return a list of training records (one per step).
    episode_score is filled in post-hoc after the episode ends.
    """
    env  = MetaSignalEnv()
    bot  = ExpertBot()
    obs  = env.reset(task_id=task_id, seed=seed)
    bot.reset()
    total_steps = get_task_config(task_id).max_steps

    records_pending: List[dict] = []
    raw_steps: List[tuple] = []

    while not env.state().is_done:
        action = bot.act(obs, total_steps=total_steps)
        result = env.step(action)
        raw_steps.append((obs, action, result))
        obs = result.observation

    grader = env.compute_final_score()
    episode_score = grader.score

    for (step_obs, step_action, step_result) in raw_steps:
        rec = _make_record(step_obs, step_action, step_result, task_id, seed, episode_score)
        records_pending.append(rec)

    return records_pending


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_dataset(
    tasks: List[int],
    episodes_per_task: int,
    output_path: Path,
    base_seed: int = 0,
    min_score: float = 0.0,
) -> None:
    """
    Generate `episodes_per_task` episodes for each task in `tasks`.
    Writes records to `output_path` as JSONL.

    Records from episodes scoring below `min_score` are excluded (quality filter).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(base_seed)

    total_records = 0
    total_episodes = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for task_id in tasks:
            ep_scores = []
            for ep_idx in range(episodes_per_task):
                seed = rng.randint(0, 999_999)
                try:
                    records = generate_episode(task_id=task_id, seed=seed)
                except Exception as exc:
                    print(f"  [WARN] task={task_id} ep={ep_idx} seed={seed} error: {exc}")
                    skipped += 1
                    continue

                ep_score = records[0]["metadata"]["episode_score"] if records else 0.0
                ep_scores.append(ep_score)

                if ep_score < min_score:
                    skipped += 1
                    continue

                for rec in records:
                    fout.write(json.dumps(rec) + "\n")
                    total_records += 1

                total_episodes += 1

                if (ep_idx + 1) % 10 == 0:
                    avg = sum(ep_scores[-10:]) / min(10, len(ep_scores))
                    print(
                        f"  task={task_id} ep={ep_idx+1}/{episodes_per_task} "
                        f"last_score={ep_score:.3f} avg10={avg:.3f} records={total_records}"
                    )

    print(f"\nDone. {total_episodes} episodes, {total_records} records -> {output_path}")
    if skipped:
        print(f"Skipped {skipped} episodes (below min_score={min_score} or errors).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate expert demonstration dataset.")
    p.add_argument("--tasks",    type=int, nargs="+", default=[5, 6, 7], choices=[5, 6, 7])
    p.add_argument("--episodes", type=int, default=200, help="Episodes per task")
    p.add_argument("--out",      type=str, default="data/expert_demos.jsonl")
    p.add_argument("--seed",     type=int, default=0,   help="Base random seed")
    p.add_argument(
        "--min-score", type=float, default=0.3,
        help="Drop episodes below this score (quality filter)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"Generating {args.episodes} episodes × {len(args.tasks)} tasks "
          f"(min_score={args.min_score}) → {args.out}")
    generate_dataset(
        tasks=args.tasks,
        episodes_per_task=args.episodes,
        output_path=Path(args.out),
        base_seed=args.seed,
        min_score=args.min_score,
    )
