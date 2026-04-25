"""
Expert Bot — deterministic oracle strategy for the Q4 Gauntlet.

Encodes the "right" behaviour for all four phases so the dataset generator
can produce high-quality demonstrations without a trained model.

Usage (single episode):
    python -m training.expert_bot --task 7 --seed 42 --verbose

The bot is also importable:
    from training.expert_bot import ExpertBot, run_episode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.env import MetaSignalEnv
from app.models import (
    Action,
    AttributionMethod,
    LearningStatus,
    Observation,
    PlatformHealth,
)
from app.tasks import get_task_config

# ---------------------------------------------------------------------------
# Constants matching env.py
# ---------------------------------------------------------------------------

_CAMPAIGNS = ["camp_feed", "camp_reels", "camp_stories"]

# Natural CVR ranking (feed > reels > stories) — used by Phase 1 explore logic
_PRIOR_RANK = ["camp_feed", "camp_reels", "camp_stories"]

# Andromeda safe delta: stay strictly under 20% of total_budget
_PHASE3_MAX_DELTA_FRAC = 0.18

# Concentration cap to avoid correlation penalty
_MAX_CONCENTRATION = 0.65

# Phase 1 exploration: equal split for first N steps, then shift
_EXPLORE_STEPS = 5

# Phase 2 CAPI ration: one call every N steps
_CAPI_EVERY = 4


class ExpertBot:
    """
    Stateful expert that tracks episode observations and emits optimal actions.

    Phase strategy:
      Phase 1 (NOMINAL, days 1–20)     — explore equally → shift to best CVR campaign
      Phase 2 (SIGNAL_LOSS, days 21–50) — ration CAPI, hold allocation steady between calls
      Phase 3 (ANDROMEDA_GLITCHED, 51–80) — freeze allocations, tiny nudges only
      Phase 4 (PEAK_LOAD, 81–100)     — hold Phase 2 allocation, pacing_speed=1.0
    """

    def __init__(self) -> None:
        self._step_count = 0
        self._best_camp: str = "camp_feed"
        self._last_alloc: Dict[str, float] = {}
        self._capi_counter: int = 0
        # Running CVR estimates per campaign (spend → conversions)
        self._cvr: Dict[str, float] = {c: 0.0 for c in _CAMPAIGNS}
        self._spend_sum: Dict[str, float] = {c: 1e-9 for c in _CAMPAIGNS}
        self._conv_sum: Dict[str, float] = {c: 0.0 for c in _CAMPAIGNS}

    def reset(self) -> None:
        self.__init__()

    def _update_estimates(self, obs: Observation) -> None:
        for cs in obs.campaigns:
            # noisy_conversions can be negative due to Laplace noise — clip
            nc = max(0.0, cs.noisy_conversions)
            self._spend_sum[cs.campaign_id] += cs.spend
            self._conv_sum[cs.campaign_id] += nc
            self._cvr[cs.campaign_id] = (
                self._conv_sum[cs.campaign_id] / self._spend_sum[cs.campaign_id]
            )
        self._best_camp = max(_CAMPAIGNS, key=lambda c: self._cvr[c])

    def _phase1_action(self, obs: Observation, total_steps: int = 100) -> Action:
        per_step = self._per_step_budget(obs, total_steps)

        if self._step_count < _EXPLORE_STEPS:
            # Equal split — gather data
            alloc = {c: per_step / 3.0 for c in _CAMPAIGNS}
        else:
            # Shift toward best, respect concentration cap
            best_frac = min(_MAX_CONCENTRATION, 0.55 + self._step_count * 0.01)
            rest_frac = (1.0 - best_frac) / 2.0
            alloc = {c: per_step * rest_frac for c in _CAMPAIGNS}
            alloc[self._best_camp] = per_step * best_frac

        self._last_alloc = alloc
        return Action(
            allocations=alloc,
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=["I1"],  # one cheap feature for marginal signal
        )

    def _phase2_action(self, obs: Observation, use_capi: bool, total_steps: int = 100) -> Action:
        per_step = self._per_step_budget(obs, total_steps)

        # Use same allocation shape as end of Phase 1, just scaled
        if self._last_alloc:
            total_last = sum(self._last_alloc.values()) or per_step
            scale = per_step / total_last
            alloc = {c: v * scale for c, v in self._last_alloc.items()}
        else:
            # Fallback if Phase 2 starts without Phase 1 history (task 5 starts here)
            alloc = {
                "camp_feed":    per_step * 0.55,
                "camp_reels":   per_step * 0.28,
                "camp_stories": per_step * 0.17,
            }

        self._last_alloc = alloc
        return Action(
            allocations=alloc,
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=["I1"],
            use_capi=use_capi,
        )

    def _phase3_action(self, obs: Observation, total_steps: int = 100) -> Action:
        per_step = self._per_step_budget(obs, total_steps)

        if obs.learning_status == LearningStatus.RESET:
            # Freeze completely — do not risk another reset
            alloc = {**self._last_alloc} if self._last_alloc else {c: per_step / 3 for c in _CAMPAIGNS}
            total = sum(alloc.values()) or per_step
            scale = per_step / total
            alloc = {c: v * scale for c, v in alloc.items()}
        elif self._last_alloc:
            # Tiny nudge allowed — scale existing allocation to current step budget
            total_last = sum(self._last_alloc.values()) or per_step
            scale = per_step / total_last
            alloc = {c: v * scale for c, v in self._last_alloc.items()}
            # Verify delta stays safe
            delta = sum(abs(alloc[c] - self._last_alloc.get(c, 0.0)) for c in _CAMPAIGNS)
            total_budget_ep = obs.total_budget_remaining + sum(
                cs.spend for cs in obs.campaigns
            )
            safe_delta = _PHASE3_MAX_DELTA_FRAC * total_budget_ep
            if delta > safe_delta:
                # Snap back to last alloc scaled
                alloc = {c: self._last_alloc[c] * scale for c in _CAMPAIGNS}
        else:
            alloc = {c: per_step / 3 for c in _CAMPAIGNS}

        self._last_alloc = alloc
        return Action(
            allocations=alloc,
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=["I1"],
        )

    def _phase4_action(self, obs: Observation, total_steps: int = 100) -> Action:
        per_step = self._per_step_budget(obs, total_steps)

        if self._last_alloc:
            total_last = sum(self._last_alloc.values()) or per_step
            scale = per_step / total_last
            alloc = {c: v * scale for c, v in self._last_alloc.items()}
        else:
            alloc = {c: per_step / 3 for c in _CAMPAIGNS}

        return Action(
            allocations=alloc,
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=["I1"],
            pacing_speed=1.0,  # never risk overspend bug
        )

    @staticmethod
    def _per_step_budget(obs: Observation, total_steps: int = 100) -> float:
        """Spread remaining budget evenly across remaining episode steps."""
        steps_left = max(1, total_steps - obs.step)
        return obs.total_budget_remaining / steps_left

    def act(self, obs: Observation, total_steps: int = 100) -> Action:
        self._update_estimates(obs)

        phase = obs.platform_health
        self._capi_counter += 1

        use_capi = False
        if phase == PlatformHealth.SIGNAL_LOSS:
            # Ration: fire CAPI every _CAPI_EVERY steps, only if epsilon allows
            if self._capi_counter % _CAPI_EVERY == 0 and obs.epsilon_remaining >= 3.0:
                use_capi = True

        if phase == PlatformHealth.NOMINAL:
            action = self._phase1_action(obs, total_steps)
        elif phase == PlatformHealth.SIGNAL_LOSS:
            action = self._phase2_action(obs, use_capi, total_steps)
        elif phase == PlatformHealth.ANDROMEDA_GLITCHED:
            action = self._phase3_action(obs, total_steps)
        else:  # PEAK_LOAD
            action = self._phase4_action(obs, total_steps)

        self._step_count += 1
        return action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    task_id: int = 7,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> dict:
    """
    Run one expert episode. Returns summary dict with score, avg_roas, steps.
    """
    env  = MetaSignalEnv()
    bot  = ExpertBot()
    obs  = env.reset(task_id=task_id, seed=seed)
    bot.reset()
    cfg  = get_task_config(task_id)
    total_steps = cfg.max_steps

    if verbose:
        print(f"[START] task={task_id} seed={seed} budget={obs.total_budget_remaining:.0f} epsilon={obs.epsilon_remaining:.1f} total_steps={total_steps}")

    step_num = 0
    while not env.state().is_done:
        action = bot.act(obs, total_steps=total_steps)
        result = env.step(action)
        obs    = result.observation

        if verbose:
            print(
                f"[STEP] step={obs.step:3d} day={obs.day:3d} "
                f"phase={obs.platform_health.value:20s} "
                f"roas={result.info.step_roas:.3f} "
                f"eps={obs.epsilon_remaining:.2f} "
                f"capi={'Y' if action.use_capi else 'N'} "
                f"reward={result.reward:.3f}"
            )
        step_num += 1

    grader = env.compute_final_score()
    if verbose:
        print(f"[END] score={grader.score:.4f} breakdown={grader.breakdown}")

    avg_roas = (
        grader.summary.get("avg_roas")
        or grader.summary.get("overall_avg_roas")
        or grader.summary.get("phase3_avg_roas")
        or 0.0
    )
    return {
        "task_id":    task_id,
        "seed":       seed,
        "score":      grader.score,
        "avg_roas":   avg_roas,
        "steps":      step_num,
        "breakdown":  grader.breakdown,
        "summary":    grader.summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one expert episode.")
    p.add_argument("--task",    type=int, default=7, choices=[5, 6, 7])
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_episode(task_id=args.task, seed=args.seed, verbose=args.verbose)
    print(f"\nFinal score: {result['score']:.4f}  avg_roas: {result['avg_roas']:.3f}")
