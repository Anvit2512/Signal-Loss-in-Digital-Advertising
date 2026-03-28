"""
Layer 4 -- Core Environment Logic

MetaSignalEnv wires together:
  - CriteoSnapshot  (data)
  - PrivacyEngine   (noise + epsilon budget)
  - TaskDefinition  (per-task config)
  - Graders         (tasks.py)

Public interface matches OpenEnv spec:
  reset(task_id, seed)   -> Observation
  step(action)           -> StepResult
  state()                -> EpisodeState
  compute_final_score()  -> GraderResult
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from app.data_loader import (
    CAMPAIGN_NAMES,
    get_snapshot,
)
from app.models import (
    Action,
    AttributionMethod,
    CampaignStats,
    EpisodeState,
    GraderResult,
    Observation,
    PrivacyRegime,
    StepInfo,
    StepResult,
)
from app.privacy import PrivacyEngine, regulatory_penalty
from app.tasks import get_task_config, grade_task1, grade_task2, grade_task3

# ---------------------------------------------------------------------------
# Episode constants
# ---------------------------------------------------------------------------

ROWS_PER_STEP      = 100    # Criteo rows consumed per step
ROAS_MULTIPLIER    = 25.0   # $ revenue per $ spent × CVR
                            # camp_feed CVR ~0.085 → ROAS ~2.1 (above target 1.5)
                            # camp_reels CVR ~0.036 → ROAS ~0.9
                            # camp_stories CVR ~0.020 → ROAS ~0.5

# Placement label derived from campaign id
_PLACEMENT: Dict[str, str] = {
    "camp_feed":    "feed",
    "camp_reels":   "reels",
    "camp_stories": "stories",
}


class MetaSignalEnv:
    """
    Core Meta-Signal environment.

    One instance per server process. reset() starts a fresh episode;
    step() advances it. State is held in memory between calls.
    """

    def __init__(self) -> None:
        self._snapshot      = get_snapshot()
        self._state:        Optional[EpisodeState]     = None
        self._privacy:      Optional[PrivacyEngine]    = None
        self._alloc_history: List[Dict[str, float]]    = []   # per-step allocations

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: int, seed: Optional[int] = None) -> Observation:
        """
        Start a fresh episode for the given task.

        seed controls:
          - The random start index into the Criteo snapshot
          - The PrivacyEngine RNG (Laplace noise sequence)

        Returns the first Observation (step 0, no action taken yet).
        """
        cfg = get_task_config(task_id)

        # Derive start_row so we never run out of rows mid-episode
        rng = np.random.default_rng(seed)
        max_start = self._snapshot.total_rows - cfg.max_steps * ROWS_PER_STEP
        start_row = int(rng.integers(0, max(1, max_start + 1)))

        # Privacy engine seeded separately so noise is reproducible
        noise_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        forced_regime = (
            PrivacyRegime.MINIMAL_DATA
            if cfg.privacy_regime == PrivacyRegime.MINIMAL_DATA
            else None
        )
        self._privacy = PrivacyEngine(
            initial_epsilon=cfg.initial_epsilon,
            seed=noise_seed,
            forced_regime=forced_regime,
        )

        self._state = EpisodeState(
            task_id=task_id,
            step=0,
            total_steps=cfg.max_steps,
            start_row=start_row,
            budget_initial=cfg.initial_budget,
            budget_remaining=cfg.initial_budget,
            epsilon_initial=cfg.initial_epsilon,
            epsilon_remaining=cfg.initial_epsilon,
            privacy_regime=self._privacy.regime,
            regulatory_violations=0,
            history=[],
            is_done=False,
        )
        self._alloc_history = []

        return self._build_initial_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Advance the episode by one step.

        1. Validate and clip allocations to remaining budget
        2. Consume epsilon for feature usage + attribution choice
        3. Pull ROWS_PER_STEP rows from the snapshot
        4. Compute true conversions per campaign (ground truth, hidden from agent)
        5. Compute agent revenue and step ROAS (from true data)
        6. Compute oracle ROAS (best possible with same spend)
        7. Add Laplace noise to conversion counts for next observation
        8. Compute regulatory penalty and reward
        9. Build and store StepResult; advance state
        10. Fire Task 2 privacy update if at step 3
        """
        if self._state is None or self._state.is_done:
            raise RuntimeError("Call reset() before step(), or episode is already done.")

        cfg = get_task_config(self._state.task_id)

        # --- 1. Clip allocations to remaining budget ---
        allocations = self._clip_allocations(
            action.allocations, self._state.budget_remaining
        )
        actual_spend = sum(allocations.values())
        self._alloc_history.append(dict(allocations))

        # --- 2. Consume epsilon ---
        epsilon_cost = self._privacy.consume(action.feature_mask, action.attribution)

        # --- 3. Pull rows from snapshot ---
        row_start = self._state.start_row + self._state.step * ROWS_PER_STEP

        # --- 4. True conversions per campaign ---
        true_conversions: Dict[str, int] = {}
        impressions:      Dict[str, int] = {}
        for camp in CAMPAIGN_NAMES:
            labels = self._snapshot.campaign_labels(camp, row_start, ROWS_PER_STEP)
            true_conversions[camp] = int(labels.sum())
            impressions[camp]      = len(labels)

        # --- 5. Agent revenue (from true CVR × allocation × multiplier) ---
        true_cvr: Dict[str, float] = {
            c: true_conversions[c] / max(impressions[c], 1)
            for c in CAMPAIGN_NAMES
        }
        revenue: Dict[str, float] = {
            c: allocations.get(c, 0.0) * true_cvr[c] * ROAS_MULTIPLIER
            for c in CAMPAIGN_NAMES
        }
        total_revenue = sum(revenue.values())
        step_roas     = total_revenue / actual_spend if actual_spend > 0 else 0.0

        # --- 6. Oracle ROAS (all spend on best campaign) ---
        best_camp  = max(true_cvr, key=true_cvr.get)
        oracle_roas = true_cvr[best_camp] * ROAS_MULTIPLIER

        # --- 7. Noisy observations ---
        noisy_conversions: Dict[str, float] = {
            c: max(0.0, self._privacy.add_noise(true_conversions[c]))
            for c in CAMPAIGN_NAMES
        }

        # --- 8. Regulatory penalty + reward ---
        penalty      = regulatory_penalty(action.feature_mask, cfg.max_features)
        reg_violation = len(action.feature_mask) > cfg.max_features
        if reg_violation:
            self._state.regulatory_violations += 1

        budget_fraction = self._privacy.budget_fraction_remaining
        reward = (
            step_roas       * 0.7
            - penalty       * 2.0
            + budget_fraction * 0.1
        )

        # --- 9. Build StepResult ---
        self._state.step           += 1
        self._state.budget_remaining = max(0.0, self._state.budget_remaining - actual_spend)
        self._state.epsilon_remaining = self._privacy.epsilon_remaining
        self._state.privacy_regime    = self._privacy.regime

        # Task 2: fire privacy update after processing step 3 (step counter is now 3)
        if self._state.task_id == 2 and self._state.step == 3:
            self._privacy.force_high_noise()
            self._state.privacy_regime    = self._privacy.regime
            self._state.epsilon_remaining = self._privacy.epsilon_remaining

        done = (
            self._state.step >= self._state.total_steps
            or self._state.budget_remaining <= 0.0
        )
        self._state.is_done = done

        # Warning
        warning: Optional[str] = None
        if self._privacy.epsilon_remaining < 0.2:
            warning = (
                f"epsilon={self._privacy.epsilon_remaining:.3f} -- "
                "signal severely degraded"
            )
        elif self._privacy.regime == PrivacyRegime.HIGH_NOISE:
            warning = "Privacy update active -- noise level elevated"

        obs = Observation(
            step=self._state.step,
            campaigns=self._build_campaign_stats(
                allocations, impressions, noisy_conversions, true_cvr
            ),
            total_budget_remaining=self._state.budget_remaining,
            epsilon_remaining=self._privacy.epsilon_remaining,
            privacy_regime=self._privacy.regime,
            available_features=self._privacy.available_features(),
            regulatory_violation=reg_violation,
            warning=warning,
        )

        info = StepInfo(
            step_roas=round(step_roas, 6),
            oracle_roas=round(oracle_roas, 6),
            epsilon_cost=round(epsilon_cost, 6),
            regulatory_penalty=round(penalty, 6),
            true_conversions=true_conversions,
            budget_fraction_remaining=round(budget_fraction, 6),
        )

        result = StepResult(observation=obs, reward=round(reward, 6), done=done, info=info)
        self._state.history.append(result)
        return result

    # ------------------------------------------------------------------
    # state / score
    # ------------------------------------------------------------------

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def compute_final_score(self) -> GraderResult:
        """Route to the correct task grader and return a 0.0-1.0 score."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")

        task_id = self._state.task_id
        self._state.is_done = True

        if task_id == 1:
            result = grade_task1(self._state, self._alloc_history)
        elif task_id == 2:
            result = grade_task2(self._state, self._alloc_history, self._privacy)
        elif task_id == 3:
            result = grade_task3(self._state, self._alloc_history, self._privacy)
        else:
            raise ValueError(f"Unknown task_id {task_id}")

        self._state.final_score = result.score
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clip_allocations(
        self,
        allocations: Dict[str, float],
        budget_remaining: float,
    ) -> Dict[str, float]:
        """
        Ensure allocations are non-negative and don't exceed remaining budget.
        Clips proportionally if over budget; fills missing campaigns with 0.
        """
        clipped = {c: max(0.0, allocations.get(c, 0.0)) for c in CAMPAIGN_NAMES}
        total = sum(clipped.values())
        if total > budget_remaining and total > 0:
            scale = budget_remaining / total
            clipped = {c: v * scale for c, v in clipped.items()}
        return clipped

    def _build_campaign_stats(
        self,
        allocations:       Dict[str, float],
        impressions:       Dict[str, int],
        noisy_conversions: Dict[str, float],
        true_cvr:          Dict[str, float],
    ) -> List[CampaignStats]:
        """Build the CampaignStats list for the Observation."""
        stats = []
        for camp in CAMPAIGN_NAMES:
            spend    = allocations.get(camp, 0.0)
            noisy_c  = noisy_conversions[camp]
            n_imps   = max(impressions[camp], 1)
            noisy_cvr = noisy_c / n_imps
            est_roas  = noisy_cvr * ROAS_MULTIPLIER

            stats.append(CampaignStats(
                campaign_id=camp,
                placement=_PLACEMENT[camp],
                impressions=impressions[camp],
                spend=round(spend, 4),
                noisy_conversions=round(noisy_c, 4),
                estimated_roas=round(est_roas, 4),
                ctr=round(true_cvr[camp], 6),   # CTR: no noise, observable
            ))
        return stats

    def _build_initial_observation(self) -> Observation:
        """
        Step-0 observation before any action is taken.
        Campaigns show zero spend and zero noisy conversions.
        """
        return Observation(
            step=0,
            campaigns=[
                CampaignStats(
                    campaign_id=camp,
                    placement=_PLACEMENT[camp],
                    impressions=0,
                    spend=0.0,
                    noisy_conversions=0.0,
                    estimated_roas=0.0,
                    ctr=0.0,
                )
                for camp in CAMPAIGN_NAMES
            ],
            total_budget_remaining=self._state.budget_remaining,
            epsilon_remaining=self._privacy.epsilon_remaining,
            privacy_regime=self._privacy.regime,
            available_features=self._privacy.available_features(),
            regulatory_violation=False,
            warning=None,
        )


# ---------------------------------------------------------------------------
# Smoke test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = MetaSignalEnv()

    for task_id in [1, 2, 3]:
        cfg  = get_task_config(task_id)
        obs  = env.reset(task_id=task_id, seed=42)
        print(f"\n{'='*55}")
        print(f"Task {task_id}: {cfg.name}")
        print(f"  start_row={env.state().start_row}  "
              f"budget={obs.total_budget_remaining}  "
              f"epsilon={obs.epsilon_remaining}  "
              f"regime={obs.privacy_regime.value}")
        print(f"  available_features (first 5): {obs.available_features[:5]}")

        rewards = []
        for step_n in range(cfg.max_steps):
            # Spread budget evenly across remaining steps
            steps_left = max(1, cfg.max_steps - step_n)
            per_camp = obs.total_budget_remaining / (steps_left * 3)
            action = Action(
                allocations={c: per_camp for c in CAMPAIGN_NAMES},
                attribution=AttributionMethod.LAST_CLICK,
                feature_mask=["I1"] if cfg.max_features == 1 else ["I1", "I2"],
            )
            result = env.step(action)
            rewards.append(result.reward)
            obs = result.observation
            if result.done:
                break

        score = env.compute_final_score()
        print(f"  steps={env.state().step}  "
              f"avg_reward={sum(rewards)/len(rewards):.3f}  "
              f"final_score={score.score:.4f}")
        print(f"  breakdown: {score.breakdown}")
