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
    MarketTrendGenerator,
    get_snapshot,
)
from app.models import (
    Action,
    AttributionMethod,
    CampaignStats,
    EpisodeState,
    GraderResult,
    LearningStatus,
    MarketTrend,
    Observation,
    PlatformHealth,
    PrivacyRegime,
    StepInfo,
    StepResult,
)
from app.privacy import PrivacyEngine, regulatory_penalty
from app.tasks import (
    AUDIT_STEP,
    get_task_config,
    grade_task1,
    grade_task2,
    grade_task3,
    grade_task4,
    grade_task5,
    grade_task6,
    grade_task7,
)

# ---------------------------------------------------------------------------
# Episode constants
# ---------------------------------------------------------------------------

ROWS_PER_STEP      = 100    # Criteo rows consumed per step
ROAS_MULTIPLIER    = 25.0   # $ revenue per $ spent × CVR
                            # camp_feed CVR ~0.085 → ROAS ~2.1 (above target 1.5)
                            # camp_reels CVR ~0.036 → ROAS ~0.9
                            # camp_stories CVR ~0.020 → ROAS ~0.5

# Q4 Gauntlet phase boundaries (day numbers, inclusive)
Q4_TASKS           = frozenset({5, 6, 7})
PHASE1_END         = 20    # days 1-20: clean signal
PHASE2_END         = 50    # days 21-50: ATT blackout
PHASE3_END         = 80    # days 51-80: Andromeda glitch
# days 81-100: Black Friday

ATT_NOISE_MULT     = 3.0   # Phase 2 structural ATT signal loss multiplier
PHASE4_NOISE_MULT  = 2.0   # Phase 4 additional noise volatility multiplier
ANDROMEDA_THRESHOLD = 0.20  # allocation share change that triggers a learning reset
LEARNING_RESET_CVR  = 0.30  # CVR fraction during reset window
LEARNING_RESET_DAYS = 7     # steps of degraded CVR after a reset
OVERSPEND_THRESHOLD = 1.5   # pacing_speed above this risks overspend bug
OVERSPEND_PROB      = 0.30  # probability of midnight overspend bug firing
OVERSPEND_MULT      = 2.5   # how much budget gets dumped when it fires
SELF_IMPROVE_ROAS   = 3.0   # ROAS threshold for self-improvement trigger
SELF_IMPROVE_STREAK = 5     # consecutive steps above threshold to escalate

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
        self._snapshot       = get_snapshot()
        self._state:         Optional[EpisodeState]     = None
        self._privacy:       Optional[PrivacyEngine]    = None
        self._alloc_history: List[Dict[str, float]]     = []
        self._legal_codes:   List[Optional[str]]        = []
        # Q4 Gauntlet state
        self._market_trend:         Optional[MarketTrendGenerator] = None
        self._current_phase:        int   = 0     # 1-4 for Q4 tasks, 0 otherwise
        self._learning_reset_countdown: int = 0   # steps remaining in Andromeda reset window
        self._difficulty_level:     int   = 0     # self-improvement escalation counter
        self._episode_rng:          Optional[np.random.Generator] = None  # for overspend rolls

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
        self._legal_codes   = []

        # Task 4: pre-determine which campaign the regulator will flag
        if task_id == 4:
            camp_list = list(CAMPAIGN_NAMES)
            self._state.flagged_campaign = camp_list[start_row % len(camp_list)]

        # Q4 Gauntlet initialisation
        if task_id in Q4_TASKS:
            trend_seed = int(rng.integers(0, 2**31)) if seed is not None else None
            self._market_trend           = MarketTrendGenerator(seed=trend_seed or 42)
            self._current_phase          = 1
            self._learning_reset_countdown = 0
            self._episode_rng            = np.random.default_rng(noise_seed)
        else:
            self._market_trend           = None
            self._current_phase          = 0
            self._learning_reset_countdown = 0
            self._episode_rng            = None

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

        # Task 4: enforce zero spend on flagged campaign once audit has fired
        audit_already_fired = (
            self._state.task_id == 4
            and self._state.audit_fired_at is not None
        )
        if audit_already_fired and self._state.flagged_campaign:
            allocations[self._state.flagged_campaign] = 0.0

        # Q4: Andromeda glitch check (Phase 3) — before appending to history
        is_q4 = self._state.task_id in Q4_TASKS
        if is_q4 and self._current_phase == 3 and self._alloc_history:
            self._check_andromeda_glitch(allocations, cfg.initial_budget)

        # Q4: overspend bug (Phase 4)
        if is_q4 and self._current_phase == 4 and action.pacing_speed > OVERSPEND_THRESHOLD:
            assert self._episode_rng is not None
            if float(self._episode_rng.random()) < OVERSPEND_PROB:
                allocations = {
                    c: v * OVERSPEND_MULT for c, v in allocations.items()
                }
                allocations = self._clip_allocations(allocations, self._state.budget_remaining)
                self._state.overspend_events += 1

        actual_spend = sum(allocations.values())
        self._alloc_history.append(dict(allocations))
        self._legal_codes.append(action.legal_reason_code)

        # --- 2. Consume epsilon (wires CAPI cost if use_capi=True) ---
        epsilon_cost = self._privacy.consume(
            action.feature_mask, action.attribution, use_capi=action.use_capi
        )
        if action.use_capi:
            self._state.capi_calls_used += 1

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

        # Q4: apply learning reset CVR suppression if active
        if is_q4 and self._learning_reset_countdown > 0:
            for camp in CAMPAIGN_NAMES:
                true_cvr[camp]          = true_cvr[camp] * LEARNING_RESET_CVR
                true_conversions[camp]  = int(true_conversions[camp] * LEARNING_RESET_CVR)
            self._learning_reset_countdown -= 1

        # --- Market shift (Task 2, step 9+): camp_reels CVR temporarily doubles ---
        # Fires from the 9th action onward (step counter == 8 before increment).
        # An agent that preserved epsilon budget can still read the noisy signal;
        # one that burned through epsilon is flying blind.
        market_shift_active = (
            self._state.task_id == 2
            and self._state.step >= 8
        )
        if market_shift_active:
            true_cvr["camp_reels"] = true_cvr["camp_reels"] * 2.0
            # Keep true_conversions consistent so noisy signal reflects the shift
            true_conversions["camp_reels"] = int(
                true_conversions["camp_reels"] * 2.0
            )

        # --- Correlation penalty: >70% spend concentration drops other campaigns' CTR by 15% ---
        # Mirrors real auction dynamics where aggressive Feed bidding wins the same
        # user twice, suppressing Reels/Stories performance.  Forces portfolio balance
        # and makes the naive "put everything on camp_feed" answer subtly wrong.
        correlation_penalty_active = False
        if actual_spend > 0:
            camp_shares = {
                c: allocations.get(c, 0.0) / actual_spend for c in CAMPAIGN_NAMES
            }
            dominant_camp = max(camp_shares, key=camp_shares.get)
            if camp_shares[dominant_camp] > 0.70:
                correlation_penalty_active = True
                for c in CAMPAIGN_NAMES:
                    if c != dominant_camp:
                        true_cvr[c] = true_cvr[c] * 0.85
                        true_conversions[c] = int(true_conversions[c] * 0.85)

        revenue: Dict[str, float] = {
            c: allocations.get(c, 0.0) * true_cvr[c] * ROAS_MULTIPLIER
            for c in CAMPAIGN_NAMES
        }
        total_revenue = sum(revenue.values())
        step_roas     = total_revenue / actual_spend if actual_spend > 0 else 0.0

        # --- 6. Oracle ROAS (all spend on best campaign) ---
        best_camp  = max(true_cvr, key=true_cvr.get)
        oracle_roas = true_cvr[best_camp] * ROAS_MULTIPLIER

        # --- 7. Noisy observations (CAPI bypasses noise entirely) ---
        noisy_conversions: Dict[str, float] = {
            c: max(0.0, self._privacy.add_noise(true_conversions[c], use_capi=action.use_capi))
            for c in CAMPAIGN_NAMES
        }

        # --- 8. Regulatory penalty + reward ---
        penalty       = regulatory_penalty(action.feature_mask, cfg.max_features)
        reg_violation = len(action.feature_mask) > cfg.max_features

        # Task 4: extra penalty if agent spent on flagged campaign after audit fired
        if audit_already_fired and self._state.flagged_campaign:
            original_flagged_spend = action.allocations.get(self._state.flagged_campaign, 0.0)
            if original_flagged_spend > 0.0:
                penalty       += 4.0   # quadratic-style fixed penalty per violation
                reg_violation  = True

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

        # Task 4: fire regulatory audit after processing step AUDIT_STEP
        if (
            self._state.task_id == 4
            and self._state.step == AUDIT_STEP
            and self._state.audit_fired_at is None
        ):
            self._state.audit_fired_at = self._state.step

        # Q4 Gauntlet: phase transitions keyed on current day (= step after increment)
        if is_q4:
            self._update_q4_phase(self._state.step)
            self._state.capi_calls_used = self._privacy.capi_calls

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
        if market_shift_active:
            shift_msg = "Market shift: camp_reels CVR spiked (viral trend) -- consider reallocating"
            warning = shift_msg if warning is None else f"{warning} | {shift_msg}"
        if correlation_penalty_active:
            penalty_msg = "Concentration penalty active: >70% spend on one campaign depresses other campaigns' CTR by 15%"
            warning = penalty_msg if warning is None else f"{warning} | {penalty_msg}"

        audit_now_active = (
            self._state.task_id == 4
            and self._state.audit_fired_at is not None
        )
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
            audit_active=audit_now_active,
            flagged_campaign=self._state.flagged_campaign if audit_now_active else None,
            warning=warning,
            # Q4 Gauntlet narrative fields
            day=self._state.step,
            platform_health=self._get_platform_health(),
            learning_status=(
                LearningStatus.RESET if self._learning_reset_countdown > 0
                else LearningStatus.OPTIMIZED
            ),
            market_trend=(
                MarketTrend(self._market_trend.get(self._state.step))
                if self._market_trend else MarketTrend.RISING
            ),
        )

        info = StepInfo(
            step_roas=round(step_roas, 6),
            oracle_roas=round(oracle_roas, 6),
            epsilon_cost=round(epsilon_cost, 6),
            regulatory_penalty=round(penalty, 6),
            true_conversions=true_conversions,
            budget_fraction_remaining=round(budget_fraction, 6),
            correlation_penalty_active=correlation_penalty_active,
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
        elif task_id == 4:
            result = grade_task4(self._state, self._alloc_history, self._legal_codes)
        elif task_id == 5:
            result = grade_task5(self._state, self._alloc_history, self._privacy)
        elif task_id == 6:
            result = grade_task6(self._state, self._alloc_history, self._privacy)
        elif task_id == 7:
            result = grade_task7(self._state, self._alloc_history, self._privacy)
        else:
            raise ValueError(f"Unknown task_id {task_id}")

        self._state.final_score = result.score
        self._self_improve(task_id)
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
        noise_scale = self._privacy.noise_scale
        stats = []
        for camp in CAMPAIGN_NAMES:
            spend     = allocations.get(camp, 0.0)
            noisy_c   = noisy_conversions[camp]
            n_imps    = max(impressions[camp], 1)
            noisy_cvr = noisy_c / n_imps
            est_roas  = noisy_cvr * ROAS_MULTIPLIER
            ci_half   = 1.96 * noise_scale
            ci        = (round(noisy_c - ci_half, 4), round(noisy_c + ci_half, 4))

            stats.append(CampaignStats(
                campaign_id=camp,
                placement=_PLACEMENT[camp],
                impressions=impressions[camp],
                spend=round(spend, 4),
                noisy_conversions=round(noisy_c, 4),
                estimated_roas=round(est_roas, 4),
                ctr=round(true_cvr[camp], 6),   # CTR: no noise, observable
                confidence_interval=ci,
            ))
        return stats

    # ------------------------------------------------------------------
    # Q4 Gauntlet phase helpers
    # ------------------------------------------------------------------

    def _get_phase(self, day: int) -> int:
        if day <= PHASE1_END:
            return 1
        if day <= PHASE2_END:
            return 2
        if day <= PHASE3_END:
            return 3
        return 4

    def _update_q4_phase(self, day: int) -> None:
        """Fire phase transitions when the day counter crosses a boundary."""
        new_phase = self._get_phase(day)
        if new_phase == self._current_phase:
            return

        old_phase = self._current_phase
        self._current_phase = new_phase

        if old_phase == 1 and new_phase == 2:
            # ATT fires: structural noise that epsilon cannot fix
            self._privacy.force_att_loss(ATT_NOISE_MULT)

        elif old_phase == 2 and new_phase == 3:
            # ATT normalises; Andromeda glitch activates (no noise change needed here —
            # the glitch is about allocation stability, handled in _check_andromeda_glitch)
            self._privacy.clear_att_loss()

        elif old_phase == 3 and new_phase == 4:
            # Black Friday: noise volatility doubles via Task2-style multiplier
            self._privacy.force_high_noise()   # reuses 8x multiplier from existing method

    def _get_platform_health(self) -> PlatformHealth:
        if self._current_phase == 0:
            return PlatformHealth.NOMINAL
        if self._current_phase == 1:
            return PlatformHealth.NOMINAL
        if self._current_phase == 2:
            return PlatformHealth.SIGNAL_LOSS
        if self._current_phase == 3:
            return PlatformHealth.ANDROMEDA_GLITCHED
        return PlatformHealth.PEAK_LOAD

    def _check_andromeda_glitch(
        self, allocations: Dict[str, float], total_budget: float
    ) -> None:
        """
        Compare current allocations to the previous step.
        If any campaign share changes by > 20% of total budget, trigger a
        7-step learning reset (CVR drops to 30% of normal).
        """
        if not self._alloc_history or self._learning_reset_countdown > 0:
            return
        prev = self._alloc_history[-1]
        denom = max(total_budget, 1.0)
        for camp in CAMPAIGN_NAMES:
            delta = abs(allocations.get(camp, 0.0) - prev.get(camp, 0.0)) / denom
            if delta > ANDROMEDA_THRESHOLD:
                self._learning_reset_countdown = LEARNING_RESET_DAYS
                self._state.learning_resets   += 1
                break

    def _self_improve(self, task_id: int) -> None:
        """
        After episode ends: if the agent beat ROAS_THRESHOLD for
        SELF_IMPROVE_STREAK consecutive steps, escalate difficulty next episode.
        Difficulty increases the base noise multiplier on the next reset.
        """
        if self._state is None or not self._state.history:
            return
        recent = self._state.history[-SELF_IMPROVE_STREAK:]
        if (
            len(recent) == SELF_IMPROVE_STREAK
            and all(r.info.step_roas > SELF_IMPROVE_ROAS for r in recent)
        ):
            self._difficulty_level = min(self._difficulty_level + 1, 5)

    # ------------------------------------------------------------------

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
                    confidence_interval=(0.0, 0.0),
                )
                for camp in CAMPAIGN_NAMES
            ],
            total_budget_remaining=self._state.budget_remaining,
            epsilon_remaining=self._privacy.epsilon_remaining,
            privacy_regime=self._privacy.regime,
            available_features=self._privacy.available_features(),
            regulatory_violation=False,
            audit_active=False,
            flagged_campaign=None,
            warning=None,
        )


# ---------------------------------------------------------------------------
# Smoke test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = MetaSignalEnv()

    for task_id in [1, 2, 3, 4, 5, 6, 7]:
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
