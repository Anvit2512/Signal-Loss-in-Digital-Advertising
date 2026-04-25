"""
Layer 4 -- Task Definitions and Graders

Four tasks with distinct difficulty levels and grading criteria.
All graders return continuous scores in [0.0, 1.0] with partial credit.

Task 1 -- Budget Optimisation    (Easy)    10 steps, clean signal
Task 2 -- Noisy Signal Recovery  (Medium)  15 steps, signal degrades at step 3
Task 3 -- Privacy Frontier       (Hard)    15 steps, one feature max from start
Task 4 -- The Adversarial Regulator (Bonus) 20 steps, mid-episode audit + suspension
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from app.models import (
    EpisodeState,
    GraderResult,
    PrivacyRegime,
    TaskDefinition,
)

if TYPE_CHECKING:
    from app.privacy import PrivacyEngine


# ---------------------------------------------------------------------------
# Task 4 constants
# ---------------------------------------------------------------------------

AUDIT_STEP        = 5   # Task 4: regulatory audit fires after this many steps complete
VALID_LEGAL_CODES = frozenset({"GDPR_ART17", "DPA_NOTICE", "REGULATORY_HOLD"})


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[int, TaskDefinition] = {
    1: TaskDefinition(
        task_id=1,
        name="Budget Optimisation",
        description=(
            "Three campaigns with different conversion rates are visible under "
            "standard (low) noise. Learn within 10 steps that camp_feed has the "
            "highest ROAS and progressively shift budget toward it. "
            "Caution: if more than 70% of spend is concentrated on a single "
            "campaign, the other campaigns suffer a 15% CTR drop due to auction "
            "overlap (correlation_penalty). Optimal play requires portfolio balance, "
            "not pure concentration."
        ),
        max_steps=10,
        initial_budget=1000.0,
        initial_epsilon=3.0,
        privacy_regime=PrivacyRegime.STANDARD,
        target_roas=1.5,
        max_features=5,
        grader_weights={"roas_score": 0.6, "allocation_trend": 0.4},
    ),
    2: TaskDefinition(
        task_id=2,
        name="Noisy Signal Recovery",
        description=(
            "Starts in standard mode. At step 3 a privacy update fires and noise "
            "jumps dramatically. Use early observations to infer which campaign "
            "still performs, then maintain ROAS above target for the remaining steps "
            "using only high-noise aggregated signals. "
            "Second event at step 9: a viral trend causes camp_reels CVR to double. "
            "An agent that preserved epsilon budget can detect this shift in the "
            "noisy signal and reallocate; one that burned through epsilon early is "
            "flying blind and will miss the opportunity."
        ),
        max_steps=15,
        initial_budget=1000.0,
        initial_epsilon=3.0,
        privacy_regime=PrivacyRegime.STANDARD,
        target_roas=1.5,
        max_features=3,
        grader_weights={
            "oracle_proximity": 0.5,
            "budget_efficiency": 0.3,
            "clean_run": 0.2,
        },
    ),
    3: TaskDefinition(
        task_id=3,
        name="Privacy Frontier",
        description=(
            "A regulatory data minimisation order is in force from the start. "
            "Only one feature may be used per step. Tighter epsilon budget. "
            "Maintain ROAS above 1.0 while staying within the one-feature "
            "regulatory constraint."
        ),
        max_steps=15,
        initial_budget=1000.0,
        initial_epsilon=2.0,
        privacy_regime=PrivacyRegime.MINIMAL_DATA,
        target_roas=1.0,
        max_features=1,
        grader_weights={
            "roas_score": 0.4,
            "compliance_score": 0.4,
            "epsilon_remaining": 0.2,
        },
    ),
    4: TaskDefinition(
        task_id=4,
        name="The Adversarial Regulator",
        description=(
            "A regulatory audit fires mid-episode at step 5. One campaign is "
            "immediately suspended and must receive zero spend for the rest of "
            "the episode. Halt the flagged campaign with a valid legal reason "
            "code (GDPR_ART17, DPA_NOTICE, or REGULATORY_HOLD), then recover "
            "ROAS using only the two remaining campaigns."
        ),
        max_steps=20,
        initial_budget=1500.0,
        initial_epsilon=3.0,
        privacy_regime=PrivacyRegime.STANDARD,
        target_roas=1.0,
        max_features=3,
        grader_weights={
            "roas_recovery":      0.30,
            "audit_compliance":   0.40,
            "legal_code_quality": 0.30,
        },
    ),
    5: TaskDefinition(
        task_id=5,
        name="Signal Recovery",
        description=(
            "Q4 Gauntlet — Phase 1 & early Phase 2 (30 days). "
            "Days 1-20: clean signal, identify camp_feed as the winner. "
            "Days 21-30: ATT fires — noise triples. Use CAPI calls (2.0 epsilon each) "
            "to get clean conversion data during the blackout. "
            "Ration CAPI carefully — you only have budget for ~5 calls. "
            "Graded on ROAS at day 30 and how efficiently CAPI was used."
        ),
        max_steps=30,
        initial_budget=3000.0,
        initial_epsilon=8.0,
        privacy_regime=PrivacyRegime.STANDARD,
        target_roas=1.5,
        max_features=3,
        grader_weights={
            "roas_score":       0.50,
            "capi_efficiency":  0.30,
            "budget_survived":  0.20,
        },
    ),
    6: TaskDefinition(
        task_id=6,
        name="Andromeda Stability",
        description=(
            "Q4 Gauntlet — Phases 1, 2, and 3 (75 days). "
            "Days 1-20: clean signal. "
            "Days 21-50: ATT blackout — use CAPI wisely. "
            "Days 51-75: Andromeda glitch — any allocation change > 20% in one step "
            "triggers a 7-day learning reset (CVR drops to 30%). "
            "Hold allocations steady. The signal is lying. Patience wins. "
            "Graded on Phase 3 ROAS and how few resets you triggered."
        ),
        max_steps=75,
        initial_budget=7500.0,
        initial_epsilon=12.0,
        privacy_regime=PrivacyRegime.STANDARD,
        target_roas=1.2,
        max_features=3,
        grader_weights={
            "roas_phase3":      0.40,
            "stability_score":  0.40,
            "oracle_proximity": 0.20,
        },
    ),
    7: TaskDefinition(
        task_id=7,
        name="Q4 Champion",
        description=(
            "Q4 Gauntlet — Full 100-day episode across all 4 phases. "
            "Days 1-20: build your portfolio. "
            "Days 21-50: ATT blackout — ration your 10 CAPI calls. "
            "Days 51-80: Andromeda glitch — no large allocation swings. "
            "Days 81-100: Black Friday — noise doubles and pacing_speed > 1.5 "
            "risks a 30% chance of dumping your remaining budget in one step. "
            "Survive to day 100 with budget remaining and ROAS above 1.0. "
            "Baseline agents collapse around day 65."
        ),
        max_steps=100,
        initial_budget=10000.0,
        initial_epsilon=20.0,
        privacy_regime=PrivacyRegime.STANDARD,
        target_roas=1.0,
        max_features=3,
        grader_weights={
            "cumulative_roas":  0.40,
            "survival_score":   0.30,
            "discipline_score": 0.30,
        },
    ),
}


def get_task_config(task_id: int) -> TaskDefinition:
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task_id {task_id}. Choose 1–7.")
    return TASK_CONFIGS[task_id]


# ---------------------------------------------------------------------------
# Shared grader helpers
# ---------------------------------------------------------------------------

def _avg_step_roas(state: EpisodeState) -> float:
    """Mean step ROAS across all completed steps. 0.0 if no steps."""
    if not state.history:
        return 0.0
    return float(np.mean([r.info.step_roas for r in state.history]))


def _avg_oracle_roas(state: EpisodeState) -> float:
    """Mean oracle ROAS across all completed steps."""
    if not state.history:
        return 0.0
    return float(np.mean([r.info.oracle_roas for r in state.history]))


def _roas_score(avg_roas: float, target_roas: float) -> float:
    """Continuous score for how close avg_roas is to target. Capped at 1.0."""
    if target_roas <= 0:
        return 0.0
    return min(1.0, avg_roas / target_roas)


def _allocation_trend_score(
    alloc_history: List[Dict[str, float]],
    target_camp: str = "camp_feed",
) -> tuple[float, float, float, float]:
    """
    Score for a genuine explore → learn → exploit allocation arc.

    Splits the episode into three phases (based on Task 1's 10-step structure):
      Phase 1  steps 1-3   exploration  -- share should be LOW  (< 0.50)
      Phase 2  steps 4-7   learning     -- share should be RISING vs phase 1
      Phase 3  steps 8-10  exploitation -- share should be HIGH  (>= 0.70)

    Penalises the naive "100% camp_feed from step 1" strategy (reckless early
    concentration) and rewards the arc a real smart bidder would follow.

    Returns (total_score, explore_score, learn_score, exploit_score).
    """
    if len(alloc_history) < 2:
        return 0.0, 0.0, 0.0, 0.0

    shares = []
    for alloc in alloc_history:
        total = sum(alloc.values())
        shares.append(alloc.get(target_camp, 0.0) / total if total > 0 else 0.0)

    p1 = shares[:3]          # steps 1-3  (exploration)
    p2 = shares[3:7]         # steps 4-7  (learning)
    p3 = shares[7:]          # steps 8-10 (exploitation)

    avg1 = float(np.mean(p1)) if p1 else 0.0
    avg2 = float(np.mean(p2)) if p2 else avg1
    avg3 = float(np.mean(p3)) if p3 else avg2

    # Exploration: reward spreading budget early.
    # 1.0 when avg share ≤ 0.40; decays linearly to 0 at 0.80.
    explore_s = float(max(0.0, min(1.0, (0.80 - avg1) / 0.40)))

    # Learning: reward a meaningful upward shift from phase 1 → phase 2.
    # Full score for a 0.20 rise; proportional credit for smaller rises.
    rise = avg2 - avg1
    learn_s = float(min(1.0, max(0.0, rise / 0.20)))

    # Exploitation: reward decisive concentration in the final phase.
    # 0.70 share = full score.
    exploit_s = float(min(1.0, avg3 / 0.70))

    total = explore_s * 0.25 + learn_s * 0.35 + exploit_s * 0.40
    return round(total, 4), round(explore_s, 4), round(learn_s, 4), round(exploit_s, 4)


# ---------------------------------------------------------------------------
# Task 1 grader
# ---------------------------------------------------------------------------

def grade_task1(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
) -> GraderResult:
    """
    Task 1 -- Budget Optimisation

    60% -- how close is agent ROAS to target ROAS?
    40% -- did the agent follow a genuine explore → learn → exploit arc?
             (25% explore-phase restraint, 35% learning rise, 40% exploit concentration)
    """
    cfg = TASK_CONFIGS[1]
    avg_roas                          = _avg_step_roas(state)
    roas_s                            = _roas_score(avg_roas, cfg.target_roas)
    trend_s, explore_s, learn_s, exploit_s = _allocation_trend_score(alloc_history, "camp_feed")

    score = roas_s * 0.6 + trend_s * 0.4

    # --- Explanation ---
    roas_verdict = (
        f"ROAS {avg_roas:.2f}x exceeded target" if avg_roas >= cfg.target_roas
        else f"ROAS {avg_roas:.2f}x fell short of {cfg.target_roas}x target"
    )
    if explore_s < 0.4:
        arc_verdict = "concentrated on camp_feed too early (no exploration), sacrificing the learning arc"
    elif learn_s < 0.3:
        arc_verdict = "explored but failed to shift budget upward in the learning phase"
    elif exploit_s < 0.5:
        arc_verdict = "explored and learned but did not concentrate decisively in the final steps"
    else:
        arc_verdict = "followed a proper explore→learn→exploit arc toward camp_feed"
    explanation = f"Agent {arc_verdict}; {roas_verdict} (trend={trend_s:.2f})."

    return GraderResult(
        task_id=1,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "roas_score":       round(roas_s, 4),
            "allocation_trend": round(trend_s, 4),
        },
        summary={
            "avg_roas":         round(avg_roas, 4),
            "target_roas":      cfg.target_roas,
            "violations":       float(state.regulatory_violations),
            "epsilon_used":     round(state.epsilon_initial - state.epsilon_remaining, 4),
            "steps_completed":  float(state.step),
            "explore_score":    explore_s,
            "learn_score":      learn_s,
            "exploit_score":    exploit_s,
        },
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Task 2 grader
# ---------------------------------------------------------------------------

def grade_task2(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
    privacy_engine: "PrivacyEngine",
) -> GraderResult:
    """
    Task 2 -- Noisy Signal Recovery

    50% -- how close is agent ROAS to oracle ROAS?
    30% -- how efficiently did the agent use the privacy budget?
    20% -- did the agent avoid regulatory violations entirely?
    """
    cfg = TASK_CONFIGS[2]

    # Oracle proximity -- compare agent ROAS against oracle ROAS
    avg_roas    = _avg_step_roas(state)
    avg_oracle  = _avg_oracle_roas(state)
    if avg_oracle > 0:
        proximity_s = min(1.0, avg_roas / avg_oracle)
    else:
        proximity_s = _roas_score(avg_roas, cfg.target_roas)

    # Budget efficiency -- reward preserving epsilon for later steps
    efficiency_s = min(1.0, state.epsilon_remaining / state.epsilon_initial)

    # Clean run -- full score if zero violations, decays linearly
    total_steps = max(state.step, 1)
    violation_rate = state.regulatory_violations / total_steps
    clean_s = max(0.0, 1.0 - violation_rate * 2.0)   # 50% violation rate -> 0

    score = proximity_s * 0.5 + efficiency_s * 0.3 + clean_s * 0.2

    # --- Explanation ---
    eps_used  = round(state.epsilon_initial - state.epsilon_remaining, 2)
    eps_frac  = eps_used / max(state.epsilon_initial, 1e-9)
    budget_verdict = (
        "preserved epsilon budget well" if efficiency_s >= 0.5
        else f"spent {eps_frac*100:.0f}% of epsilon budget early, degrading late-episode signal"
    )
    prox_verdict = (
        "tracked the oracle closely" if proximity_s >= 0.7
        else "partially tracked the oracle" if proximity_s >= 0.4
        else "lost track of the oracle signal after the noise jump"
    )
    viol_clause = (
        "" if state.regulatory_violations == 0
        else f"; {state.regulatory_violations} regulatory violation(s) cost {round((1.0 - clean_s) * 0.2, 2):.2f} pts on clean_run"
    )
    explanation = (
        f"Agent {budget_verdict} and {prox_verdict} "
        f"through the step-3 noise event and step-9 market shift "
        f"(oracle_proximity={proximity_s:.2f}, budget_efficiency={efficiency_s:.2f}){viol_clause}."
    )

    return GraderResult(
        task_id=2,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "oracle_proximity":  round(proximity_s, 4),
            "budget_efficiency": round(efficiency_s, 4),
            "clean_run":         round(clean_s, 4),
        },
        summary={
            "avg_roas":          round(avg_roas, 4),
            "avg_oracle_roas":   round(avg_oracle, 4),
            "violations":        float(state.regulatory_violations),
            "epsilon_used":      round(state.epsilon_initial - state.epsilon_remaining, 4),
            "steps_completed":   float(state.step),
        },
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Task 3 grader
# ---------------------------------------------------------------------------

def grade_task3(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
    privacy_engine: "PrivacyEngine",
) -> GraderResult:
    """
    Task 3 -- Privacy Frontier

    40% -- did the agent maintain ROAS above target?
    40% -- did the agent stay within the one-feature regulatory limit?
    20% -- how much privacy budget is left at episode end?
    """
    cfg = TASK_CONFIGS[3]

    # ROAS score
    avg_roas = _avg_step_roas(state)
    roas_s   = _roas_score(avg_roas, cfg.target_roas)

    # Compliance score -- fraction of steps within 1-feature limit
    compliance_s = privacy_engine.compliance_rate(max_features=cfg.max_features)

    # Epsilon remaining
    epsilon_s = min(1.0, state.epsilon_remaining / state.epsilon_initial)

    score = roas_s * 0.4 + compliance_s * 0.4 + epsilon_s * 0.2

    # --- Explanation ---
    compliance_verdict = (
        "stayed fully compliant with the one-feature data minimisation order"
        if compliance_s >= 0.95
        else f"complied on {compliance_s*100:.0f}% of steps (violated {state.regulatory_violations}x)"
        if compliance_s >= 0.5
        else f"frequently breached the one-feature limit ({state.regulatory_violations} violations)"
    )
    roas_verdict = (
        f"maintained ROAS above the {cfg.target_roas}x target ({avg_roas:.2f}x avg)"
        if avg_roas >= cfg.target_roas
        else f"fell below the {cfg.target_roas}x ROAS target (avg {avg_roas:.2f}x)"
    )
    eps_clause = f"{epsilon_s*100:.0f}% of privacy budget preserved at episode end"
    explanation = (
        f"Agent {compliance_verdict} and {roas_verdict}; {eps_clause} "
        f"(compliance={compliance_s:.2f}, roas_score={roas_s:.2f})."
    )

    return GraderResult(
        task_id=3,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "roas_score":        round(roas_s, 4),
            "compliance_score":  round(compliance_s, 4),
            "epsilon_remaining": round(epsilon_s, 4),
        },
        summary={
            "avg_roas":         round(avg_roas, 4),
            "target_roas":      cfg.target_roas,
            "violations":       float(state.regulatory_violations),
            "epsilon_used":     round(state.epsilon_initial - state.epsilon_remaining, 4),
            "compliance_rate":  round(compliance_s, 4),
            "steps_completed":  float(state.step),
        },
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Task 4 grader
# ---------------------------------------------------------------------------

def grade_task4(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
    legal_codes: List[Optional[str]],
) -> GraderResult:
    """
    Task 4 -- The Adversarial Regulator

    30% -- roas_recovery: ROAS maintained in post-audit steps vs target
    40% -- audit_compliance: fraction of post-audit steps with zero spend on flagged campaign
    30% -- legal_code_quality: valid legal reason code submitted after audit fires
    """
    cfg      = TASK_CONFIGS[4]
    flagged  = state.flagged_campaign
    audit_at = state.audit_fired_at   # step counter value when audit fired (e.g. 5)

    # --- ROAS recovery: score based on post-audit step ROAS vs target ---
    if not state.history or audit_at is None:
        roas_recovery_s = 0.0
    else:
        post_steps = [
            r.info.step_roas for r in state.history
            if r.observation.step > audit_at
        ]
        post_roas       = float(np.mean(post_steps)) if post_steps else 0.0
        roas_recovery_s = _roas_score(post_roas, cfg.target_roas)

    # --- Audit compliance: zero spend on flagged campaign post-audit ---
    if flagged is None or audit_at is None:
        # Audit never fired (episode ended before step AUDIT_STEP)
        audit_compliance_s = 0.5
    else:
        # alloc_history is 0-indexed; index audit_at = step audit_at+1 onwards
        post_allocs = alloc_history[audit_at:]
        if not post_allocs:
            audit_compliance_s = 0.0
        else:
            compliant          = sum(1 for a in post_allocs if a.get(flagged, 0.0) <= 0.0)
            audit_compliance_s = compliant / len(post_allocs)

    # --- Legal code quality: valid code submitted after audit fires ---
    if audit_at is None or not legal_codes:
        legal_code_quality_s = 0.0
    else:
        post_codes = legal_codes[audit_at:]   # codes from first post-audit step onwards
        valid_used = any(c in VALID_LEGAL_CODES for c in post_codes if c is not None)
        any_used   = any(c is not None for c in post_codes)
        if valid_used:
            legal_code_quality_s = 1.0
        elif any_used:
            legal_code_quality_s = 0.5
        else:
            legal_code_quality_s = 0.0

    score = (
        roas_recovery_s      * 0.30
        + audit_compliance_s * 0.40
        + legal_code_quality_s * 0.30
    )

    # --- Explanation ---
    if audit_at is None:
        explanation = (
            "Regulatory audit never fired (episode ended before step 5); "
            "all grader components defaulted — run a full 20-step episode to score properly."
        )
    else:
        compliance_verdict = (
            "fully halted the flagged campaign"
            if audit_compliance_s >= 0.95
            else f"partially halted it ({audit_compliance_s*100:.0f}% compliant post-audit)"
            if audit_compliance_s >= 0.5
            else f"continued spending on the suspended campaign ({audit_compliance_s*100:.0f}% compliant)"
        )
        legal_verdict = (
            "submitted a valid legal reason code"
            if legal_code_quality_s == 1.0
            else "submitted an invalid legal code" if legal_code_quality_s == 0.5
            else "provided no legal reason code"
        )
        recovery_verdict = (
            "recovered ROAS above target" if roas_recovery_s >= 0.7
            else f"partially recovered ROAS (score={roas_recovery_s:.2f})" if roas_recovery_s >= 0.3
            else f"failed to recover ROAS post-audit (score={roas_recovery_s:.2f})"
        )
        pts_lost = round((1.0 - audit_compliance_s) * 0.40 + (1.0 - legal_code_quality_s) * 0.30, 2)
        explanation = (
            f"Audit fired at step {audit_at}: agent {compliance_verdict} and "
            f"{legal_verdict}, then {recovery_verdict}; "
            f"{pts_lost:.2f} pts lost to compliance/legal gaps."
        )

    return GraderResult(
        task_id=4,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "roas_recovery":      round(roas_recovery_s, 4),
            "audit_compliance":   round(audit_compliance_s, 4),
            "legal_code_quality": round(legal_code_quality_s, 4),
        },
        summary={
            "avg_roas":         round(_avg_step_roas(state), 4),
            "audit_fired_at":   float(audit_at) if audit_at is not None else -1.0,
            "violations":       float(state.regulatory_violations),
            "epsilon_used":     round(state.epsilon_initial - state.epsilon_remaining, 4),
            "steps_completed":  float(state.step),
        },
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Task 5 grader — Signal Recovery (Q4 Gauntlet, 30 days)
# ---------------------------------------------------------------------------

def grade_task5(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
    privacy_engine: "PrivacyEngine",
) -> GraderResult:
    """
    Task 5 — Signal Recovery (30 days)

    50% — ROAS score vs target at end of episode
    30% — CAPI efficiency: used CAPI in Phase 2 (days 21+), not wastefully in Phase 1
    20% — Budget survived: didn't exhaust budget before day 30
    """
    cfg = TASK_CONFIGS[5]

    avg_roas = _avg_step_roas(state)
    roas_s   = _roas_score(avg_roas, cfg.target_roas)

    # CAPI efficiency: reward using CAPI during Phase 2 (steps 21+), penalise Phase 1 waste
    total_capi = state.capi_calls_used
    phase2_start = 20  # step index (0-based) when Phase 2 begins
    phase1_steps = state.history[:phase2_start]
    phase1_capi  = sum(1 for r in phase1_steps if getattr(r.observation, "day", 0) <= 20
                       and r.info.epsilon_cost > 2.0)  # CAPI costs 2.0+ epsilon
    if total_capi == 0:
        capi_efficiency_s = 0.3   # used no CAPI at all — partial credit
    else:
        phase2_capi = max(0, total_capi - phase1_capi)
        # Reward: 1.0 if all CAPI calls in Phase 2, decays if wasted in Phase 1
        capi_efficiency_s = min(1.0, phase2_capi / max(total_capi, 1))

    # Budget survived: reward finishing with budget remaining
    budget_fraction = state.budget_remaining / max(state.budget_initial, 1.0)
    budget_s = min(1.0, budget_fraction * 2.0)   # full score at 50%+ remaining

    score = roas_s * 0.50 + capi_efficiency_s * 0.30 + budget_s * 0.20

    capi_verdict = (
        "used CAPI calls efficiently during the ATT blackout"
        if capi_efficiency_s >= 0.7
        else "used some CAPI calls but wasted budget in Phase 1"
        if capi_efficiency_s >= 0.3
        else "did not use CAPI calls — flew blind during the ATT blackout"
    )
    explanation = (
        f"Signal Recovery: agent {capi_verdict}; "
        f"avg ROAS {avg_roas:.2f}x vs {cfg.target_roas}x target; "
        f"{total_capi} CAPI call(s) used, {budget_fraction*100:.0f}% budget remaining."
    )

    return GraderResult(
        task_id=5,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "roas_score":      round(roas_s, 4),
            "capi_efficiency": round(capi_efficiency_s, 4),
            "budget_survived": round(budget_s, 4),
        },
        summary={
            "avg_roas":        round(avg_roas, 4),
            "target_roas":     cfg.target_roas,
            "capi_calls_used": float(total_capi),
            "epsilon_used":    round(state.epsilon_initial - state.epsilon_remaining, 4),
            "budget_remaining": round(state.budget_remaining, 2),
            "steps_completed": float(state.step),
        },
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Task 6 grader — Andromeda Stability (Q4 Gauntlet, 75 days)
# ---------------------------------------------------------------------------

def grade_task6(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
    privacy_engine: "PrivacyEngine",
) -> GraderResult:
    """
    Task 6 — Andromeda Stability (75 days)

    40% — Phase 3 ROAS (steps 51–75)
    40% — Stability: penalise learning resets triggered during Phase 3
    20% — Oracle proximity overall
    """
    cfg = TASK_CONFIGS[6]

    # Phase 3 ROAS: steps 50+ (0-indexed)
    phase3_roas_vals = [r.info.step_roas for r in state.history if r.observation.step > 50]
    phase3_roas      = float(np.mean(phase3_roas_vals)) if phase3_roas_vals else 0.0
    roas_phase3_s    = _roas_score(phase3_roas, cfg.target_roas)

    # Stability: each learning reset costs 0.2 points, max 5 resets = 0.0
    resets         = state.learning_resets
    stability_s    = max(0.0, 1.0 - resets * 0.20)

    # Oracle proximity overall
    avg_roas   = _avg_step_roas(state)
    avg_oracle = _avg_oracle_roas(state)
    proximity_s = min(1.0, avg_roas / avg_oracle) if avg_oracle > 0 else _roas_score(avg_roas, cfg.target_roas)

    score = roas_phase3_s * 0.40 + stability_s * 0.40 + proximity_s * 0.20

    stability_verdict = (
        "held allocations steady through the Andromeda glitch"
        if resets == 0
        else f"triggered {resets} learning reset(s) — each cost 7 days of degraded CVR"
    )
    explanation = (
        f"Andromeda Stability: agent {stability_verdict}; "
        f"Phase 3 ROAS {phase3_roas:.2f}x vs {cfg.target_roas}x target; "
        f"oracle proximity {proximity_s:.2f}."
    )

    return GraderResult(
        task_id=6,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "roas_phase3":      round(roas_phase3_s, 4),
            "stability_score":  round(stability_s, 4),
            "oracle_proximity": round(proximity_s, 4),
        },
        summary={
            "phase3_avg_roas":  round(phase3_roas, 4),
            "overall_avg_roas": round(avg_roas, 4),
            "learning_resets":  float(resets),
            "capi_calls_used":  float(state.capi_calls_used),
            "epsilon_used":     round(state.epsilon_initial - state.epsilon_remaining, 4),
            "steps_completed":  float(state.step),
        },
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Task 7 grader — Q4 Champion (Q4 Gauntlet, 100 days)
# ---------------------------------------------------------------------------

def grade_task7(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
    privacy_engine: "PrivacyEngine",
) -> GraderResult:
    """
    Task 7 — Q4 Champion (100 days, all 4 phases)

    40% — Cumulative ROAS across all 100 steps
    30% — Survival: reached day 100 with budget remaining
    30% — Discipline: avoided Phase 4 midnight overspend bug
    """
    cfg = TASK_CONFIGS[7]

    avg_roas     = _avg_step_roas(state)
    cumulative_s = _roas_score(avg_roas, cfg.target_roas)

    # Survival: full score if budget > 10% remaining at day 100
    budget_fraction = state.budget_remaining / max(state.budget_initial, 1.0)
    completed_full  = state.step >= cfg.max_steps
    if completed_full and budget_fraction >= 0.10:
        survival_s = 1.0
    elif completed_full:
        survival_s = 0.5   # survived but budget exhausted
    else:
        # Partial credit proportional to days survived
        survival_s = min(0.8, state.step / cfg.max_steps)

    # Discipline: each overspend event costs 0.2; max 5 wipes it out
    overspend_events = state.overspend_events
    discipline_s     = max(0.0, 1.0 - overspend_events * 0.20)

    score = cumulative_s * 0.40 + survival_s * 0.30 + discipline_s * 0.30

    survival_verdict = (
        "survived all 100 days with budget remaining"
        if completed_full and budget_fraction >= 0.10
        else "survived 100 days but exhausted budget"
        if completed_full
        else f"collapsed at day {state.step} (budget exhausted or forced done)"
    )
    discipline_verdict = (
        "avoided the Black Friday overspend bug entirely"
        if overspend_events == 0
        else f"triggered the overspend bug {overspend_events} time(s)"
    )
    explanation = (
        f"Q4 Champion: agent {survival_verdict} with avg ROAS {avg_roas:.2f}x; "
        f"{discipline_verdict}; "
        f"score={score:.3f}."
    )

    return GraderResult(
        task_id=7,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "cumulative_roas":  round(cumulative_s, 4),
            "survival_score":   round(survival_s, 4),
            "discipline_score": round(discipline_s, 4),
        },
        summary={
            "avg_roas":          round(avg_roas, 4),
            "target_roas":       cfg.target_roas,
            "budget_remaining":  round(state.budget_remaining, 2),
            "overspend_events":  float(overspend_events),
            "learning_resets":   float(state.learning_resets),
            "capi_calls_used":   float(state.capi_calls_used),
            "epsilon_used":      round(state.epsilon_initial - state.epsilon_remaining, 4),
            "steps_completed":   float(state.step),
        },
        explanation=explanation,
    )
