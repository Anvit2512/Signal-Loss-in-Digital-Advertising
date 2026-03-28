"""
Layer 4 -- Task Definitions and Graders

Three tasks with distinct difficulty levels and grading criteria.
All graders return continuous scores in [0.0, 1.0] with partial credit.

Task 1 -- Budget Optimisation    (Easy)    10 steps, clean signal
Task 2 -- Noisy Signal Recovery  (Medium)  15 steps, signal degrades at step 3
Task 3 -- Privacy Frontier       (Hard)    15 steps, one feature max from start
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

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
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[int, TaskDefinition] = {
    1: TaskDefinition(
        task_id=1,
        name="Budget Optimisation",
        description=(
            "Three campaigns with different conversion rates are visible under "
            "standard (low) noise. Learn within 10 steps that camp_feed has the "
            "highest ROAS and progressively shift budget toward it."
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
            "still performs, then maintain ROAS above target for the remaining 12 "
            "steps using only high-noise aggregated signals."
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
}


def get_task_config(task_id: int) -> TaskDefinition:
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task_id {task_id}. Choose 1, 2, or 3.")
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
) -> float:
    """
    Score for whether allocations progressively shifted toward target_camp.

    Compares the target camp's budget share in the first half of the episode
    vs the second half. Score = 1.0 if second-half share >= 0.7 of total.
    Score = 0.0 if share never changes.
    """
    if len(alloc_history) < 2:
        return 0.0

    shares = []
    for alloc in alloc_history:
        total = sum(alloc.values())
        if total <= 0:
            shares.append(0.0)
        else:
            shares.append(alloc.get(target_camp, 0.0) / total)

    mid = len(shares) // 2
    first_half_avg = float(np.mean(shares[:mid])) if mid > 0 else 0.0
    second_half_avg = float(np.mean(shares[mid:])) if shares[mid:] else 0.0

    # Reward final share magnitude and upward trend
    share_score  = min(1.0, second_half_avg / 0.7)   # 0.7 share = full score
    trend_bonus  = min(0.2, max(0.0, second_half_avg - first_half_avg))
    return min(1.0, share_score + trend_bonus)


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
    40% -- did budget allocation progressively shift toward camp_feed?
    """
    cfg = TASK_CONFIGS[1]
    avg_roas   = _avg_step_roas(state)
    roas_s     = _roas_score(avg_roas, cfg.target_roas)
    trend_s    = _allocation_trend_score(alloc_history, "camp_feed")

    score = roas_s * 0.6 + trend_s * 0.4

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
        },
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

    return GraderResult(
        task_id=3,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "roas_score":       round(roas_s, 4),
            "compliance_score": round(compliance_s, 4),
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
    )
