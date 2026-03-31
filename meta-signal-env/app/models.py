"""
Layer 2 -- Pydantic Models

All typed contracts for the Meta-Signal environment.
Nothing in this file contains logic -- pure data shapes only.

Import order reflects data flow:
  CampaignStats -> Observation -> Action -> StepResult -> EpisodeState
  TaskDefinition and GraderResult are standalone.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PrivacyRegime(str, Enum):
    """
    Controls the noise level applied to conversion signals.
    Transitions are triggered by epsilon depletion or task configuration.
    """
    STANDARD     = "standard"      # Normal noise -- signal is readable
    HIGH_NOISE   = "high_noise"    # Noise jumps at Task 2 step 3 or low epsilon
    MINIMAL_DATA = "minimal_data"  # Task 3 regulatory constraint: 1 feature max
    EXHAUSTED    = "exhausted"     # Epsilon near zero -- signal is near random


class AttributionMethod(str, Enum):
    """
    How the agent wants conversions attributed.
    Probabilistic is more accurate but costs extra epsilon per step.
    """
    LAST_CLICK    = "last_click"    # Epsilon cost: 0.0 extra
    PROBABILISTIC = "probabilistic" # Epsilon cost: +0.20 per step


# ---------------------------------------------------------------------------
# Per-campaign noisy stats (what the agent observes per campaign per step)
# ---------------------------------------------------------------------------


class CampaignStats(BaseModel):
    """
    Noisy, aggregated view of one campaign for the current step.
    The agent receives this -- it never sees raw labels.
    """
    campaign_id:       str
    placement:         str   = Field(description="Ad format: feed | reels | stories")
    impressions:       int   = Field(ge=0, description="Rows processed this step")
    spend:             float = Field(ge=0.0, description="Budget allocated this step ($)")
    noisy_conversions:   float              = Field(description="True conversions + Laplace noise")
    estimated_roas:      float              = Field(description="noisy_conversions / spend; inf if spend=0")
    ctr:                 float              = Field(ge=0.0, description="Click-through rate -- observable, no noise applied")
    confidence_interval: Tuple[float, float] = Field(
        description="95% CI around noisy_conversions. Narrows when epsilon is high, widens as it depletes."
    )


# ---------------------------------------------------------------------------
# Observation -- what the agent sees at the start of each step
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """
    Full observation returned by reset() and each step().
    This is the agent's only window into the environment.
    """
    step:                  int            = Field(ge=0)
    campaigns:             List[CampaignStats]
    total_budget_remaining: float         = Field(ge=0.0)
    epsilon_remaining:     float          = Field(ge=0.0)
    privacy_regime:        PrivacyRegime
    available_features:    List[str]      = Field(
        description="Feature names the agent is allowed to use in feature_mask this step"
    )
    regulatory_violation:  bool           = Field(
        default=False,
        description="True if the last action breached the feature limit"
    )
    audit_active:          bool           = Field(
        default=False,
        description="Task 4: regulatory audit is in progress"
    )
    flagged_campaign:      Optional[str]  = Field(
        default=None,
        description="Task 4: campaign suspended by regulator -- must receive zero spend"
    )
    warning:               Optional[str]  = Field(
        default=None,
        description="Human-readable warning e.g. 'epsilon below 0.2'"
    )


# ---------------------------------------------------------------------------
# Action -- what the agent submits on each step
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """
    Agent decision for one step.

    allocations:    spend per campaign, must be >= 0 and sum <= budget_remaining
    attribution:    last_click (free) or probabilistic (+0.20 epsilon)
    feature_mask:   subset of I1-I13 / C1-C26 to use; each costs 0.05 epsilon
    """
    allocations:    Dict[str, float] = Field(
        description="Campaign ID -> spend amount ($). Must sum <= budget_remaining."
    )
    attribution:    AttributionMethod = Field(
        default=AttributionMethod.LAST_CLICK
    )
    feature_mask:       List[str]      = Field(
        default_factory=list,
        description="Feature names from I1-I13 / C1-C26. Each costs 0.05 epsilon."
    )
    halted_campaigns:   List[str]      = Field(
        default_factory=list,
        description="Task 4: campaigns the agent is halting per regulatory order."
    )
    legal_reason_code:  Optional[str]  = Field(
        default=None,
        description="Task 4: legal justification code e.g. GDPR_ART17, DPA_NOTICE, REGULATORY_HOLD."
    )

    @model_validator(mode="after")
    def allocations_non_negative(self) -> "Action":
        for camp, amount in self.allocations.items():
            if amount < 0:
                raise ValueError(
                    f"Allocation for '{camp}' is negative ({amount}). "
                    "All allocations must be >= 0."
                )
        return self


# ---------------------------------------------------------------------------
# StepInfo -- internal diagnostics attached to each StepResult
# ---------------------------------------------------------------------------


class StepInfo(BaseModel):
    """
    Diagnostic breakdown returned in StepResult.info.
    Agents may use this for logging/debugging; graders use it for scoring.
    """
    step_roas:           float  = Field(description="Revenue / spend this step")
    oracle_roas:         float  = Field(description="Best possible ROAS given hidden labels")
    epsilon_cost:        float  = Field(description="Epsilon consumed this step")
    regulatory_penalty:  float  = Field(ge=0.0, description="Penalty applied this step")
    true_conversions:    Dict[str, int] = Field(
        description="Actual conversions per campaign -- for grader use only, hidden from agent"
    )
    budget_fraction_remaining: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# StepResult -- full return value of step()
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """
    Returned by env.step(). Contains everything needed to continue or grade.
    """
    observation: Observation
    reward:      float
    done:        bool
    info:        StepInfo


# ---------------------------------------------------------------------------
# EpisodeState -- full state object, used by /state and /grader
# ---------------------------------------------------------------------------


class EpisodeState(BaseModel):
    """
    Complete episode record.
    Passed to compute_final_score() at episode end.
    """
    task_id:              int
    step:                 int
    total_steps:          int
    start_row:            int   = Field(description="Random start index in Criteo snapshot")
    budget_initial:       float
    budget_remaining:     float
    epsilon_initial:      float
    epsilon_remaining:    float
    privacy_regime:       PrivacyRegime
    regulatory_violations: int           = Field(ge=0, default=0)
    history:              List[StepResult] = Field(default_factory=list)
    final_score:          Optional[float]  = Field(default=None, ge=0.0, le=1.0)
    is_done:              bool             = False
    audit_fired_at:       Optional[int]   = Field(default=None, description="Task 4: step when audit fired")
    flagged_campaign:     Optional[str]   = Field(default=None, description="Task 4: suspended campaign")


# ---------------------------------------------------------------------------
# TaskDefinition -- describes a task to any agent via /tasks
# ---------------------------------------------------------------------------


class TaskDefinition(BaseModel):
    """
    Self-describing task spec returned by GET /tasks.
    Lets any agent understand what it needs to do without reading source code.
    """
    task_id:         int
    name:            str
    description:     str
    max_steps:       int
    initial_budget:  float
    initial_epsilon: float
    privacy_regime:  PrivacyRegime
    target_roas:     float = Field(description="ROAS the agent should aim to exceed")
    max_features:    int   = Field(description="Max features allowed per step (Task 3 = 1)")
    grader_weights:  Dict[str, float] = Field(
        description="Score component names and their weights, must sum to 1.0"
    )


# ---------------------------------------------------------------------------
# ResetRequest -- body for POST /reset
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: int   = Field(ge=1, le=4)
    seed:    Optional[int] = Field(
        default=None,
        description="Fix the random start index for reproducible episodes"
    )


# ---------------------------------------------------------------------------
# GraderRequest -- body for POST /grader
# ---------------------------------------------------------------------------


class GraderRequest(BaseModel):
    task_id: int = Field(ge=1, le=4)


# ---------------------------------------------------------------------------
# GraderResult -- response from POST /grader
# ---------------------------------------------------------------------------


class GraderResult(BaseModel):
    """
    Final score returned by POST /grader after an episode ends.
    Always in [0.0, 1.0] -- never binary.
    """
    task_id:   int
    score:     float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(
        description="Per-component scores matching TaskDefinition.grader_weights"
    )
    summary:   Dict[str, float] = Field(
        description="Episode stats: avg_roas, violations, epsilon_used, steps_completed"
    )


# ---------------------------------------------------------------------------
# BaselineResult -- response from POST /baseline
# ---------------------------------------------------------------------------


class BaselineResult(BaseModel):
    """
    Response from POST /baseline -- scores for all three tasks in one call.
    """
    model:    str = Field(description="Model used for baseline inference")
    seed:     int
    scores:   Dict[str, float] = Field(description="task_1, task_2, task_3 -> score")
    details:  Dict[str, GraderResult]
