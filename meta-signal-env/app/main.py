"""
Layer 5 -- FastAPI Server

Exposes the Meta-Signal environment as a REST API.
One shared MetaSignalEnv instance protected by a threading lock
so concurrent HTTP requests cannot corrupt episode state.

Endpoints
---------
GET  /health          liveness probe
GET  /tasks           all task definitions + action schema
POST /reset           start a new episode
POST /step            advance the episode by one step
GET  /state           current episode state
POST /grader          compute final score for active episode
POST /baseline        run the GPT-based baseline and return all three scores
GET  /                web UI (HTML)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.env import MetaSignalEnv
from app.models import (
    Action,
    AttributionMethod,
    BaselineResult,
    EpisodeState,
    GraderRequest,
    GraderResult,
    Observation,
    ResetRequest,
    SimulateRequest,
    SimulateResult,
    SimulateStepTrace,
    StepResult,
    TaskDefinition,
)
from app.tasks import TASK_CONFIGS, get_task_config

# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Meta-Signal",
    description=(
        "Privacy-constrained ad budget optimisation environment. "
        "An agent manages spend across three campaigns using only "
        "noisy, aggregated conversion signals."
    ),
    version="1.0.0",
)

_env  = MetaSignalEnv()
_lock = threading.Lock()

# Mount static files (index.html, CSS, JS)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.get("/tasks", response_model=List[TaskDefinition], tags=["environment"])
def get_tasks() -> List[TaskDefinition]:
    """Return all three task definitions including grader weights and action schema."""
    return list(TASK_CONFIGS.values())


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation, tags=["environment"])
def reset(request: ResetRequest) -> Observation:
    """
    Start a fresh episode.

    - task_id: 1 (Budget Opt), 2 (Noisy Recovery), 3 (Privacy Frontier)
    - seed: optional int for fully reproducible episodes
    """
    with _lock:
        obs = _env.reset(task_id=request.task_id, seed=request.seed)
    return obs


@app.post("/step", response_model=StepResult, tags=["environment"])
def step(action: Action) -> StepResult:
    """
    Advance the episode by one step.

    Provide allocations ($ per campaign), attribution method, and feature mask.
    Returns noisy observation, reward, done flag, and diagnostic info.
    """
    with _lock:
        try:
            result = _env.step(action)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state", response_model=EpisodeState, tags=["environment"])
def get_state() -> EpisodeState:
    """Return the full current episode state including step history."""
    with _lock:
        try:
            return _env.state()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))


@app.post("/grader", response_model=GraderResult, tags=["environment"])
def grader(request: GraderRequest) -> GraderResult:
    """
    Compute the final score for the active episode.

    Can be called before done=True for a mid-episode score,
    or after to get the official final grade.
    """
    with _lock:
        try:
            state = _env.state()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if state.task_id != request.task_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Active episode is task {state.task_id}, "
                    f"but grader request is for task {request.task_id}. "
                    "Call /reset first."
                ),
            )
        result = _env.compute_final_score()
    return result


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = frozenset({"equal", "greedy", "conservative"})
_CAMPAIGNS        = ["camp_feed", "camp_reels", "camp_stories"]


def _build_allocations(
    strategy:       str,
    per_step_budget: float,
    obs_campaigns:   list,
    greedy_best:     list,  # mutable single-element list so we can update state
) -> dict:
    """
    Return allocations dict for one step given a strategy.

    equal        — even three-way split.
    greedy       — 80% to best noisy-signal campaign, 10% each to the others.
                   Picks the best campaign freshly each step from the observation.
    conservative — fixed 60 / 25 / 15 split (feed / reels / stories), stays
                   below the 70% concentration threshold to avoid the
                   correlation penalty.
    """
    if strategy == "equal":
        share = per_step_budget / 3.0
        return {c: share for c in _CAMPAIGNS}

    if strategy == "greedy":
        # Update best from current noisy signal if we have observations
        if obs_campaigns:
            best = max(obs_campaigns, key=lambda c: c.noisy_conversions)
            greedy_best[0] = best.campaign_id
        best_camp = greedy_best[0]
        alloc = {c: per_step_budget * 0.10 for c in _CAMPAIGNS}
        alloc[best_camp] = per_step_budget * 0.80
        return alloc

    # conservative
    return {
        "camp_feed":    per_step_budget * 0.60,
        "camp_reels":   per_step_budget * 0.25,
        "camp_stories": per_step_budget * 0.15,
    }


@app.post("/simulate", response_model=SimulateResult, tags=["environment"])
def simulate(request: SimulateRequest) -> SimulateResult:
    """
    Run a complete episode with a built-in hardcoded strategy.

    No coding required — pick a strategy string and get back the full score
    and a step-by-step trace.  Useful for exploring the environment behaviour
    and baseline comparisons without writing an agent.

    **Strategies**
    - `equal`        — even budget split across all three campaigns every step.
    - `greedy`       — 80% to whichever campaign had the best noisy signal last step.
    - `conservative` — fixed 60/25/15 split (Feed/Reels/Stories), deliberately
                        staying below the 70% concentration threshold to avoid
                        the auction-overlap correlation penalty.
    """
    if request.strategy not in _VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown strategy '{request.strategy}'. "
                f"Valid options: {sorted(_VALID_STRATEGIES)}"
            ),
        )

    # Use a fresh env so we never clobber the shared episode state
    sim_env = MetaSignalEnv()
    cfg     = get_task_config(request.task_id)
    obs     = sim_env.reset(task_id=request.task_id, seed=request.seed)

    trace:        list[SimulateStepTrace] = []
    greedy_best   = ["camp_feed"]   # mutable state for greedy tracker

    while not sim_env.state().is_done:
        state          = sim_env.state()
        steps_left     = max(1, cfg.max_steps - state.step)
        per_step_budget = obs.total_budget_remaining / steps_left

        allocations = _build_allocations(
            request.strategy, per_step_budget, obs.campaigns, greedy_best
        )

        # Task 4: honour the suspension
        if state.task_id == 4 and state.audit_fired_at is not None and state.flagged_campaign:
            allocations[state.flagged_campaign] = 0.0

        features = ["I1"] if cfg.max_features >= 1 else []
        action   = Action(
            allocations=allocations,
            attribution=AttributionMethod.LAST_CLICK,
            feature_mask=features,
            legal_reason_code=(
                "REGULATORY_HOLD"
                if state.task_id == 4 and state.audit_fired_at is not None
                else None
            ),
        )

        result = sim_env.step(action)
        obs    = result.observation

        trace.append(SimulateStepTrace(
            step=obs.step,
            allocations=allocations,
            step_roas=result.info.step_roas,
            oracle_roas=result.info.oracle_roas,
            epsilon_remaining=obs.epsilon_remaining,
            privacy_regime=obs.privacy_regime.value,
            reward=result.reward,
            correlation_penalty_active=result.info.correlation_penalty_active,
            warning=obs.warning,
        ))

    grader_result = sim_env.compute_final_score()

    return SimulateResult(
        task_id=request.task_id,
        strategy=request.strategy,
        score=grader_result.score,
        grader=grader_result,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

@app.post("/baseline", response_model=BaselineResult, tags=["evaluation"])
def baseline() -> BaselineResult:
    """
    Run the GPT-4o-mini baseline agent across all three tasks (seed=42).
    Returns scores and per-task GraderResult breakdowns.

    Requires OPENAI_API_KEY environment variable.
    """
    try:
        from baseline import run_baseline
        return run_baseline(seed=42)
    except ImportError:
        raise HTTPException(status_code=500, detail="baseline.py not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Web UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["system"])
def web() -> str:
    """Serve the terminal-style dashboard frontend."""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return html_file.read_text()
    return "<h1>Meta-Signal</h1><p><a href='/docs'>API Docs</a></p>"
