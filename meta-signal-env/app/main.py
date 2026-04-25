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
POST /hint            Q4 Gauntlet: context-aware hint for current phase
POST /baseline        run the GPT-based baseline and return all three scores
GET  /                web UI (HTML)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
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
    PlatformHealth,
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
    return {"status": "healthy", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# OpenEnv compliance endpoints
# ---------------------------------------------------------------------------

@app.get("/metadata", tags=["system"])
def metadata() -> dict:
    return {
        "name": "meta-signal",
        "display_name": "Meta-Signal: Privacy-Constrained Ad Optimisation",
        "version": "1.0.0",
        "description": (
            "An RL environment where an AI agent allocates advertising budget across "
            "three campaigns but can only observe noisy, aggregated conversion data -- "
            "exactly how Meta's real ad system works after signal loss. "
            "Models differential privacy budget depletion, mid-episode regime changes, "
            "correlation penalties, market-shift events, and regulatory audit mechanics."
        ),
        "author": "Anvit2512",
        "tags": ["advertising", "differential-privacy", "reinforcement-learning", "budget-optimisation", "signal-loss"],
    }


@app.get("/schema", tags=["system"])
def schema() -> dict:
    return {
        "action": {
            "type": "object",
            "properties": {
                "allocations": {
                    "type": "object",
                    "description": "Dollar amount to spend per campaign. Keys: camp_feed, camp_reels, camp_stories",
                    "additionalProperties": {"type": "number", "minimum": 0},
                },
                "attribution": {
                    "type": "string",
                    "enum": ["last_click", "probabilistic"],
                    "default": "last_click",
                    "description": "last_click is free; probabilistic costs 0.20 epsilon extra",
                },
                "feature_mask": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature names from I1-I13 or C1-C26. Each costs 0.05 epsilon per step.",
                },
                "halted_campaigns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task 4: campaigns to halt per regulatory order",
                },
                "legal_reason_code": {
                    "type": "string",
                    "nullable": True,
                    "description": "Task 4: GDPR_ART17 | GDPR_ART21 | CCPA_OPT_OUT | COPPA",
                },
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "step": {"type": "integer"},
                "campaigns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "campaign_id": {"type": "string"},
                            "placement": {"type": "string"},
                            "impressions": {"type": "integer"},
                            "spend": {"type": "number"},
                            "noisy_conversions": {"type": "number"},
                            "estimated_roas": {"type": "number"},
                            "ctr": {"type": "number"},
                            "confidence_interval": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                },
                "total_budget_remaining": {"type": "number"},
                "epsilon_remaining": {"type": "number"},
                "privacy_regime": {"type": "string", "enum": ["standard", "high_noise", "minimal_data", "exhausted"]},
                "available_features": {"type": "array", "items": {"type": "string"}},
                "regulatory_violation": {"type": "boolean"},
                "audit_active": {"type": "boolean"},
                "flagged_campaign": {"type": "string", "nullable": True},
                "warning": {"type": "string", "nullable": True},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "step": {"type": "integer"},
                "is_done": {"type": "boolean"},
                "total_spend": {"type": "number"},
                "total_budget": {"type": "number"},
                "cumulative_roas": {"type": "number"},
            },
        },
    }


@app.post("/mcp", tags=["system"])
async def mcp(request: Request) -> dict:
    """Model Context Protocol (MCP) JSON-RPC 2.0 endpoint."""
    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "meta-signal", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": "reset", "description": "Start a new episode", "inputSchema": {"type": "object", "properties": {"task_id": {"type": "integer"}}}},
                    {"name": "step", "description": "Advance the episode by one step", "inputSchema": {"type": "object"}},
                ]
            },
        }
    # Default: return empty success
    return {"jsonrpc": "2.0", "id": req_id, "result": {}}


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
def reset(request: Optional[ResetRequest] = None) -> Observation:
    """
    Start a fresh episode.

    - task_id: 1 (Budget Opt), 2 (Noisy Recovery), 3 (Privacy Frontier)
    - seed: optional int for fully reproducible episodes
    """
    if request is None:
        request = ResetRequest()
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
# Hint  (Q4 Gauntlet Snorkel AI bonus)
# ---------------------------------------------------------------------------

# Hint templates keyed by PlatformHealth value
_HINTS: dict = {
    PlatformHealth.NOMINAL: {
        "phase": 1,
        "title": "Phase 1 — The Setup (Days 1–20)",
        "situation": "Signal is clean. Use these steps to identify which campaign has the best ROAS.",
        "advice": (
            "Spread budget across all three campaigns in the first few steps to learn their CVRs. "
            "camp_feed typically has the highest conversion rate (~8.5%). "
            "Once identified, progressively shift budget toward it — but stay below 70% to avoid "
            "the correlation penalty."
        ),
        "watch_for": "Correlation penalty: >70% spend on one campaign drops other campaigns' CTR by 15%.",
        "capi_advice": "No need for CAPI calls yet — signal is clean. Save your CAPI budget for Phase 2.",
    },
    PlatformHealth.SIGNAL_LOSS: {
        "phase": 2,
        "title": "Phase 2 — ATT Blackout (Days 21–50)",
        "situation": "iOS App Tracking Transparency has fired. Noise is 3× higher than normal. Epsilon cannot fix this.",
        "advice": (
            "The noisy signal is unreliable. Use CAPI calls (set use_capi=True, costs 2.0 epsilon each) "
            "to get clean conversion data. Ration carefully — you have limited epsilon. "
            "Aim for 1 CAPI call every 3–5 steps. Between calls, hold your allocations steady "
            "rather than reacting to corrupted signals."
        ),
        "watch_for": "Epsilon exhaustion: if epsilon drops below 0.5 you enter high_noise regime — avoid this.",
        "capi_advice": "Use CAPI now. This is what it is for. Do not save all calls for Phase 3.",
    },
    PlatformHealth.ANDROMEDA_GLITCHED: {
        "phase": 3,
        "title": "Phase 3 — Andromeda Glitch (Days 51–80)",
        "situation": (
            "The Andromeda update is live. Any allocation change greater than 20% of total budget "
            "in a single step triggers a 7-day learning reset — CVR drops to 30% of normal."
        ),
        "advice": (
            "Do NOT react to the noisy signal. Hold your allocations steady from Phase 2. "
            "The signal is still noisy and will mislead you. Patience is the correct strategy. "
            "If you must adjust, change by less than 20% of total budget per step. "
            "learning_status=Reset in the observation means you already triggered one — wait 7 steps."
        ),
        "watch_for": "learning_status field: if Reset, do not make any allocation changes for 7 steps.",
        "capi_advice": "CAPI is still useful here for clean data, but Phase 2 is when you need it most.",
    },
    PlatformHealth.PEAK_LOAD: {
        "phase": 4,
        "title": "Phase 4 — Black Friday Peak (Days 81–100)",
        "situation": (
            "Maximum traffic load. Noise volatility has doubled. "
            "If you set pacing_speed above 1.5, there is a 30% chance per step of a midnight "
            "budget dump — your remaining budget is spent in one step."
        ),
        "advice": (
            "Set pacing_speed=1.0 (default). Do not be greedy. "
            "A budget dump this late in the episode is catastrophic — there are not enough steps "
            "to recover. Prioritise survival over ROAS maximisation. "
            "Hold the allocation you built in Phase 2. Do not trigger Andromeda resets."
        ),
        "watch_for": "pacing_speed > 1.5 = 30% overspend risk per step. Not worth it.",
        "capi_advice": "Only use CAPI if you have epsilon to spare. Budget first.",
    },
}


@app.post("/hint", tags=["environment"])
def hint() -> dict:
    """
    Q4 Gauntlet: returns a context-aware strategic hint for the current episode phase.

    Reads the active episode state to determine which phase the agent is in,
    then returns tailored advice about what to do next.

    This simulates the Snorkel AI expert-in-the-loop mechanic — a domain expert
    who knows the environment's hidden mechanics and can guide the agent.

    Only meaningful for Q4 Gauntlet tasks (5, 6, 7). Returns a generic hint
    for tasks 1–4.
    """
    with _lock:
        try:
            state = _env.state()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if state.task_id not in (5, 6, 7):
            return {
                "phase": 0,
                "title": "Standard Task",
                "situation": f"Task {state.task_id} does not use Q4 Gauntlet phases.",
                "advice": (
                    "Identify the best-performing campaign via noisy ROAS signals. "
                    "Preserve epsilon budget. Follow the explore → learn → exploit arc."
                ),
                "watch_for": "Correlation penalty (>70% concentration) and epsilon depletion.",
                "capi_advice": "N/A — use_capi is a Q4 Gauntlet mechanic.",
                "current_day": state.step,
                "epsilon_remaining": state.epsilon_remaining,
                "budget_remaining": state.budget_remaining,
            }

        # Get latest observation to read platform_health
        if not state.history:
            platform = PlatformHealth.NOMINAL
        else:
            platform = state.history[-1].observation.platform_health

        base = _HINTS.get(platform, _HINTS[PlatformHealth.NOMINAL])

        # Enrich with live episode stats
        epsilon_pct = round(state.epsilon_remaining / max(state.epsilon_initial, 1e-9) * 100, 1)
        budget_pct  = round(state.budget_remaining  / max(state.budget_initial,   1e-9) * 100, 1)

        return {
            **base,
            "current_day":        state.step,
            "epsilon_remaining":  round(state.epsilon_remaining, 3),
            "epsilon_pct":        epsilon_pct,
            "budget_remaining":   round(state.budget_remaining, 2),
            "budget_pct":         budget_pct,
            "learning_resets":    state.learning_resets,
            "overspend_events":   state.overspend_events,
            "capi_calls_used":    state.capi_calls_used,
        }


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

@app.post("/baseline", response_model=BaselineResult, tags=["evaluation"])
def baseline() -> BaselineResult:
    """
    Run the baseline agent across all 7 tasks (seed=42).
    Returns scores and per-task GraderResult breakdowns.

    Requires HF_TOKEN environment variable (or API_BASE_URL + MODEL_NAME).
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
