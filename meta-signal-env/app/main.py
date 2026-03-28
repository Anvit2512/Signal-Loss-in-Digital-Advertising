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
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.env import MetaSignalEnv
from app.models import (
    Action,
    BaselineResult,
    EpisodeState,
    GraderRequest,
    GraderResult,
    Observation,
    ResetRequest,
    StepResult,
    TaskDefinition,
)
from app.tasks import TASK_CONFIGS

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


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["system"])
def web() -> str:
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Meta-Signal Environment</title>
  <style>
    body { font-family: monospace; max-width: 720px; margin: 40px auto; padding: 0 20px; }
    h1   { border-bottom: 2px solid #333; padding-bottom: 8px; }
    a    { color: #0066cc; }
    .tag { background: #eee; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    td, th { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
    th { background: #f5f5f5; }
  </style>
</head>
<body>
  <h1>Meta-Signal</h1>
  <p>Privacy-constrained ad budget optimisation environment.<br>
     An agent allocates budget across three campaigns using only noisy,
     aggregated conversion signals — exactly how Meta's real ad system
     works after signal loss.</p>

  <h2>Endpoints</h2>
  <table>
    <tr><th>Method</th><th>Path</th><th>Purpose</th></tr>
    <tr><td>GET</td> <td><a href="/tasks">/tasks</a></td>    <td>All task definitions</td></tr>
    <tr><td>POST</td><td>/reset</td>   <td>Start new episode</td></tr>
    <tr><td>POST</td><td>/step</td>    <td>Submit action, get observation</td></tr>
    <tr><td>GET</td> <td><a href="/state">/state</a></td>    <td>Current episode state</td></tr>
    <tr><td>POST</td><td>/grader</td>  <td>Compute final score (0.0-1.0)</td></tr>
    <tr><td>POST</td><td>/baseline</td><td>Run GPT-4o-mini baseline</td></tr>
    <tr><td>GET</td> <td><a href="/docs">/docs</a></td>      <td>Interactive API docs</td></tr>
    <tr><td>GET</td> <td><a href="/health">/health</a></td>  <td>Liveness probe</td></tr>
  </table>

  <h2>Tasks</h2>
  <table>
    <tr><th>ID</th><th>Name</th><th>Steps</th><th>Target ROAS</th><th>Max Features</th></tr>
    <tr><td>1</td><td>Budget Optimisation</td>   <td>10</td><td>1.5</td><td>5</td></tr>
    <tr><td>2</td><td>Noisy Signal Recovery</td> <td>15</td><td>1.5</td><td>3</td></tr>
    <tr><td>3</td><td>Privacy Frontier</td>      <td>15</td><td>1.0</td><td>1</td></tr>
  </table>

  <p><a href="/docs">Full API documentation (Swagger UI)</a></p>
</body>
</html>
"""
