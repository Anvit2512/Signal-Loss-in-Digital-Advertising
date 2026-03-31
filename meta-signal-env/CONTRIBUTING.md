# Contributing to Meta-Signal

This guide explains how to add a new task to the environment. The architecture
is deliberately layered so that a new task requires touching exactly four places —
nothing else.

---

## Architecture overview

```
models.py        Pure data shapes. No logic.
privacy.py       Epsilon budget + Laplace noise. Task-agnostic.
data_loader.py   Criteo snapshot loader. Task-agnostic.
tasks.py         Task configs + grader functions. This is where new tasks live.
env.py           Wires everything together. Minimal task-specific code.
main.py          FastAPI endpoints. Usually no changes needed for a new task.
```

The key insight: **a task is just a config dict + a grader function**. Everything
else (noise, budget, observation format, API shape) is shared infrastructure.

---

## Step 1 — Define the task config in `tasks.py`

Add an entry to `TASK_CONFIGS`:

```python
TASK_CONFIGS: Dict[int, TaskDefinition] = {
    # ... existing tasks 1–4 ...
    5: TaskDefinition(
        task_id=5,
        name="My New Task",
        description=(
            "What the agent needs to do and why it is hard. "
            "This string is returned verbatim by GET /tasks and shown in the web UI."
        ),
        max_steps=12,
        initial_budget=1200.0,
        initial_epsilon=2.5,
        privacy_regime=PrivacyRegime.STANDARD,   # or HIGH_NOISE / MINIMAL_DATA
        target_roas=1.2,
        max_features=3,
        grader_weights={
            "component_a": 0.5,
            "component_b": 0.3,
            "component_c": 0.2,   # weights must sum to 1.0
        },
    ),
}
```

**`PrivacyRegime` options**

| Value | Effect |
|---|---|
| `STANDARD` | Normal Laplace noise. Good starting point. |
| `HIGH_NOISE` | Elevated noise from step 1. |
| `MINIMAL_DATA` | Forced 1-feature-max regulatory constraint. |

---

## Step 2 — Write the grader function in `tasks.py`

```python
def grade_task5(
    state: EpisodeState,
    alloc_history: List[Dict[str, float]],
    # add privacy_engine: "PrivacyEngine" if you need compliance_rate()
) -> GraderResult:
    """
    Task 5 — My New Task

    50% component_a: ...
    30% component_b: ...
    20% component_c: ...
    """
    cfg = TASK_CONFIGS[5]

    # Use the shared helpers from the top of this file:
    avg_roas  = _avg_step_roas(state)
    roas_s    = _roas_score(avg_roas, cfg.target_roas)

    # Write your own scoring logic for other components:
    component_b_s = ...
    component_c_s = ...

    score = roas_s * 0.5 + component_b_s * 0.3 + component_c_s * 0.2

    # Write a one-sentence verdict for non-technical judges:
    explanation = (
        f"Agent {'exceeded' if avg_roas >= cfg.target_roas else 'missed'} "
        f"the {cfg.target_roas}x ROAS target (avg {avg_roas:.2f}x); "
        f"component_b={component_b_s:.2f}, component_c={component_c_s:.2f}."
    )

    return GraderResult(
        task_id=5,
        score=round(min(1.0, max(0.0, score)), 4),
        breakdown={
            "component_a": round(roas_s, 4),
            "component_b": round(component_b_s, 4),
            "component_c": round(component_c_s, 4),
        },
        summary={
            "avg_roas":        round(avg_roas, 4),
            "target_roas":     cfg.target_roas,
            "violations":      float(state.regulatory_violations),
            "epsilon_used":    round(state.epsilon_initial - state.epsilon_remaining, 4),
            "steps_completed": float(state.step),
        },
        explanation=explanation,
    )
```

**Shared helper functions** (already in `tasks.py`, import-free):

| Helper | Returns |
|---|---|
| `_avg_step_roas(state)` | Mean step ROAS across all completed steps |
| `_avg_oracle_roas(state)` | Mean oracle ROAS (best possible each step) |
| `_roas_score(avg_roas, target)` | Continuous score in [0, 1], capped at 1.0 |
| `_allocation_trend_score(alloc_history, camp)` | 3-phase explore/learn/exploit arc score |
| `privacy_engine.compliance_rate(max_features)` | Fraction of steps within feature limit |

---

## Step 3 — Wire the grader into `env.py`

In `compute_final_score()`, add one branch:

```python
def compute_final_score(self) -> GraderResult:
    ...
    elif task_id == 5:
        result = grade_task5(self._state, self._alloc_history)
    ...
```

Also update the import at the top of `env.py`:

```python
from app.tasks import (
    AUDIT_STEP,
    get_task_config,
    grade_task1,
    grade_task2,
    grade_task3,
    grade_task4,
    grade_task5,   # add this
)
```

If your task needs a mid-episode event (like Task 2's noise jump or Task 4's
audit), add it in `env.step()` in the block labelled `# Task N: ...`. Keep
each task's logic self-contained and guarded by `if self._state.task_id == N`.

---

## Step 4 — Update the `ResetRequest` validator in `models.py`

Increase the upper bound so the new task_id is accepted:

```python
class ResetRequest(BaseModel):
    task_id: int = Field(ge=1, le=5)   # was le=4
```

Same for `GraderRequest`:

```python
class GraderRequest(BaseModel):
    task_id: int = Field(ge=1, le=5)
```

---

## Step 5 — Add tests in `tests/test_server.py`

At minimum, add:

```python
def test_reset_task5_succeeds():
    r = client.post("/reset", json={"task_id": 5, "seed": 42})
    assert r.status_code == 200

def test_task5_full_episode_score_in_range():
    from app.tasks import TASK_CONFIGS
    cfg = TASK_CONFIGS[5]
    client.post("/reset", json={"task_id": 5, "seed": 42})
    for _ in range(cfg.max_steps):
        result = client.post("/step", json={
            "allocations": {"camp_feed": 20.0, "camp_reels": 10.0, "camp_stories": 10.0},
            "attribution": "last_click",
            "feature_mask": ["I1"],
        }).json()
        if result["done"]:
            break
    grade = client.post("/grader", json={"task_id": 5}).json()
    assert 0.0 <= grade["score"] <= 1.0
    assert "explanation" in grade

def test_task5_grader_breakdown_keys():
    # ... run episode, then:
    grade = client.post("/grader", json={"task_id": 5}).json()
    for key in ("component_a", "component_b", "component_c"):
        assert key in grade["breakdown"]
```

Run the suite before opening a PR:

```bash
pytest tests/ -v
```

All tests must be green.

---

## Design principles to follow

**Keep tasks self-contained.** Each task's grader reads only `EpisodeState`,
`alloc_history`, and optionally `PrivacyEngine`. It does not call other graders
or mutate shared state.

**Continuous scores only.** All grader components return floats in [0, 1].
No binary pass/fail — partial credit is what makes the leaderboard meaningful.

**Write the explanation.** Every `GraderResult` must include an `explanation`
string. One sentence, plain English, names what the agent did well and where it
lost points. Non-technical judges read this; make it useful.

**Stay below the 70% correlation threshold.** If your task involves budget
allocation, the `correlation_penalty` in `env.step()` fires automatically when
any single campaign exceeds 70% of spend. You do not need to implement this
yourself — it is environment-level behaviour shared across all tasks.

**Use the existing privacy regimes.** The `PrivacyEngine` supports four regimes
out of the box. If you need a custom noise profile, extend `privacy.py` — do not
fork it inside a task.

---

## Running the environment locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# Run tests
pytest tests/ -v

# Smoke-test all tasks via the env directly
python -m app.env
```

The web UI at `http://localhost:7860` includes a **⚡ SIMULATE** button that
runs any task with a built-in strategy (equal / greedy / conservative) and
shows a full step trace — useful for sanity-checking a new task without writing
agent code.
