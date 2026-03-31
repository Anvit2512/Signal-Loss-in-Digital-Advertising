"""
End-to-end tests for the Meta-Signal FastAPI server.

Uses FastAPI's TestClient -- no separate server process needed.
Run with: pytest tests/test_server.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

def test_tasks_returns_three():
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()
    assert len(tasks) == 4

def test_tasks_ids_are_1_2_3():
    tasks = client.get("/tasks").json()
    ids = {t["task_id"] for t in tasks}
    assert ids == {1, 2, 3, 4}

def test_tasks_have_required_fields():
    for task in client.get("/tasks").json():
        assert "name" in task
        assert "max_steps" in task
        assert "target_roas" in task
        assert "grader_weights" in task
        assert "initial_epsilon" in task


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

def test_reset_task1():
    r = client.post("/reset", json={"task_id": 1, "seed": 42})
    assert r.status_code == 200
    obs = r.json()
    assert obs["step"] == 0
    assert obs["total_budget_remaining"] == 1000.0
    assert obs["epsilon_remaining"] == 3.0
    assert obs["privacy_regime"] == "standard"
    assert len(obs["campaigns"]) == 3

def test_reset_task3_regime_is_minimal_data():
    r = client.post("/reset", json={"task_id": 3, "seed": 42})
    assert r.status_code == 200
    assert r.json()["privacy_regime"] == "minimal_data"

def test_reset_invalid_task_id():
    r = client.post("/reset", json={"task_id": 99})
    assert r.status_code == 422   # Pydantic validation: ge=1 le=4

def test_reset_reproducible_with_seed():
    obs_a = client.post("/reset", json={"task_id": 1, "seed": 7}).json()
    obs_b = client.post("/reset", json={"task_id": 1, "seed": 7}).json()
    assert obs_a["epsilon_remaining"] == obs_b["epsilon_remaining"]
    assert obs_a["total_budget_remaining"] == obs_b["total_budget_remaining"]


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------

VALID_STEP = {
    "allocations": {"camp_feed": 100.0, "camp_reels": 50.0, "camp_stories": 50.0},
    "attribution": "last_click",
    "feature_mask": ["I1"],
}

def test_step_returns_correct_shape():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    r = client.post("/step", json=VALID_STEP)
    assert r.status_code == 200
    result = r.json()
    assert "observation" in result
    assert "reward" in result
    assert "done" in result
    assert "info" in result

def test_step_reduces_budget():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    r = client.post("/step", json=VALID_STEP)
    obs = r.json()["observation"]
    assert obs["total_budget_remaining"] < 1000.0

def test_step_reduces_epsilon():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    r = client.post("/step", json=VALID_STEP)
    obs = r.json()["observation"]
    assert obs["epsilon_remaining"] < 3.0

def test_step_without_reset_raises():
    # Force a fresh env by importing directly and calling state before reset
    from app.main import _env
    _env._state = None
    r = client.post("/step", json=VALID_STEP)
    assert r.status_code == 400

def test_step_negative_allocation_rejected():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    bad_action = {
        "allocations": {"camp_feed": -100.0, "camp_reels": 50.0, "camp_stories": 50.0},
        "attribution": "last_click",
        "feature_mask": ["I1"],
    }
    r = client.post("/step", json=bad_action)
    assert r.status_code == 422

def test_step_invalid_feature_raises():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    bad = dict(VALID_STEP)
    bad["feature_mask"] = ["FAKE_FEATURE"]
    r = client.post("/step", json=bad)
    assert r.status_code == 400

def test_step_info_has_oracle_roas():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    result = client.post("/step", json=VALID_STEP).json()
    assert "oracle_roas" in result["info"]
    assert result["info"]["oracle_roas"] >= 0.0

def test_step_regulatory_violation_flagged():
    # Task 3 allows only 1 feature -- send 3
    client.post("/reset", json={"task_id": 3, "seed": 42})
    action = {
        "allocations": {"camp_feed": 100.0, "camp_reels": 50.0, "camp_stories": 50.0},
        "attribution": "last_click",
        "feature_mask": ["I1", "I2", "I3"],   # 3 > max_features=1
    }
    result = client.post("/step", json=action).json()
    assert result["observation"]["regulatory_violation"] is True


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------

def test_state_after_reset():
    client.post("/reset", json={"task_id": 2, "seed": 10})
    r = client.get("/state")
    assert r.status_code == 200
    state = r.json()
    assert state["task_id"] == 2
    assert state["step"] == 0
    assert state["total_steps"] == 15

def test_state_step_increments():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    client.post("/step", json=VALID_STEP)
    client.post("/step", json=VALID_STEP)
    state = client.get("/state").json()
    assert state["step"] == 2

def test_state_history_grows():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    for _ in range(3):
        client.post("/step", json=VALID_STEP)
    state = client.get("/state").json()
    assert len(state["history"]) == 3


# ---------------------------------------------------------------------------
# /grader
# ---------------------------------------------------------------------------

def test_grader_returns_score_in_range():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    for _ in range(10):
        client.post("/step", json=VALID_STEP)
    r = client.post("/grader", json={"task_id": 1})
    assert r.status_code == 200
    result = r.json()
    assert 0.0 <= result["score"] <= 1.0

def test_grader_returns_breakdown():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    for _ in range(10):
        client.post("/step", json=VALID_STEP)
    result = client.post("/grader", json={"task_id": 1}).json()
    assert "roas_score" in result["breakdown"]
    assert "allocation_trend" in result["breakdown"]

def test_grader_wrong_task_id_rejected():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    r = client.post("/grader", json={"task_id": 2})
    assert r.status_code == 400

def test_grader_task2_breakdown_keys():
    client.post("/reset", json={"task_id": 2, "seed": 42})
    for _ in range(15):
        client.post("/step", json=VALID_STEP)
    result = client.post("/grader", json={"task_id": 2}).json()
    bd = result["breakdown"]
    assert "oracle_proximity" in bd
    assert "budget_efficiency" in bd
    assert "clean_run" in bd

def test_grader_task3_breakdown_keys():
    client.post("/reset", json={"task_id": 3, "seed": 42})
    action = {
        "allocations": {"camp_feed": 20.0, "camp_reels": 10.0, "camp_stories": 10.0},
        "attribution": "last_click",
        "feature_mask": ["I1"],
    }
    for _ in range(15):
        client.post("/step", json=action)
    result = client.post("/grader", json={"task_id": 3}).json()
    bd = result["breakdown"]
    assert "roas_score" in bd
    assert "compliance_score" in bd
    assert "epsilon_remaining" in bd


# ---------------------------------------------------------------------------
# Full episode flow (all three tasks)
# ---------------------------------------------------------------------------

def _run_full_episode(task_id: int, feature_mask: list) -> dict:
    from app.tasks import TASK_CONFIGS
    cfg = TASK_CONFIGS[task_id]
    client.post("/reset", json={"task_id": task_id, "seed": 42})
    for _ in range(cfg.max_steps):
        action = {
            "allocations": {"camp_feed": 20.0, "camp_reels": 10.0, "camp_stories": 10.0},
            "attribution": "last_click",
            "feature_mask": feature_mask,
        }
        result = client.post("/step", json=action).json()
        if result["done"]:
            break
    return client.post("/grader", json={"task_id": task_id}).json()

def test_full_episode_task1():
    grade = _run_full_episode(1, ["I1", "I2"])
    assert 0.0 <= grade["score"] <= 1.0

def test_full_episode_task2_high_noise_fires():
    """After step 3, regime should be high_noise or exhausted."""
    client.post("/reset", json={"task_id": 2, "seed": 42})
    for _ in range(3):
        client.post("/step", json=VALID_STEP)
    state = client.get("/state").json()
    assert state["privacy_regime"] in ("high_noise", "exhausted")

def test_full_episode_task3_compliance():
    """With 1 feature per step, compliance_score should be 1.0."""
    grade = _run_full_episode(3, ["I1"])
    assert grade["breakdown"]["compliance_score"] == 1.0

def test_task3_compliance_penalised_with_excess_features():
    """Using 3 features on Task 3 (max=1) should lower compliance score."""
    grade = _run_full_episode(3, ["I1", "I2", "I3"])
    assert grade["breakdown"]["compliance_score"] < 1.0


# ---------------------------------------------------------------------------
# Task 4 -- The Adversarial Regulator
# ---------------------------------------------------------------------------

def test_reset_task4_succeeds():
    r = client.post("/reset", json={"task_id": 4, "seed": 42})
    assert r.status_code == 200
    obs = r.json()
    assert obs["step"] == 0
    assert obs["audit_active"] is False
    assert obs["flagged_campaign"] is None


def test_task4_confidence_interval_present():
    """CampaignStats must include confidence_interval after a step."""
    client.post("/reset", json={"task_id": 4, "seed": 42})
    r = client.post("/step", json=VALID_STEP)
    assert r.status_code == 200
    camps = r.json()["observation"]["campaigns"]
    for cs in camps:
        assert "confidence_interval" in cs
        ci = cs["confidence_interval"]
        assert len(ci) == 2
        assert ci[1] >= ci[0]   # upper >= lower


def test_task4_audit_fires_at_step5():
    """After 5 steps, audit_active should be True and flagged_campaign set."""
    client.post("/reset", json={"task_id": 4, "seed": 42})
    obs = None
    for _ in range(5):
        r = client.post("/step", json=VALID_STEP)
        obs = r.json()["observation"]
    assert obs["audit_active"] is True
    assert obs["flagged_campaign"] in ("camp_feed", "camp_reels", "camp_stories")


def test_task4_full_episode_score_in_range():
    """Full Task 4 episode with compliant agent returns valid score."""
    from app.tasks import TASK_CONFIGS
    cfg = TASK_CONFIGS[4]
    client.post("/reset", json={"task_id": 4, "seed": 42})
    flagged = None
    for step_n in range(cfg.max_steps):
        action = {
            "allocations": {
                "camp_feed":    0.0 if flagged == "camp_feed"    else 20.0,
                "camp_reels":   0.0 if flagged == "camp_reels"   else 10.0,
                "camp_stories": 0.0 if flagged == "camp_stories" else 10.0,
            },
            "attribution": "last_click",
            "feature_mask": ["I1"],
            "halted_campaigns": [flagged] if flagged else [],
            "legal_reason_code": "GDPR_ART17" if flagged else None,
        }
        result = client.post("/step", json=action).json()
        obs = result["observation"]
        if obs.get("audit_active") and obs.get("flagged_campaign"):
            flagged = obs["flagged_campaign"]
        if result["done"]:
            break
    grade = client.post("/grader", json={"task_id": 4}).json()
    assert 0.0 <= grade["score"] <= 1.0
    assert "roas_recovery" in grade["breakdown"]
    assert "audit_compliance" in grade["breakdown"]
    assert "legal_code_quality" in grade["breakdown"]
