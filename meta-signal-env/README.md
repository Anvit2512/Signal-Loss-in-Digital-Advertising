---
title: Meta-Signal
emoji: 📡
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - advertising
  - reinforcement-learning
  - differential-privacy
---

# Meta-Signal: Privacy-Constrained Ad Budget Optimisation

An RL environment where an AI agent manages advertising budget across three campaigns
but can only observe **noisy, aggregated conversion data** — exactly how Meta's real
ad system works after signal loss from iOS privacy changes.

## Motivation

Meta's $160B ad business runs on conversion signals that have become increasingly
blurry since ATT. This environment models that exact tension: an agent must allocate
budget intelligently using only differential-privacy-protected aggregated signals,
with a finite epsilon budget that degrades signal quality as it depletes.

No existing OpenEnv environment models privacy-constrained ad optimisation.

---

## Environment Description

### Campaigns

| Campaign | Placement | True CVR | Difficulty to find |
|---|---|---|---|
| `camp_feed` | Feed | ~8.5% | Easy with clean signal |
| `camp_reels` | Reels | ~3.6% | Medium |
| `camp_stories` | Stories | ~2.0% | Easy to misidentify as best |

### Privacy Mechanic

The agent has a finite **epsilon budget** (differential privacy). Each step:
- Every feature used costs **0.05 epsilon**
- Probabilistic attribution costs **+0.20 epsilon** extra
- Laplace noise scale = `1 / epsilon_remaining` — grows as budget depletes
- When epsilon hits 0, signal is essentially random

### Privacy Regimes

| Regime | Epsilon | Noise Scale | Signal Quality |
|---|---|---|---|
| `standard` | > 0.5 | Low | Readable |
| `high_noise` | 0.1 – 0.5 | High | Degraded |
| `minimal_data` | Any | Task 3 forced | 1 feature max |
| `exhausted` | < 0.1 | ~50x | Near random |

---

## Action Space

```json
{
  "allocations": {
    "camp_feed":    200.0,
    "camp_reels":   100.0,
    "camp_stories": 100.0
  },
  "attribution": "last_click",
  "feature_mask": ["I1", "I2"]
}
```

| Field | Type | Description |
|---|---|---|
| `allocations` | `Dict[str, float]` | Dollar spend per campaign. Must be >= 0, sum <= budget |
| `attribution` | `str` | `last_click` (free) or `probabilistic` (+0.20 epsilon) |
| `feature_mask` | `List[str]` | Features to use from I1-I13, C1-C26. Each costs 0.05 epsilon |

---

## Observation Space

```json
{
  "step": 3,
  "campaigns": [
    {
      "campaign_id": "camp_feed",
      "placement": "feed",
      "impressions": 35,
      "spend": 200.0,
      "noisy_conversions": 3.2,
      "estimated_roas": 2.29,
      "ctr": 0.0857
    }
  ],
  "total_budget_remaining": 600.0,
  "epsilon_remaining": 2.65,
  "privacy_regime": "standard",
  "available_features": ["I1", "I2", "I3"],
  "regulatory_violation": false,
  "warning": null
}
```

---

## Tasks

### Task 1 — Budget Optimisation (Easy)
- **Steps:** 10 | **Epsilon:** 3.0 | **Max features:** 5
- **Goal:** Learn within 10 steps that `camp_feed` has the highest ROAS and
  progressively shift budget toward it.
- **Grader:** 60% ROAS score + 40% allocation trend toward `camp_feed`
- **Expected baseline score:** ~0.43

### Task 2 — Noisy Signal Recovery (Medium)
- **Steps:** 15 | **Epsilon:** 3.0 | **Max features:** 3
- **Goal:** At step 3 a privacy update fires and noise jumps dramatically.
  Use early observations to infer the best campaign, then maintain ROAS
  using only degraded signals for the remaining 12 steps.
- **Grader:** 50% oracle proximity + 30% budget efficiency + 20% clean run
- **Expected baseline score:** ~0.54

### Task 3 — Privacy Frontier (Hard)
- **Steps:** 15 | **Epsilon:** 2.0 | **Max features:** 1
- **Goal:** Regulatory data minimisation from the start. Only 1 feature per step.
  Maintain ROAS above 1.0 while staying compliant. Violations penalised quadratically.
- **Grader:** 40% ROAS + 40% compliance rate + 20% epsilon remaining
- **Expected baseline score:** ~0.72

### Task 4 — The Adversarial Regulator (Bonus)
- **Steps:** 20 | **Epsilon:** 3.0 | **Budget:** $1500 | **Max features:** 3
- **Goal:** At step 5 a regulatory audit fires and one campaign is immediately suspended.
  The agent must: (1) set the flagged campaign's allocation to zero, (2) submit a valid
  `legal_reason_code` (`GDPR_ART17`, `DPA_NOTICE`, or `REGULATORY_HOLD`), and (3) recover
  ROAS above 1.0 using only the two remaining campaigns.
- **Grader:** 30% ROAS recovery + 40% audit compliance + 30% legal code quality
- **New in all tasks:** `confidence_interval` field on every `CampaignStats` — 95% CI
  around `noisy_conversions`, narrows with high epsilon, widens as budget depletes.

---

## Reward Function

```
reward = (step_roas × 0.7) - (regulatory_penalty × 2.0) + (epsilon_fraction × 0.1)
```

- **Revenue signal:** Are your allocations making money?
- **Compliance signal:** Penalised every step you exceed the feature limit (quadratic)
- **Efficiency signal:** Small bonus for preserving privacy budget for later steps

---

## Baseline Scores

Reproducible scores from `python inference.py` with `seed=42`:

| Task | Score | Model | API |
|---|---|---|---|
| Task 1 — Budget Optimisation | 0.4252 | llama-3.3-70b-versatile | Groq |
| Task 2 — Noisy Signal Recovery | 0.5402 | llama-3.3-70b-versatile | Groq |
| Task 3 — Privacy Frontier | 0.7233 | llama-3.3-70b-versatile | Groq |

These gaps are intentional — a trained RL agent should score significantly higher,
especially on Task 2 (signal recovery after noise jump) and Task 3 (compliance).

---

## Setup

### Local (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t meta-signal .
docker run -p 7860:7860 -e HF_TOKEN=your_key meta-signal
```

### Run inference script

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=your_hf_token

python inference.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/tasks` | All task definitions |
| POST | `/reset` | Start new episode `{"task_id": 1, "seed": 42}` |
| POST | `/step` | Submit action, get observation |
| GET | `/state` | Current episode state |
| POST | `/grader` | Compute final score `{"task_id": 1}` |
| POST | `/baseline` | Run LLM baseline internally |
| GET | `/docs` | Swagger UI |

---

## Project Structure

```
meta-signal-env/
├── data/
│   ├── ad_logs_sampled.csv        Frozen Criteo-schema snapshot (10k rows)
│   └── generate_snapshot.py       Regenerate snapshot
├── app/
│   ├── __init__.py
│   ├── data_loader.py             Criteo loader + campaign partitioner
│   ├── models.py                  Pydantic types
│   ├── privacy.py                 Epsilon budget + Laplace noise engine
│   ├── tasks.py                   Task definitions + graders
│   ├── env.py                     Core environment logic
│   └── main.py                    FastAPI server
├── tests/
│   └── test_server.py             28 end-to-end tests
├── inference.py                   Competition inference script
├── baseline.py                    Extended baseline utility
├── openenv.yaml                   Competition manifest
├── Dockerfile
└── requirements.txt
```

---

## Tags

`openenv` `advertising` `differential-privacy` `reinforcement-learning` `budget-optimisation`
