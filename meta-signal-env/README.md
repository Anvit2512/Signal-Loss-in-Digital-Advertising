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

## Why This Matters

On October 26, 2022, Meta reported its third-quarter earnings. Revenue had fallen
year-over-year for the second consecutive quarter — the first time that had happened
in the company's history as a public company. The stock dropped 24% in after-hours
trading. By the following morning, **$232 billion in market capitalisation had been
erased in a single session** — the largest single-day destruction of market value for
any company in US stock market history.

Zuckerberg named two causes on the earnings call. One was the metaverse investment.
The other was **signal loss**.

Eighteen months earlier, Apple had shipped the **App Tracking Transparency (ATT)
prompt** in iOS 14.5. The mechanic was simple: before any app could track a user
across other apps and websites, it had to ask. Roughly 80% of users said no.
Overnight, the pixel-level, deterministic, user-level conversion signals that Meta's
ad auction had been trained on for a decade were replaced by something far noisier:
aggregated counts, delayed postbacks, and Apple's own **SKAdNetwork** attribution
— a single coarse-grained conversion value per install, delivered up to 72 hours
late, with no user identity attached.

Meta's ad system had been built on a feedback loop so tight that a campaign manager
could see a conversion attributed to a specific ad impression within minutes.
ATT didn't slow that loop — it broke it. Advertisers who had been spending confidently
on iOS performance campaigns suddenly couldn't tell which placements were working.
Budgets rotated to Android. CPMs on iOS dropped because demand collapsed. Meta's
targeting models, starved of signal, began recommending the wrong audiences. The
auction cleared at lower prices. Revenue fell.

The technical response was **Aggregated Event Measurement (AEM)** — a
privacy-preserving measurement API that adds calibrated Laplace noise to reported
conversion counts and caps the number of trackable event types per campaign.
It preserved some signal. But it introduced a new constraint that the ad system
had never had to reason about before: **the quality of the signal degrades as you
query it more.** More measurement events consumed meant more noise applied to each.
Budget allocation decisions that had previously been made on clean, dense data now
had to be made on a finite and depletable information budget.

That is precisely the problem this environment models.

An agent in Meta-Signal manages spend across Feed, Reels, and Stories campaigns
using only **differentially private, aggregated conversion signals**. It has a finite
epsilon budget: the more aggressively it queries signal, the noisier future
observations become. It faces mid-episode market shifts — a viral trend that doubles
Reels CVR — that it can only detect if it preserved enough budget to see through the
noise. It faces an auction-overlap correlation penalty when it concentrates spend too
heavily, because in a real multi-placement auction, bidding aggressively on Feed
wins the same user twice and suppresses Reels performance. And it faces regulatory
audits that can suspend a campaign without warning, forcing real-time reallocation
under incomplete information.

These are not toy constraints invented for a competition. They are the specific
mechanics that caused a $232 billion loss and that Meta's engineering teams have
spent three years building **Advantage+** to address.

A trained RL agent on this environment would be directly applicable to Meta's
Advantage+ signal recovery pipeline.

---

## Motivation

Meta's $160B ad business was built on pixel-level conversion signals. Then Apple
shipped the **App Tracking Transparency (ATT) prompt** in iOS 14.5, and overnight
roughly 80% of users opted out of cross-app tracking. Campaigns that previously had
deterministic, event-level attribution were suddenly flying on aggregated,
statistically-noised data.

Meta's response was **Aggregated Event Measurement (AEM)** — a privacy-preserving
measurement API that caps the number of reportable conversion event types, delays
postbacks by up to 72 hours, and adds calibrated noise to prevent fingerprinting.
On the iOS side, Apple's **SKAdNetwork** imposes its own postback structure: a single
coarse-grained conversion value per install, delivered days late, with no user-level
signal at all.

**Advantage+ Shopping Campaigns (ASC)**, Meta's AI-driven campaign automation, was
built partly in response to this: if you can't trust fine-grained signals, lean into
the ML model. But even Advantage+ can't escape the fundamental constraint — a budget
allocation decision must still be made across placements (Feed, Reels, Stories), and
the noisy aggregate signal is all the system has to reason from.

Meta-Signal models exactly this tension. An RL agent manages spend across Feed, Reels,
and Stories campaigns using only **differential-privacy-protected aggregated conversion
signals**, with a finite epsilon budget that degrades signal quality as it depletes.
The correlation penalty (auction-overlap dynamics) and mid-episode market shifts are
the kind of second-order effects that Advantage+ has to handle internally — and that
no existing OpenEnv environment exposes to an agent in a testable, reproducible form.

No existing OpenEnv submission models privacy-constrained ad optimisation with
real auction-overlap mechanics and dual-event mid-episode shifts.

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
| POST | `/simulate` | Run a full episode with a built-in strategy, no code required |
| POST | `/baseline` | Run LLM baseline internally |
| GET | `/docs` | Swagger UI |

### /simulate — no-code exploration

```json
POST /simulate
{
  "task_id": 2,
  "strategy": "greedy",
  "seed": 42
}
```

Returns a score, grader breakdown, and a full step-by-step trace. Strategy options:

| Strategy | Policy |
|---|---|
| `equal` | Even three-way split every step |
| `greedy` | 80% to whichever campaign had the best noisy signal last step |
| `conservative` | Fixed 60/25/15 split, stays below 70% to avoid the auction-overlap correlation penalty |

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
