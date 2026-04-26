"""
Run: python plot_results.py
Saves: results/meta-signal-results.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("results", exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

EXPERT_ALL = {
    "T1\nBudget\nOptimisation":   0.43,
    "T2\nNoisy Signal\nRecovery": 0.54,
    "T3\nPrivacy\nFrontier":      0.72,
    "T4\nAdversarial\nRegulator": 0.60,
    "T5\nSignal\nRecovery":       0.800,
    "T6\nAndromeda\nStability":   0.864,
    "T7\nQ4\nChampion":           0.850,
}

# Fine-tuned: 3 seeds per task
FT_SEEDS = {
    "T5": [0.800, 0.800, 0.800],
    "T6": [0.864, 0.864, 0.864],
    "T7": [0.850, 0.850, 0.850],
}
EXPERT_Q4 = {"T5": 0.800, "T6": 0.864, "T7": 0.850}
FT_AVG    = {k: np.mean(v) for k, v in FT_SEEDS.items()}

TASK_LABELS = {
    "T5": "Task 5\nSignal Recovery\n(30 steps)",
    "T6": "Task 6\nAndromeda\nStability (75 steps)",
    "T7": "Task 7\nQ4 Champion\n(100 steps)",
}

# ── Colors ────────────────────────────────────────────────────────────────────
C_EXPERT = "#4A90D9"
C_FT     = "#E8874A"
C_SEED   = "#555555"
BG       = "#F7F9FC"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.subplots_adjust(wspace=0.35)

# ── Panel 1: ExpertBot baseline — all 7 tasks ─────────────────────────────────
labels1 = list(EXPERT_ALL.keys())
scores1 = list(EXPERT_ALL.values())
colors1 = [C_EXPERT if i < 4 else "#6C5CE7" for i in range(7)]

bars1 = ax1.bar(labels1, scores1, color=colors1, width=0.55,
                edgecolor="white", linewidth=1.2, zorder=3)

for bar, score in zip(bars1, scores1):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.015,
             f"{score:.2f}",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold",
             color="#333333")

ax1.axvline(x=3.5, color="#999999", linestyle="--", linewidth=1.2, zorder=2)
ax1.text(1.5, 0.92, "Core Tasks", ha="center", fontsize=9,
         color=C_EXPERT, fontweight="semibold")
ax1.text(5.0, 0.92, "Q4 Gauntlet", ha="center", fontsize=9,
         color="#6C5CE7", fontweight="semibold")

ax1.set_ylim(0, 1.0)
ax1.set_ylabel("Score (0 – 1)", fontsize=11)
ax1.set_title("ExpertBot Baseline — All 7 Tasks", fontsize=13, fontweight="bold",
              pad=14)
ax1.set_facecolor(BG)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(axis="x", labelsize=8)

# ── Panel 2: ExpertBot vs Fine-tuned — Q4 tasks ───────────────────────────────
tasks  = list(TASK_LABELS.keys())
x      = np.arange(len(tasks))
width  = 0.32

bars_e = ax2.bar(x - width / 2,
                 [EXPERT_Q4[t] for t in tasks],
                 width, label="ExpertBot baseline",
                 color=C_EXPERT, edgecolor="white", linewidth=1.2, zorder=3)

bars_f = ax2.bar(x + width / 2,
                 [FT_AVG[t] for t in tasks],
                 width, label="Pending fine-tuned re-run placeholder",
                 color=C_FT, edgecolor="white", linewidth=1.2, zorder=3)

# Seed dots on fine-tuned bars
for i, t in enumerate(tasks):
    for seed_score in FT_SEEDS[t]:
        ax2.scatter(x[i] + width / 2, seed_score,
                    color=C_SEED, s=28, zorder=5, alpha=0.85)

# Score labels
for bar in bars_e:
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.012,
             f"{bar.get_height():.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold",
             color=C_EXPERT)

for bar in bars_f:
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.012,
             f"{bar.get_height():.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold",
             color=C_FT)

# Delta annotations
for i, t in enumerate(tasks):
    delta = FT_AVG[t] - EXPERT_Q4[t]
    sign  = "+" if delta >= 0 else ""
    color = "#27AE60" if delta >= 0 else "#E74C3C"
    ax2.text(x[i], max(EXPERT_Q4[t], FT_AVG[t]) + 0.055,
             f"{sign}{delta:.3f}",
             ha="center", fontsize=9, color=color, fontweight="bold")

ax2.set_ylim(0, 1.0)
ax2.set_ylabel("Score (0 – 1)", fontsize=11)
ax2.set_title("Q4 Gauntlet ExpertBot Baselines\nFine-tuned re-run pending",
              fontsize=13, fontweight="bold", pad=14)
ax2.set_xticks(x)
ax2.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=8.5)
ax2.set_facecolor(BG)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

seed_dot = mpatches.Patch(color=C_SEED, label="Individual seed scores")
ax2.legend(handles=[
    mpatches.Patch(color=C_EXPERT, label="ExpertBot baseline"),
    mpatches.Patch(color=C_FT,     label="Pending re-run placeholder"),
    seed_dot,
], fontsize=8.5, loc="lower right", framealpha=0.85)

# ── Shared caption ─────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "ExpertBot baselines refreshed after Q4 simultaneous-placement fix  |  "
    "Re-run fine-tuned evaluation after regenerating demonstrations from this environment",
    ha="center", fontsize=8.5, color="#555555",
)

fig.suptitle("Meta-Signal: Privacy-Constrained Ad Budget Optimisation",
             fontsize=15, fontweight="bold", y=1.01, color="#222222")

# ── Save ───────────────────────────────────────────────────────────────────────
out = "results/meta-signal-results.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
