import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = [
    ("formula recognition cn",           0.005, 0.694),
    ("document parsing en",              0.389, 0.859),
    ("document parsing cn",              0.368, 0.713),
    ("text translation cn",              0.379, 0.673),
    ("chart parsing en",                 0.323, 0.561),
    ("table parsing cn",                 0.432, 0.688),
    ("table parsing en",                 0.563, 0.803),
    ("cognition VQA cn",                 0.570, 0.812),
    ("key information extraction cn",    0.660, 0.870),
    ("VQA with position en",             0.644, 0.836),
    ("fine-grained text recognition en", 0.410, 0.605),
    ("reasoning VQA en",                 0.495, 0.692),
    ("formula recognition en",           0.163, 0.537),
    ("cognition VQA en",                 0.781, 0.913),
    ("full-page OCR en",                 0.822, 0.935),
    ("full-page OCR cn",                 0.772, 0.842),
    ("APP agent en",                     0.796, 0.868),
    ("text recognition en",              0.759, 0.822),
    ("key information extraction en",    0.853, 0.905),
    ("reasoning VQA cn",                 0.557, 0.612),
    ("science QA en",                    0.617, 0.639),
    ("document classification en",       0.685, 0.697),
    ("math QA en",                       0.420, 0.423),
    ("ASCII art classification en",      0.525, 0.530),
    ("handwritten answer extraction cn", 0.756, 0.763),
    ("diagram QA en",                    0.884, 0.884),
    ("key information mapping en",       0.955, 0.934),
    ("text spotting en",                 0.537, 0.585),
    ("text counting en",                 0.446, 0.343),
    ("text grounding en",                0.512, 0.030),
]

# Sort by delta descending
data.sort(key=lambda x: (x[2] - x[1]), reverse=True)

tasks   = [d[0] for d in data]
stricts = [d[1] for d in data]
judges  = [d[2] for d in data]
deltas  = [j - s for s, j in zip(stricts, judges)]

n = len(tasks)
y = np.arange(n)
h = 0.35

fig, ax = plt.subplots(figsize=(13, 14))
fig.patch.set_facecolor("#f8f7f4")
ax.set_facecolor("#ffffff")

# Bars
bars_s = ax.barh(y + h/2, stricts, height=h, color="#93c5fd", label="Strict Score",        zorder=3)
bars_j = ax.barh(y - h/2, judges,  height=h, color="#1d4ed8", label="Judge Score (GPT-4o)", zorder=3)

# Value labels
for i, (s, j) in enumerate(zip(stricts, judges)):
    ax.text(s + 0.008, i + h/2, f"{s:.3f}", va="center", fontsize=7.5, color="#475569")
    ax.text(j + 0.008, i - h/2, f"{j:.3f}", va="center", fontsize=7.5, color="#1d4ed8")

# Delta labels on the right
for i, d in enumerate(deltas):
    color = "#10b981" if d >= 0.05 else ("#ef4444" if d < -0.05 else "#f59e0b")
    sign  = "+" if d >= 0 else ""
    ax.text(1.02, i, f"{sign}{d:.3f}", va="center", fontsize=8,
            color=color, fontweight="bold", transform=ax.get_yaxis_transform())

# 0.5 threshold line
ax.axvline(0.5, color="#f59e0b", linestyle="--", linewidth=1.5, zorder=2, label="0.5 threshold")

# Grid
ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# Axes
ax.set_yticks(y)
ax.set_yticklabels(tasks, fontsize=9.5, color="#374151")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Score", fontsize=10, color="#374151")
ax.tick_params(axis="x", colors="#94a3b8")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#e2e8f0")

# Legend
patch_s = mpatches.Patch(color="#93c5fd", label="Strict Score")
patch_j = mpatches.Patch(color="#1d4ed8", label="Judge Score (GPT-4o)")
patch_t = plt.Line2D([0], [0], color="#f59e0b", linestyle="--", linewidth=1.5, label="0.5 threshold")
ax.legend(handles=[patch_s, patch_j, patch_t], loc="lower right",
          fontsize=9, framealpha=0.9, edgecolor="#e2e8f0")

# Title
fig.text(0.04, 0.98, "Bolt-8B · Strict vs LLM Judge Scores",
         fontsize=14, fontweight="bold", color="#0f172a", va="top")
fig.text(0.04, 0.965, "OCRBench v2 · 10,000 samples · Judge: GPT-4o · Sorted by delta (Judge − Strict)",
         fontsize=9, color="#64748b", va="top")

# Summary boxes
summary = [
    ("Biggest Gain",     "+0.689", "formula recognition cn", "#10b981"),
    ("Biggest Drop",     "−0.482", "text grounding en",      "#ef4444"),
    ("Tasks Improved ⬆", "27 / 30", "judge > strict",        "#1d4ed8"),
]
for idx, (label, val, sub, color) in enumerate(summary):
    x0 = 0.04 + idx * 0.32
    fig.text(x0, 0.015, label, fontsize=8, color="#94a3b8", va="bottom", transform=fig.transFigure)
    fig.text(x0, 0.032, val,   fontsize=13, fontweight="bold", color=color, va="bottom", transform=fig.transFigure)
    fig.text(x0, 0.005, sub,   fontsize=7.5, color="#94a3b8", va="bottom", transform=fig.transFigure)

plt.tight_layout(rect=[0, 0.05, 1, 0.955])
plt.savefig("bolt8b_strict_vs_judge.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: bolt8b_strict_vs_judge.png")
