"""
SAGE-QD-DE Framework Diagram — Section 3 (Methodology).
Clean journal-quality version: no title inside figure, larger fonts,
clearer loop arrow.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(15, 9.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── colours ──────────────────────────────────────────────────────────────────
C_MAIN  = "#1F5C8B"
C_SAGE  = "#B85C1A"
C_QD    = "#1E6B40"
C_LLM   = "#4A2D80"
C_EVAL  = "#145F5F"
C_ARROW = "#333333"
C_SAGEBG = "#FEF3EC"
C_QDBG   = "#EAF5EE"
C_LOOPBG = "#EEF3F9"
FONT = "DejaVu Sans"

# ── helpers ──────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, lw=1.8, alpha=1.0):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.07", lw=lw,
                       edgecolor=ec, facecolor=fc, alpha=alpha, zorder=3)
    ax.add_patch(p)

def label(ax, x, y, top, bot=None, tc="#FFFFFF", bc="#DDDDFF", fs_top=9, fs_bot=7.5):
    ax.text(x, y + (0.13 if bot else 0), top,
            color=tc, fontsize=fs_top, fontweight="bold",
            ha="center", va="center", fontfamily=FONT, zorder=4)
    if bot:
        ax.text(x, y - 0.22, bot, color=bc, fontsize=fs_bot,
                ha="center", va="center", fontfamily=FONT, zorder=4,
                style="italic")

def arrow(ax, x0, y0, x1, y1, col=C_ARROW, lw=1.7, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)

def side_label(ax, x, y, s, col, fs=7.8):
    ax.text(x, y, s, color=col, fontsize=fs, fontweight="bold",
            ha="center", va="center", fontfamily=FONT, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=col, lw=1.0, alpha=0.92))

# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND PANELS
# ══════════════════════════════════════════════════════════════════════════════
# Left — SAGE
ax.add_patch(FancyBboxPatch((0.25, 0.3), 3.9, 8.7,
    boxstyle="round,pad=0.1", lw=2, ec=C_SAGE, fc=C_SAGEBG, alpha=0.6, zorder=1))
ax.text(2.2, 8.72, "SAGE Module", color=C_SAGE, fontsize=11,
        fontweight="bold", ha="center", fontfamily=FONT, zorder=4)
ax.text(2.2, 8.38, "Explainable Feature Guidance", color=C_SAGE,
        fontsize=8.5, ha="center", fontfamily=FONT, zorder=4)

# Centre — Search Loop
ax.add_patch(FancyBboxPatch((4.35, 0.3), 6.3, 8.7,
    boxstyle="round,pad=0.1", lw=2.2, ec=C_MAIN, fc=C_LOOPBG, alpha=0.5, zorder=1))
ax.text(7.5, 8.72, "Evolutionary Search Loop", color=C_MAIN, fontsize=11,
        fontweight="bold", ha="center", fontfamily=FONT, zorder=4)

# Right — QD
ax.add_patch(FancyBboxPatch((10.85, 0.3), 3.9, 8.7,
    boxstyle="round,pad=0.1", lw=2, ec=C_QD, fc=C_QDBG, alpha=0.6, zorder=1))
ax.text(12.8, 8.72, "QD Module", color=C_QD, fontsize=11,
        fontweight="bold", ha="center", fontfamily=FONT, zorder=4)
ax.text(12.8, 8.38, "Quality-Diversity Archive", color=C_QD,
        fontsize=8.5, ha="center", fontfamily=FONT, zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# CENTRE — 7-STEP SEARCH LOOP
# ══════════════════════════════════════════════════════════════════════════════
steps = [
    (7.5, 7.65, C_MAIN,  "1.  Initialise Population",       "Random DE algorithm variants"),
    (7.5, 6.50, C_MAIN,  "2.  Tournament Selection",         "From MAP-Elites Archive  (size = 3)"),
    (7.5, 5.35, C_SAGE,  "3.  Build Mutation Prompt",        "DE constraints  +  SHAP feature guidance"),
    (7.5, 4.15, C_LLM,   "4.  Open-Source LLM",              "Generate / Mutate DE Algorithm Code"),
    (7.5, 3.00, C_EVAL,  "5.  Evaluate on MA-BBOB",          "10 training instances  ·  AOCC metric"),
    (7.5, 1.95, C_QD,    "6.  Compute 4-D Behavioral Descriptor", "Adaptation · Recovery · Strategy · Complexity"),
    (7.5, 0.95, C_QD,    "7.  Update MAP-Elites Archive",    "4 × 4 × 4 × 4  =  256 cells"),
]
BOX_W, BOX_H = 5.0, 0.72

for (x, y, col, top, bot) in steps:
    rbox(ax, x, y, BOX_W, BOX_H, col, col)
    label(ax, x, y, top, bot, tc="white", bc="#CCDDFF" if col == C_MAIN else
          ("#FFD8B8" if col == C_SAGE else "#AAFFCC"))

# vertical arrows between steps
ys = [s[1] for s in steps]
for i in range(len(ys) - 1):
    arrow(ax, 7.5, ys[i] - BOX_H/2, 7.5, ys[i+1] + BOX_H/2)

# ── loop-back arrow: dark gray, clearly distinct from QD green ────────────
C_LOOP = "#555555"
lx = 4.7
arrow(ax, 4.97, 0.95, lx, 0.95, col=C_LOOP, lw=2.0)
ax.plot([lx, lx], [0.95, 6.50], color=C_LOOP, lw=2.2, zorder=5,
        solid_capstyle="round")
arrow(ax, lx, 6.50, 4.97, 6.50, col=C_LOOP, lw=2.0)
ax.text(lx - 0.28, 3.7, "Next\nGeneration", color=C_LOOP, fontsize=8,
        fontweight="bold", ha="center", va="center", fontfamily=FONT,
        rotation=90, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — SAGE DETAIL
# ══════════════════════════════════════════════════════════════════════════════
sage_steps = [
    (2.2, 7.00, "AST Feature Extraction",   "28 structural code features"),
    (2.2, 5.85, "XGBoost Regressor",         "Features  →  Fitness (AOCC)"),
    (2.2, 4.70, "TreeSHAP Analysis",          "Most influential feature per parent"),
    (2.2, 3.55, "Natural Language Guidance", '"Increase cyclomatic complexity"'),
]
SBW, SBH = 3.3, 0.72
for (x, y, top, bot) in sage_steps:
    rbox(ax, x, y, SBW, SBH, C_SAGE, C_SAGE)
    label(ax, x, y, top, bot, tc="white", bc="#FFD8B8")

# arrows within SAGE
for i in range(len(sage_steps) - 1):
    y0 = sage_steps[i][1] - SBH/2
    y1 = sage_steps[i+1][1] + SBH/2
    arrow(ax, 2.2, y0, 2.2, y1, col=C_SAGE)

# DE constraints box
rbox(ax, 2.2, 1.90, 3.3, 1.55, "#5A4500", "#C9960C", alpha=0.88)
ax.text(2.2, 2.58, "DE-Only Constraints", color="#FFE566", fontsize=9,
        fontweight="bold", ha="center", fontfamily=FONT, zorder=4)
constraints = [
    ("✓", "rand/1,  best/1,  current-to-best/1"),
    ("✓", "Binomial / Exponential crossover"),
    ("✗", "No PSO, CMA-ES, gradients"),
    ("✗", "No scipy / finite-differences"),
]
for i, (sym, txt) in enumerate(constraints):
    col = "#AAFFAA" if sym == "✓" else "#FF9999"
    ax.text(1.05, 2.28 - i * 0.33, sym, color=col, fontsize=9,
            fontweight="bold", ha="center", fontfamily=FONT, zorder=4)
    ax.text(1.35, 2.28 - i * 0.33, txt, color="#FFF5CC", fontsize=7.8,
            ha="left", va="center", fontfamily=FONT, zorder=4)

# cross-panel arrows
# Parent code → AST
arrow(ax, 5.0, 6.50, 3.85, 7.00, col=C_SAGE, lw=1.8, rad=0.2)
side_label(ax, 4.38, 7.0, "Parent code", C_SAGE)

# NL Guidance → Prompt box (step 3)
arrow(ax, 3.85, 3.55, 5.0, 5.35, col=C_SAGE, lw=1.8, rad=-0.25)
side_label(ax, 4.35, 4.2, "Inject guidance", C_SAGE)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — QD ARCHIVE DETAIL
# ══════════════════════════════════════════════════════════════════════════════
# MAP-Elites grid
ax.text(12.8, 7.9, "MAP-Elites Archive", color=C_QD, fontsize=9.5,
        fontweight="bold", ha="center", fontfamily=FONT, zorder=4)
ax.text(12.8, 7.6, "4-D space  →  4×4×4×4  =  256 cells", color=C_QD,
        fontsize=8, ha="center", fontfamily=FONT, zorder=4)

rng = np.random.default_rng(7)
filled = rng.random((4, 4)) > 0.35
cell, gap = 0.36, 0.06
gx0, gy0 = 11.55, 6.05
for r in range(4):
    for c in range(4):
        cx = gx0 + c * (cell + gap) + cell/2
        cy = gy0 - r * (cell + gap)
        fc = C_QD if filled[r, c] else "#C8E6C9"
        al = 0.9 if filled[r, c] else 0.35
        ax.add_patch(FancyBboxPatch((cx - cell/2, cy - cell/2), cell, cell,
                     boxstyle="round,pad=0.02", lw=0.9,
                     ec=C_QD, fc=fc, alpha=al, zorder=3))
ax.text(12.8, 4.92, "Filled = occupied cell", color=C_QD,
        fontsize=7.5, ha="center", fontfamily=FONT, zorder=4, style="italic")

# Behavioral Dimensions
rbox(ax, 12.8, 3.85, 3.5, 1.55, "#DDEEDD", C_QD, lw=1.4, alpha=0.9)
ax.text(12.8, 4.55, "Behavioral Dimensions", color=C_QD, fontsize=9,
        fontweight="bold", ha="center", fontfamily=FONT, zorder=4)
dims = [
    ("D1", "Adaptation",  "Adaptive F, CR, population control"),
    ("D2", "Recovery",    "Stagnation detection & restart"),
    ("D3", "Strategy",    "rand/1, best/1, current-to-best/1"),
    ("D4", "Complexity",  "Code length & cyclomatic depth"),
]
for i, (di, name, desc) in enumerate(dims):
    y = 4.25 - i * 0.37
    ax.text(11.2, y, di + ":", color=C_QD, fontsize=8.5,
            fontweight="bold", ha="left", fontfamily=FONT, zorder=4)
    ax.text(11.6, y, name, color="#1A4A2E", fontsize=8.5,
            fontweight="bold", ha="left", fontfamily=FONT, zorder=4)
    ax.text(12.8, y - 0.2, desc, color="#2E5A3E", fontsize=7.2,
            ha="center", fontfamily=FONT, zorder=4)

# Best Candidate box
rbox(ax, 12.8, 2.45, 3.5, 0.72, C_QD, C_QD)
label(ax, 12.8, 2.45, "Best Candidate per Method",
      "Selected for validation", tc="white", bc="#AAFFCC")

# Validation Output
rbox(ax, 12.8, 1.35, 3.5, 0.72, C_EVAL, C_EVAL)
label(ax, 12.8, 1.35, "Validation (100 held-out instances)",
      "5-D  ·  10-D  ·  20-D  |  AOCC", tc="white", bc="#AAFFEE")

# arrows in QD panel — grid → Best Candidate → Validation
arrow(ax, 12.8, 4.79, 12.8, 2.82, col=C_QD)   # archive grid bottom → Best Candidate
arrow(ax, 12.8, 2.09, 12.8, 1.72, col=C_QD)   # Best Candidate → Validation

# cross-panel arrows
# FIX 1: arrow comes from Step 6 (descriptor computed there), not Step 5
arrow(ax, 10.03, 1.95, 11.1, 4.80, col=C_QD, lw=1.8, rad=-0.3)
side_label(ax, 10.75, 3.3, "Fitness +\ndescriptor", C_QD)

# FIX 2: "Selected parent" comes from archive GRID (top of QD panel), not dims box
arrow(ax, 11.1, 6.20, 10.03, 6.50, col=C_QD, lw=1.8, rad=0.15)
side_label(ax, 10.62, 6.72, "Selected parent\n(from archive)", C_QD)

# ── legend ───────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_MAIN, label="Search Loop"),
    mpatches.Patch(color=C_SAGE, label="SAGE — Explainability"),
    mpatches.Patch(color=C_QD,   label="QD — MAP-Elites Archive"),
    mpatches.Patch(color=C_LLM,  label="Open-Source LLM"),
    mpatches.Patch(color=C_EVAL, label="Evaluation / Output"),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.02), ncol=5,
          fontsize=9, framealpha=0.95, edgecolor="#AAAAAA",
          handlelength=1.4, handleheight=1.1)

plt.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig0_framework_diagram.pdf"),
            bbox_inches="tight", dpi=150, facecolor="white")
fig.savefig(os.path.join(OUT, "fig0_framework_diagram.png"),
            bbox_inches="tight", dpi=200, facecolor="white")
print("Saved fig0_framework_diagram.pdf and .png")
plt.close(fig)
