"""
Plotting script for SAGE-QD-DE paper results.
Generates all figures for the Results section.
All data read from local CSV and JSONL files — nothing shared externally.
"""

import os
import json
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
OUT  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))   # LLaMEA-SAGE/results/figures -> LLaMEA-SAGE

VAL_5_10 = os.path.join(ROOT, "results", "MABBOB_sage_qd_de_validation_best_5_10_seeds10",
                         "validation_results.csv")
VAL_20   = os.path.join(ROOT, "results", "MABBOB_sage_qd_de_validation_best_20_seeds10",
                         "validation_results.csv")
LOG_DIR  = os.path.join(ROOT, "results", "MABBOB_sage_qd_de")

# ── palette & labels ────────────────────────────────────────────────────────
METHODS  = ["DE-LLaMEA", "SAGE-DE", "SAGE-QD-DE"]
COLORS   = {"DE-LLaMEA": "#4C72B0", "SAGE-DE": "#DD8452", "SAGE-QD-DE": "#55A868"}
MARKERS  = {"DE-LLaMEA": "o",        "SAGE-DE": "s",       "SAGE-QD-DE": "^"}
LINESTYLES = {"DE-LLaMEA": "--",     "SAGE-DE": "-.",      "SAGE-QD-DE": "-"}
DIMS     = [5, 10, 20]

# ── helpers ─────────────────────────────────────────────────────────────────

def load_validation(csv_path):
    """Return list of dicts from a validation_results.csv."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["dim"]     = int(row["dim"])
            row["fitness"] = float(row["fitness"]) if row["fitness"] else None
            row["search_fitness"] = float(row["search_fitness"])
            rows.append(row)
    return rows


def group_fitness(rows, method, dim):
    """Return numpy array of per-seed fitness values for a method × dim."""
    return np.array([r["fitness"] for r in rows
                     if r["search_method"] == method
                     and r["dim"] == dim
                     and r["fitness"] is not None])


def load_search_curves():
    """
    Parse all log.jsonl files.
    Returns dict: method -> list (one per seed) of list of (gen, best_so_far).
    Only valid entries (fitness > 0) are used.
    """
    curves = {m: [] for m in METHODS}

    method_map = {
        "DE-LLaMEA": "run-DE-LLaMEA-MA_BBOB",
        "SAGE-DE":   "run-SAGE-DE-MA_BBOB",
        "SAGE-QD-DE":"run-SAGE-QD-DE-MA_BBOB",
    }

    for method, prefix in method_map.items():
        for seed in range(1, 6):
            log_path = os.path.join(LOG_DIR, f"{prefix}-{seed}", "log.jsonl")
            if not os.path.exists(log_path):
                continue
            gen_fitness = {}
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fit = entry.get("fitness", None)
                    gen = entry.get("generation", None)
                    if fit is None or gen is None:
                        continue
                    try:
                        fit = float(fit)
                    except (ValueError, TypeError):
                        continue
                    if not math.isfinite(fit) or fit <= 0:
                        continue
                    if gen not in gen_fitness:
                        gen_fitness[gen] = []
                    gen_fitness[gen].append(fit)

            if not gen_fitness:
                continue

            # best-so-far per generation
            gens = sorted(gen_fitness.keys())
            bsf = []
            current_best = -np.inf
            for g in gens:
                best_in_gen = max(gen_fitness[g])
                current_best = max(current_best, best_in_gen)
                bsf.append((g, current_best))
            curves[method].append(bsf)

    return curves


def avg_curve(seed_curves):
    """Average best-so-far curves across seeds on a common generation grid."""
    if not seed_curves:
        return [], []
    max_gen = max(bsf[-1][0] for bsf in seed_curves)
    grid = list(range(max_gen + 1))

    all_vals = []
    for bsf in seed_curves:
        gen_map = dict(bsf)
        vals = []
        last = 0.0
        for g in grid:
            if g in gen_map:
                last = gen_map[g]
            vals.append(last)
        all_vals.append(vals)

    arr = np.array(all_vals)
    return grid, arr


def wilcoxon_p(a, b):
    """Two-sided Wilcoxon signed-rank p-value; returns 1.0 if not enough data."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative="two-sided")
        return p
    except Exception:
        return 1.0


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── load data ───────────────────────────────────────────────────────────────
rows_5_10 = load_validation(VAL_5_10)
rows_20   = load_validation(VAL_20)
all_rows  = rows_5_10 + rows_20

search_curves = load_search_curves()

# ============================================================
# FIGURE 1 — Box plots: AOCC per method per dimension
# ============================================================
fig1, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
fig1.suptitle("Validation AOCC Distribution Across Methods and Dimensions",
              fontsize=13, fontweight="bold", y=1.01)

dim_labels = ["5-D", "10-D", "20-D"]
for ax_idx, (dim, dlabel) in enumerate(zip(DIMS, dim_labels)):
    ax = axes[ax_idx]
    data_per_method = []
    valid_methods   = []
    for m in METHODS:
        vals = group_fitness(all_rows, m, dim)
        if len(vals) > 0:
            data_per_method.append(vals)
            valid_methods.append(m)

    positions = np.arange(len(valid_methods))
    bp = ax.boxplot(data_per_method, positions=positions, widths=0.5,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="x", markersize=5, alpha=0.6))

    for patch, m in zip(bp["boxes"], valid_methods):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace("-", "-\n") for m in valid_methods], fontsize=8.5)
    ax.set_title(dlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("AOCC (higher = better)" if ax_idx == 0 else "", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=max(0, min(v.min() for v in data_per_method) - 0.03))

    # significance brackets for SAGE-QD-DE vs others
    if "SAGE-QD-DE" in valid_methods:
        qd_idx   = valid_methods.index("SAGE-QD-DE")
        qd_vals  = data_per_method[qd_idx]
        y_top    = max(v.max() for v in data_per_method)
        offset   = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04
        bar_y    = y_top + offset
        for other_idx, other_m in enumerate(valid_methods):
            if other_m == "SAGE-QD-DE":
                continue
            p_val = wilcoxon_p(qd_vals, data_per_method[other_idx])
            sig   = stars(p_val)
            x0, x1 = min(qd_idx, other_idx), max(qd_idx, other_idx)
            ax.annotate("", xy=(x0, bar_y), xytext=(x1, bar_y),
                        arrowprops=dict(arrowstyle="-", color="gray", lw=1.2))
            ax.text((x0 + x1) / 2, bar_y + offset * 0.3, sig,
                    ha="center", va="bottom", fontsize=9)
            bar_y += offset * 1.5

plt.tight_layout()
out1 = os.path.join(OUT, "fig1_boxplots.pdf")
fig1.savefig(out1, bbox_inches="tight", dpi=150)
fig1.savefig(out1.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved {out1}")
plt.close(fig1)

# ============================================================
# FIGURE 2 — Grouped bar chart: mean ± std
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 4.5))
n_dims    = len(DIMS)
n_methods = len(METHODS)
bar_w     = 0.22
group_gap = 0.1
x_centers = np.arange(n_dims)

for m_idx, m in enumerate(METHODS):
    means, stds = [], []
    for dim in DIMS:
        vals = group_fitness(all_rows, m, dim)
        means.append(vals.mean() if len(vals) > 0 else 0)
        stds.append(vals.std()  if len(vals) > 0 else 0)
    offset = (m_idx - (n_methods - 1) / 2) * (bar_w + group_gap / n_methods)
    bars = ax2.bar(x_centers + offset, means, bar_w, yerr=stds,
                   label=m, color=COLORS[m], alpha=0.82,
                   error_kw=dict(elinewidth=1.4, capsize=4, ecolor="black"))
    for bar, mean_val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(stds) * 0.05,
                 f"{mean_val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax2.set_xticks(x_centers)
ax2.set_xticklabels(["5-D", "10-D", "20-D"], fontsize=11)
ax2.set_ylabel("Mean AOCC ± Std  (10 seeds)", fontsize=10)
ax2.set_title("Mean Validation Performance with Standard Deviation",
              fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_ylim(0.45, 0.96)
plt.tight_layout()
out2 = os.path.join(OUT, "fig2_bar_mean_std.pdf")
fig2.savefig(out2, bbox_inches="tight", dpi=150)
fig2.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved {out2}")
plt.close(fig2)

# ============================================================
# FIGURE 3 — Search convergence curves (best-so-far per generation)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(8, 4.5))

for m in METHODS:
    seed_curves = search_curves[m]
    grid, arr   = avg_curve(seed_curves)
    if not grid:
        continue
    mean_bsf = arr.mean(axis=0)
    std_bsf  = arr.std(axis=0)
    ax3.plot(grid, mean_bsf, color=COLORS[m], linestyle=LINESTYLES[m],
             linewidth=2.2, marker=MARKERS[m], markersize=5,
             markevery=max(1, len(grid) // 8), label=m)
    ax3.fill_between(grid, mean_bsf - std_bsf, mean_bsf + std_bsf,
                     color=COLORS[m], alpha=0.15)

ax3.set_xlabel("Generation", fontsize=10)
ax3.set_ylabel("Best AOCC (best-so-far)", fontsize=10)
ax3.set_title("Search-Phase Convergence: Best Fitness per Generation\n(mean ± std across 5 seeds)",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
out3 = os.path.join(OUT, "fig3_search_convergence.pdf")
fig3.savefig(out3, bbox_inches="tight", dpi=150)
fig3.savefig(out3.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved {out3}")
plt.close(fig3)

# ============================================================
# FIGURE 4 — Generalization gap (search fitness vs validation)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(8, 4))

# Use 5-D validation mean as proxy (same training dimension = 10D but reporting 5D/10D)
# compute mean validation fitness per method across 5D and 10D
search_vals = {}
val_5d      = {}
val_10d     = {}
for m in METHODS:
    # search fitness is the same value repeated — take first
    sf = [r["search_fitness"] for r in all_rows if r["search_method"] == m]
    search_vals[m] = sf[0] if sf else 0
    val_5d[m]  = group_fitness(all_rows, m, 5).mean()
    val_10d[m] = group_fitness(all_rows, m, 10).mean()

x = np.arange(len(METHODS))
w = 0.25
ax4.bar(x - w,     [search_vals[m] for m in METHODS], w, label="Search fitness (10-D train)",
        color=[COLORS[m] for m in METHODS], alpha=0.5, hatch="//")
ax4.bar(x,         [val_5d[m]      for m in METHODS], w, label="Validation (5-D held-out)",
        color=[COLORS[m] for m in METHODS], alpha=0.85, hatch="")
ax4.bar(x + w,     [val_10d[m]     for m in METHODS], w, label="Validation (10-D held-out)",
        color=[COLORS[m] for m in METHODS], alpha=0.55, hatch="xx")

# annotate gap arrows for 5-D
for i, m in enumerate(METHODS):
    gap = search_vals[m] - val_5d[m]
    if gap > 0:
        ax4.annotate(f"Δ={gap:.3f}",
                     xy=(x[i], val_5d[m] + 0.002),
                     xytext=(x[i], search_vals[m] - 0.002),
                     fontsize=7.5, ha="center", color="black",
                     arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))

ax4.set_xticks(x)
ax4.set_xticklabels(METHODS, fontsize=10)
ax4.set_ylabel("AOCC", fontsize=10)
ax4.set_title("Generalization Gap: Search Fitness vs. Held-Out Validation",
              fontsize=11, fontweight="bold")
ax4.legend(fontsize=8, loc="lower right")
ax4.set_ylim(0.70, 0.92)
ax4.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
out4 = os.path.join(OUT, "fig4_generalization_gap.pdf")
fig4.savefig(out4, bbox_inches="tight", dpi=150)
fig4.savefig(out4.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved {out4}")
plt.close(fig4)

# ============================================================
# FIGURE 5 — Dimensional scalability
# ============================================================
fig5, ax5 = plt.subplots(figsize=(7, 4.5))

for m in METHODS:
    means = [group_fitness(all_rows, m, d).mean() for d in DIMS]
    ax5.plot(DIMS, means, color=COLORS[m], linestyle=LINESTYLES[m],
             linewidth=2.2, marker=MARKERS[m], markersize=8, label=m)
    for d, mean_v in zip(DIMS, means):
        ax5.annotate(f"{mean_v:.3f}", (d, mean_v),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=8, color=COLORS[m])

ax5.set_xticks(DIMS)
ax5.set_xticklabels(["5-D", "10-D", "20-D"], fontsize=11)
ax5.set_xlabel("Problem Dimensionality", fontsize=10)
ax5.set_ylabel("Mean AOCC (10 seeds)", fontsize=10)
ax5.set_title("Scalability: Mean AOCC vs. Problem Dimension",
              fontsize=11, fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
out5 = os.path.join(OUT, "fig5_scalability.pdf")
fig5.savefig(out5, bbox_inches="tight", dpi=150)
fig5.savefig(out5.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved {out5}")
plt.close(fig5)

# ============================================================
# FIGURE 6 — Wilcoxon statistical significance heatmap
# ============================================================
comparisons = [
    ("SAGE-QD-DE", "DE-LLaMEA"),
    ("SAGE-QD-DE", "SAGE-DE"),
    ("DE-LLaMEA",  "SAGE-DE"),
]
comp_labels = [f"{a}\nvs\n{b}" for a, b in comparisons]
p_matrix    = np.ones((len(DIMS), len(comparisons)))

for d_idx, dim in enumerate(DIMS):
    data = {m: group_fitness(all_rows, m, dim) for m in METHODS}
    for c_idx, (ma, mb) in enumerate(comparisons):
        a_vals = data[ma]
        b_vals = data[mb]
        min_n  = min(len(a_vals), len(b_vals))
        p_matrix[d_idx, c_idx] = wilcoxon_p(a_vals[:min_n], b_vals[:min_n])

fig6, ax6 = plt.subplots(figsize=(8, 3.5))
im = ax6.imshow(p_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")

ax6.set_xticks(np.arange(len(comparisons)))
ax6.set_xticklabels(comp_labels, fontsize=9)
ax6.set_yticks(np.arange(len(DIMS)))
ax6.set_yticklabels(["5-D", "10-D", "20-D"], fontsize=10)
ax6.set_title("Wilcoxon Signed-Rank p-values (two-sided, 10 seeds)\nGreen = significant (p<0.05), Red = not significant",
              fontsize=10, fontweight="bold")

for d_idx in range(len(DIMS)):
    for c_idx in range(len(comparisons)):
        p = p_matrix[d_idx, c_idx]
        label = f"p={p:.4f}\n{stars(p)}"
        ax6.text(c_idx, d_idx, label, ha="center", va="center",
                 fontsize=8.5, fontweight="bold",
                 color="white" if p < 0.03 else "black")

plt.colorbar(im, ax=ax6, label="p-value", shrink=0.8)
plt.tight_layout()
out6 = os.path.join(OUT, "fig6_wilcoxon.pdf")
fig6.savefig(out6, bbox_inches="tight", dpi=150)
fig6.savefig(out6.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved {out6}")
plt.close(fig6)

# ============================================================
# PRINT SUMMARY TABLE (for paper Table 1)
# ============================================================
print("\n" + "="*72)
print("TABLE 1 - Mean AOCC +/- Std  (10 seeds, 100 held-out test instances)")
print("="*72)
header = f"{'Method':<15}" + "".join(f"{'Dim='+str(d):<18}" for d in DIMS)
print(header)
print("-"*72)
for m in METHODS:
    row = f"{m:<15}"
    for dim in DIMS:
        vals = group_fitness(all_rows, m, dim)
        if len(vals) > 0:
            row += f"{vals.mean():.4f} ± {vals.std():.4f}  "
        else:
            row += f"{'—':<18}"
    print(row)
print("="*72)

print("\n" + "="*72)
print("TABLE 2 - Generalization Gap  (search fitness minus mean validation)")
print("="*72)
print(f"{'Method':<15} {'Search (10-D)':<16} {'Val 5-D':<12} {'Val 10-D':<12} {'Gap(5D)':<10} {'Gap(10D)'}")
print("-"*72)
for m in METHODS:
    sf = search_vals[m]
    v5 = val_5d[m]
    v10 = val_10d[m]
    print(f"{m:<15} {sf:.4f}           {v5:.4f}       {v10:.4f}       {sf-v5:.4f}     {sf-v10:.4f}")
print("="*72)

print(f"\nAll figures saved to: {OUT}")
