"""
Ablation Study — Presentation Plot
Jensen-only GA vs Random Baseline (both FLORIS-scored, apples-to-apples)
Run: python plot_ablation_comparison.py
Saves: ablation_comparison.png  (300 dpi, transparent background optional)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
labels = ["Random Baseline", "Jensen-only GA"]
aep = [3923.46, 3950.41]  # GWh/yr  (both FLORIS-scored)
colors = ["#4A5568", "#38B2AC"]  # slate-grey  vs  teal

delta_gwh = aep[1] - aep[0]  # +11.16 GWh/yr
delta_pct = (delta_gwh / aep[0]) * 100  # +0.28 %

# ── Style ─────────────────────────────────────────────────────────────────────
BG = "#0F1117"
PANEL = "#1A1D27"
GRID = "#2A2D3A"
TEXT = "#E8EAF0"
SUBTEXT = "#8B90A0"
ACCENT = "#38B2AC"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    }
)

fig = plt.figure(figsize=(10, 6.5))
fig.patch.set_facecolor(BG)

# ── Layout: left = bar chart, right = annotation panel ────────────────────────
gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.6, 1],
    wspace=0.08,
    left=0.07,
    right=0.97,
    top=0.84,
    bottom=0.13,
)

ax = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1])
ax_r.set_visible(False)  # we'll draw on fig directly

# ── Bar chart ─────────────────────────────────────────────────────────────────
x = np.array([0, 1])
bar_width = 0.52

bars = ax.bar(
    x, aep, width=bar_width, color=colors, zorder=3, edgecolor="none", linewidth=0
)

# Subtle inner highlight on top edge
for bar, c in zip(bars, colors):
    ax.plot(
        [bar.get_x(), bar.get_x() + bar.get_width()],
        [bar.get_height(), bar.get_height()],
        color=c,
        linewidth=2.5,
        alpha=0.9,
        zorder=4,
    )

# AEP value labels inside/above bars
for bar, val in zip(bars, aep):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val - 5,
        f"{val:.2f}\nGWh/yr",
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="white",
        linespacing=1.55,
        zorder=5,
    )

# Gain arrow between the two bars
y_arrow = max(aep) + 6
ax.annotate(
    "",
    xy=(x[1], y_arrow - 0.5),
    xytext=(x[0], y_arrow - 0.5),
    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.8, mutation_scale=14),
)
ax.text(
    (x[0] + x[1]) / 2,
    y_arrow + 2,
    f"Δ +{delta_gwh:.2f} GWh/yr  (+{delta_pct:.2f}%)",
    ha="center",
    va="bottom",
    fontsize=10,
    color=ACCENT,
    fontweight="bold",
)

# Axes cosmetics
y_min = min(aep) - 30
y_max = max(aep) + 28
ax.set_ylim(y_min, y_max)
ax.set_xlim(-0.55, 1.55)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11.5, fontweight="semibold")
ax.set_ylabel("Annual Energy Production  (GWh/yr)", fontsize=11, labelpad=8)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.spines["left"].set_color(GRID)

# Colour-coded x-tick labels
ax.get_xticklabels()[0].set_color(colors[0])
ax.get_xticklabels()[1].set_color(colors[1])

# ── Right panel: "Why apples-to-apples?" info box ─────────────────────────────
panel_left = 0.625
info_ax = fig.add_axes([panel_left, 0.13, 0.345, 0.71])
info_ax.set_facecolor(PANEL)
info_ax.set_xlim(0, 1)
info_ax.set_ylim(0, 1)
info_ax.axis("off")

# Panel border
for spine_pos in [(0, 0, 1, 0), (1, 0, 1, 1), (0, 1, 1, 1), (0, 0, 0, 1)]:
    info_ax.plot(
        [spine_pos[0], spine_pos[2]],
        [spine_pos[1], spine_pos[3]],
        color=GRID,
        linewidth=1.0,
        transform=info_ax.transAxes,
    )


def row(ax, y, icon, label, value, label_col=SUBTEXT, val_col=TEXT):
    ax.text(0.08, y, icon, ha="left", va="center", fontsize=13, color=val_col)
    ax.text(0.22, y, label, ha="left", va="center", fontsize=9.5, color=label_col)
    ax.text(
        0.92,
        y,
        value,
        ha="right",
        va="center",
        fontsize=9.5,
        color=val_col,
        fontweight="bold",
    )


info_ax.text(
    0.5,
    0.94,
    "Apples-to-Apples",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=TEXT,
)
info_ax.text(
    0.5,
    0.87,
    "Both scored by FLORIS (Gauss)",
    ha="center",
    va="center",
    fontsize=8.8,
    color=ACCENT,
)

# Divider
info_ax.plot([0.06, 0.94], [0.82, 0.82], color=GRID, linewidth=0.8)

row(info_ax, 0.73, "▸", "Wake model", "FLORIS Gauss")
row(info_ax, 0.63, "▸", "Turbines", "150")
row(info_ax, 0.53, "▸", "Seed", "42")
row(info_ax, 0.43, "▸", "Population", "50 layouts")

info_ax.plot([0.06, 0.94], [0.35, 0.35], color=GRID, linewidth=0.8)

info_ax.text(
    0.5, 0.27, "Fitness signal", ha="center", va="center", fontsize=9, color=SUBTEXT
)
row(info_ax, 0.19, "✕", "Random baseline", "none (random)", val_col="#FC8181")
row(info_ax, 0.10, "✓", "Jensen-only GA", "Jensen GPU", val_col="#68D391")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(
    0.5,
    0.935,
    "Ablation Study — Jensen-only GA vs Random Baseline",
    ha="center",
    va="center",
    fontsize=14.5,
    fontweight="bold",
    color=TEXT,
)
fig.text(
    0.5,
    0.895,
    "FLORIS-scored comparison · 150 turbines · 10 × 10 km site",
    ha="center",
    va="center",
    fontsize=10,
    color=SUBTEXT,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "ablation_comparison.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.show()
