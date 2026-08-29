#!/usr/bin/env python3
"""Generate publication figures for the frozen Q2 design."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"
ROUND_DIR = ROOT / "results" / "Q2" / "experiments" / "round17_zoned_formal"
FONT_REGULAR = ROOT / "paper" / "fonts" / "SourceHanSerifCN-Regular.otf"

BLUE = "#0F4D92"
BLUE_LIGHT = "#3775BA"
GREEN = "#8BCF8B"
RED = "#B64342"
TEAL = "#42949E"
VIOLET = "#9A4D8E"
NEUTRAL = "#767676"
GRID = "#CFCECE"
DARK = "#272727"
SCI_BLUE = "#2563A9"
SCI_VERMILION = "#D45A43"
SCI_GREEN = "#18866B"
SCI_GRAPHITE = "#7A4E9D"


def configure_style() -> None:
    fm.fontManager.addfont(FONT_REGULAR)
    family = fm.FontProperties(fname=FONT_REGULAR).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 1.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=360)
    plt.close(fig)


def figure_layout() -> None:
    layout = pd.read_csv(ROUND_DIR / "tables" / "q2_final_layout.csv")
    x = layout["x_m"].to_numpy()
    y = layout["y_m"].to_numpy()
    tower = np.array([0.0, -33.09241762416299])
    fig, ax = plt.subplots(figsize=(5.8, 5.35), constrained_layout=True)
    ax.scatter(
        x, y, s=8, c=BLUE_LIGHT, linewidths=0,
        alpha=0.82, rasterized=True, label=f"定日镜（{len(layout)}面）",
    )
    ax.add_patch(Circle((0, 0), 350, fill=False, edgecolor=DARK, lw=1.4, label="建设场界"))
    ax.add_patch(Circle(tuple(tower), 100, facecolor="#F6CFCB", edgecolor=RED,
                        lw=1.2, alpha=0.38, label="塔周禁布区"))
    ax.scatter(*tower, s=82, marker="^", c=RED, edgecolors="white", linewidths=0.8,
               zorder=5, label="吸收塔")
    ax.annotate(
        "吸收塔 $(0,-33.09)$ m", xy=tower, xytext=(38, -72),
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 0.9},
        ha="left", va="center", fontsize=8.3,
    )
    ax.set_xlabel("东向坐标 $x$（m）")
    ax.set_ylabel("北向坐标 $y$（m）")
    ax.set_xlim(-365, 365)
    ax.set_ylim(-365, 365)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", ncol=2, handletextpad=0.35,
                       columnspacing=0.75, fontsize=7.5, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.95)
    save(fig, "q2_final_layout_cn")


def figure_monthly() -> None:
    data = pd.read_csv(ROUND_DIR / "tables" / "q2_final_monthly.csv")
    month = data["month"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), constrained_layout=True)

    ax = axes[0]
    series = (
        ("eta_cos", "余弦效率", SCI_BLUE, "o"),
        ("eta_sb", "阴影遮挡效率", SCI_VERMILION, "s"),
        ("eta_trunc", "截断效率", SCI_GREEN, "^"),
        ("eta_total", "综合光学效率", SCI_GRAPHITE, "D"),
    )
    for column, label, color, marker in series:
        ax.plot(month, data[column], color=color, marker=marker, markersize=3,
                markeredgewidth=0.55, lw=1.3, label=label)
    ax.set_xticks(month)
    ax.set_xlabel("月份")
    ax.set_ylabel("月平均效率")
    ax.set_ylim(0.36, 0.97)
    ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.65)
    ax.legend(ncol=2, loc="lower center")
    ax.set_title("月平均光学效率分解")
    ax.text(-0.09, 1.03, "（a）", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    power = data["unit_area_power_kw_m2"].to_numpy()
    ax.plot(month, power, color=SCI_BLUE, marker="o", markersize=3.2,
            markeredgewidth=0.55, lw=1.4)
    ax.fill_between(month, power, power.min() - 0.015, color=SCI_BLUE, alpha=0.07)
    for index in (int(np.argmin(power)), int(np.argmax(power))):
        ax.annotate(
            f"{month[index]}月：{power[index]:.4f}",
            (month[index], power[index]), xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8, color=DARK,
        )
    ax.set_xticks(month)
    ax.set_xlabel("月份")
    ax.set_ylabel("单位面积输出热功率（kW/m²）")
    ax.set_ylim(power.min() - 0.015, power.max() + 0.035)
    ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.65)
    ax.set_title("单位镜面面积月平均输出热功率")
    ax.text(-0.09, 1.03, "（b）", transform=ax.transAxes, fontweight="bold")
    save(fig, "q2_monthly_results_cn")


def main() -> None:
    configure_style()
    figure_layout()
    figure_monthly()
    for path in sorted(FIGURE_DIR.glob("q2_*_cn.*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
