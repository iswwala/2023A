from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "code" / "Q1"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "Q2"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "Q4"))

from q1_main import Q1Config, positions_at_time
from q2_collision import Q2Config, board_rectangles, collision_status
from run_fixed_boundary import FixedBoundaryTurnPath, states_at_time
from q4_path import Q4Config


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "figure.dpi": 140,
    "savefig.dpi": 300,
})


COLORS = {
    "blue": "#2B6CB0",
    "red": "#C0392B",
    "green": "#2F855A",
    "gray": "#4A5568",
    "light_gray": "#E2E8F0",
    "orange": "#D97706",
}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def q1_spiral() -> None:
    cfg = Q1Config()
    theta = np.linspace(0, cfg.theta0, 1800)
    r = cfg.b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(x, y, color=COLORS["blue"], lw=1.4)
    ax.scatter([cfg.b * cfg.theta0], [0], s=36, color=COLORS["red"], zorder=3)
    sample = cfg.theta0 * 0.72
    ps = cfg.b * sample * np.array([math.cos(sample), math.sin(sample)])
    ax.plot([0, ps[0]], [0, ps[1]], color=COLORS["orange"], lw=1.2)
    ax.text(ps[0] * 0.52, ps[1] * 0.52, r"$r=b\theta$", fontsize=11)
    ax.annotate(r"$\theta$", xy=(1.2, 0.0), xytext=(1.3, 0.9),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["gray"]),
                fontsize=11, color=COLORS["gray"])
    ax.annotate("head start", xy=(cfg.b * cfg.theta0, 0), xytext=(5.3, 1.2),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["red"]),
                fontsize=10, color=COLORS["red"])
    ax.axhline(0, color=COLORS["light_gray"], lw=0.8)
    ax.axvline(0, color=COLORS["light_gray"], lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("Archimedean spiral used in Q1")
    ax.grid(True, color="#EDF2F7", lw=0.7)
    save(fig, "q1_polar_spiral")


def q2_collision() -> None:
    cfg = Q2Config()
    metrics = json.loads((PROJECT_ROOT / "results/Q2/experiments/round1/metrics/q2_metrics.json").read_text(encoding="utf-8"))
    t_hit = metrics["terminal_bracket_s"][1]
    positions = np.asarray(positions_at_time(t_hit, cfg.q1), dtype=float)
    rects = board_rectangles(positions, cfg)
    pair = tuple(metrics["first_collision_pair"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for idx, rect in enumerate(rects):
        if idx in pair:
            color = COLORS["red"] if idx == pair[0] else COLORS["orange"]
            ax.add_patch(Polygon(rect, closed=True, facecolor=color, edgecolor="#1A202C", alpha=0.55, lw=1.0))
        elif idx < 18:
            ax.add_patch(Polygon(rect, closed=True, facecolor="#CBD5E0", edgecolor="#A0AEC0", alpha=0.35, lw=0.4))
    ax.plot(positions[:32, 0], positions[:32, 1], color=COLORS["blue"], lw=1.0, marker=".", ms=2.0)
    centers = np.array([rects[i].mean(axis=0) for i in pair])
    ax.plot(centers[:, 0], centers[:, 1], color="#1A202C", lw=1.0, ls="--")
    ax.text(centers[0, 0], centers[0, 1], "board 0", fontsize=10, ha="right", va="bottom")
    ax.text(centers[1, 0], centers[1, 1], "board 8", fontsize=10, ha="left", va="top")
    mid = centers.mean(axis=0)
    ax.set_xlim(mid[0] - 3.2, mid[0] + 3.2)
    ax.set_ylim(mid[1] - 2.4, mid[1] + 2.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("First collision geometry near terminal time")
    ax.grid(True, color="#EDF2F7", lw=0.7)
    save(fig, "q2_collision_schematic")


def q4_turn_path() -> None:
    cfg = Q4Config()
    path = FixedBoundaryTurnPath(cfg)
    theta = np.linspace(path.theta0, 32 * math.pi, 1300)
    rin = cfg.b * theta
    xin, yin = rin * np.cos(theta), rin * np.sin(theta)
    xout, yout = -xin, -yin
    t_samples = np.linspace(-25, path.turn_time + 25, 900)
    pts = np.array([path.point(float(t)) for t in t_samples])
    positions, _ = states_at_time(30.0, path, cfg)

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    ax.add_patch(Circle((0, 0), cfg.turn_radius_m, fill=False, color=COLORS["gray"], lw=1.1, ls="--"))
    ax.plot(xin, yin, color="#A0AEC0", lw=0.8)
    ax.plot(xout, yout, color="#CBD5E0", lw=0.8)
    ax.plot(pts[:, 0], pts[:, 1], color=COLORS["red"], lw=2.0, label="turn path")
    ax.plot(positions[:80, 0], positions[:80, 1], color=COLORS["blue"], lw=1.0, marker=".", ms=2.0, label="bench handles")
    pin = path.spiral_point(path.theta0)
    pout = -pin
    ax.scatter([pin[0], pout[0]], [pin[1], pout[1]], color=COLORS["red"], s=28, zorder=4)
    ax.text(pin[0], pin[1], r"$P_{\rm in}$", fontsize=10, ha="right", va="top")
    ax.text(pout[0], pout[1], r"$P_{\rm out}$", fontsize=10, ha="left", va="bottom")
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-8.5, 8.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("Fixed-boundary turning path and bench-dragon state")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, color="#EDF2F7", lw=0.7)
    save(fig, "q4_turn_path_bench")


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.axis("off")
    labels = ["spiral path", "handle recursion", "collision check", "turn path", "speed bound"]
    xs = np.linspace(0.08, 0.92, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.28", fc="#F7FAFC", ec=COLORS["gray"], lw=0.9))
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, 0.55), xytext=(x + 0.08, 0.55),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["gray"]))
    ax.text(0.5, 0.15, "same geometric recursion links Q1--Q5", ha="center", fontsize=10, color=COLORS["gray"])
    save(fig, "overall_workflow")


if __name__ == "__main__":
    workflow()
    q1_spiral()
    q2_collision()
    q4_turn_path()
    print("figures written to", FIG_DIR)
