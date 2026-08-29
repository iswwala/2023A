#!/usr/bin/env python3
"""Compute and plot the 60-state cumulative tower-shadow exposure for Q1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np

from q1_optical_model import (
    SEEDS,
    SUN_HALF_ANGLE,
    TOWER_RADIUS,
    TOWER_Z_MAX,
    build_solar_states,
    load_centers,
    mirror_frames,
    points_from_uv,
    sun_disk_directions,
    surface_uv_all,
    tower_blocks_numpy,
)


ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "results/Q1/experiments/round1/tables/q1_tower_shadow_exposure.csv"
METRIC_PATH = ROOT / "results/Q1/experiments/round1/metrics/q1_tower_shadow_exposure.json"
FIGURE_STEM = ROOT / "paper/figures/q1_tower_shadow_exposure_cn"


def compute_exposure() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heliostat_ids, centers = load_centers()
    states = build_solar_states()
    state_exposure = np.zeros((len(states), len(centers)), dtype=np.float64)

    for seed in SEEDS:
        uv = surface_uv_all(64, seed, heliostat_ids)
        for state in states:
            sun = np.asarray(state.sun_vector, dtype=np.float64)
            _, _, width_axes, height_axes = mirror_frames(centers, sun)
            points = points_from_uv(centers, width_axes, height_axes, uv)
            directions = sun_disk_directions(
                sun,
                32,
                SUN_HALF_ANGLE,
                seed + 100_000 + state.state_index * 101,
            )
            blocked = np.zeros((len(centers), 64), dtype=np.float64)
            for direction in directions:
                blocked += tower_blocks_numpy(points.reshape(-1, 3), direction).reshape(-1, 64)
            state_exposure[state.state_index] += blocked.mean(axis=1) / len(directions)

    state_exposure /= len(SEEDS)
    equivalent_states = state_exposure.sum(axis=0)
    affected_states = np.sum(state_exposure > 0.0, axis=0)
    return centers, equivalent_states, affected_states


def write_outputs(
    centers: np.ndarray, equivalent_states: np.ndarray, affected_states: np.ndarray
) -> dict[str, float | int]:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["heliostat_id", "x_m", "y_m", "tower_shadow_equivalent_states", "affected_state_count"]
        )
        for index, (center, exposure, count) in enumerate(
            zip(centers, equivalent_states, affected_states), start=1
        ):
            writer.writerow([index, center[0], center[1], exposure, int(count)])

    positive = equivalent_states[equivalent_states > 0.0]
    summary: dict[str, float | int] = {
        "schema_version": 1,
        "solar_state_count": 60,
        "surface_samples_per_seed": 64,
        "sun_samples_per_seed": 32,
        "seed_count": len(SEEDS),
        "tower_radius_m": TOWER_RADIUS,
        "tower_height_m": TOWER_Z_MAX,
        "affected_heliostat_count": int(np.sum(equivalent_states > 0.0)),
        "max_equivalent_states": float(np.max(equivalent_states)),
        "mean_equivalent_states_all_heliostats": float(np.mean(equivalent_states)),
        "mean_equivalent_states_affected_heliostats": float(np.mean(positive)) if len(positive) else 0.0,
    }
    METRIC_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def plot(centers: np.ndarray, equivalent_states: np.ndarray, summary: dict[str, float | int]) -> None:
    font_path = ROOT / "paper/fonts/SourceHanSerifCN-Regular.otf"
    if font_path.exists():
        mpl.font_manager.fontManager.addfont(font_path)
        font_family = mpl.font_manager.FontProperties(fname=font_path).get_name()
    else:
        font_family = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 1.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    cmap = LinearSegmentedColormap.from_list(
        "tower_shadow", ["#F4F4F2", "#FFD700", "#E9A6A1", "#B64342"]
    )
    vmax = max(float(summary["max_equivalent_states"]), 1e-6)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    scatter = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=equivalent_states,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        s=12,
        linewidths=0,
        rasterized=True,
    )
    ax.add_patch(Circle((0.0, 0.0), TOWER_RADIUS, facecolor="#222222", edgecolor="white", lw=0.7, zorder=4))
    ax.annotate(
        "塔身投影",
        xy=(0.0, 0.0),
        xytext=(38.0, 28.0),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 0.9},
        ha="left",
        va="bottom",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.03)
    colorbar.set_label("塔影累计暴露量（等效完全遮挡状态数）")
    ax.set_xlabel("东西方向坐标 $x$/m")
    ax.set_ylabel("南北方向坐标 $y$/m")
    ax.set_title("60个规定太阳状态叠加的定日镜塔影暴露分布", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#D8D8D8", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(
        0.01,
        0.01,
        f"受塔影影响定日镜：{summary['affected_heliostat_count']}面；颜色不含镜间阴影",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=360)
    plt.close(fig)


def main() -> None:
    centers, equivalent_states, affected_states = compute_exposure()
    summary = write_outputs(centers, equivalent_states, affected_states)
    plot(centers, equivalent_states, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
