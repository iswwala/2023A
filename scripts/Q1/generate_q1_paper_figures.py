#!/usr/bin/env python3
"""Generate deterministic Chinese paper figures for Q1 from approved evidence."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES = ROOT / "paper" / "figures"
ROUND_DIR = ROOT / "results" / "Q1" / "experiments" / "round1"
FONT_REGULAR = ROOT / "paper" / "fonts" / "SourceHanSerifCN-Regular.otf"
FONT_BOLD = ROOT / "paper" / "fonts" / "SourceHanSerifCN-Bold.otf"

PRIMARY = "#0F4D92"
PRIMARY_LIGHT = "#3775BA"
BASELINE = "#767676"
ORANGE = "#D99000"
TEAL = "#42949E"
PURPLE = "#9A4D8E"
RED = "#B64342"
DARK = "#272727"
GRID = "#CFCECE"
PALE = "#DDF3DE"
SCI_BLUE = "#2563A9"
SCI_VERMILION = "#D45A43"
SCI_GREEN = "#18866B"
SCI_GRAPHITE = "#7A4E9D"
SCHEMATIC_BLUE = "#315E7D"
SCHEMATIC_GRAY = "#777777"
SCHEMATIC_LIGHT = "#E8E8E8"


def configure_style() -> None:
    for path in (FONT_REGULAR, FONT_BOLD):
        fm.fontManager.addfont(path)
    regular_name = fm.FontProperties(fname=FONT_REGULAR).get_name()
    plt.rcParams.update(
        {
            "font.family": regular_name,
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


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.09, 1.03, text, transform=ax.transAxes, fontsize=10, fontweight="bold")


def save_figure(fig: plt.Figure, stem: str) -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / f"{stem}.pdf")
    fig.savefig(PAPER_FIGURES / f"{stem}.png", dpi=360)
    plt.close(fig)


def save_axis_figure(fig: plt.Figure, ax: plt.Axes, stem: str) -> None:
    """Export one panel of a construction as a self-contained paper figure."""
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted()
    ).expanded(1.03, 1.06)
    fig.savefig(PAPER_FIGURES / f"{stem}.pdf", bbox_inches=bbox)
    fig.savefig(PAPER_FIGURES / f"{stem}.png", dpi=360, bbox_inches=bbox)


def figure_model_workflow() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.9), constrained_layout=True)
    nodes = [
        ("太阳位置与DNI", "统一全部光学输入", "#FBE7C6"),
        ("镜面姿态与有限边界", "确定余弦损失和真实相交", "#DCEBF7"),
        ("镜面点--太阳盘联合样本", "让各项损失作用于同一份能量", "#E8F3EE"),
        ("入射阴影--反射--出射遮挡--接收命中", "按传播顺序避免重复计损", "#E8F3EE"),
        ("逐镜逐时点功率，再作月、年聚合", "保留DNI与效率的同步变化", "#E6E0F3"),
    ]
    y_values = [3.18, 2.45, 1.72, 0.99, 0.26]
    box_x, box_w, box_h = 0.58, 5.04, 0.48
    for index, (title, reason, face) in enumerate(nodes):
        y = y_values[index]
        ax.add_patch(Rectangle((box_x, y), box_w, box_h, facecolor=face, edgecolor=DARK, lw=0.9))
        ax.text(box_x + 1.58, y + box_h / 2, title, ha="center", va="center", fontsize=8.7, fontweight="bold")
        ax.plot([box_x + 3.08, box_x + 3.08], [y + 0.08, y + box_h - 0.08], color=BASELINE, lw=0.7)
        ax.text(box_x + 4.03, y + box_h / 2, reason, ha="center", va="center", fontsize=7.8, color=DARK)
        if index < len(nodes) - 1:
            ax.add_patch(FancyArrowPatch((3.10, y - 0.02), (3.10, y_values[index + 1] + box_h + 0.02),
                                         arrowstyle="-|>", mutation_scale=10, lw=1.1, color=PRIMARY))
    ax.text(3.10, 3.82, "问题一统一光学评价流程", ha="center", fontsize=10.5, fontweight="bold", color=PRIMARY)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0.12, 4.02)
    ax.axis("off")
    save_figure(fig, "q1_model_workflow_cn")


def solar_states() -> pd.DataFrame:
    latitude = math.radians(39.4)
    days = (-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275)
    times = (9.0, 10.5, 12.0, 13.5, 15.0)
    rows = []
    for month, day in enumerate(days, start=1):
        declination = math.asin(
            math.sin(2 * math.pi * day / 365) * math.sin(math.radians(23.45))
        )
        for solar_time in times:
            hour_angle = math.pi / 12 * (solar_time - 12)
            altitude = math.asin(
                math.cos(declination) * math.cos(latitude) * math.cos(hour_angle)
                + math.sin(declination) * math.sin(latitude)
            )
            cos_azimuth = (
                math.sin(declination) - math.sin(altitude) * math.sin(latitude)
            ) / (math.cos(altitude) * math.cos(latitude))
            base = math.acos(np.clip(cos_azimuth, -1, 1))
            azimuth = base if hour_angle <= 0 else 2 * math.pi - base
            a = 0.4237 - 0.00821 * (6 - 3) ** 2
            b = 0.5055 + 0.00595 * (6.5 - 3) ** 2
            c = 0.2711 + 0.01858 * (2.5 - 3) ** 2
            dni = 1.366 * (a + b * math.exp(-c / math.sin(altitude)))
            rows.append(
                {
                    "月份": month,
                    "时刻": solar_time,
                    "高度角": math.degrees(altitude),
                    "方位角": math.degrees(azimuth),
                    "DNI": dni,
                }
            )
    return pd.DataFrame(rows)


def figure_solar_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)
    alpha = math.radians(42)
    gamma = math.radians(125)
    ax = axes[0]
    ax.add_patch(Circle((0, 0), 0.90, fill=False, edgecolor=BASELINE, lw=1.3))
    ax.add_patch(FancyArrowPatch((0, -1.04), (0, 1.12), arrowstyle="-|>", mutation_scale=12, color=DARK))
    ax.add_patch(FancyArrowPatch((-1.04, 0), (1.12, 0), arrowstyle="-|>", mutation_scale=12, color=DARK))
    horizontal = np.array([math.sin(gamma), math.cos(gamma)])
    ax.add_patch(FancyArrowPatch((0, 0), tuple(horizontal * 0.88), arrowstyle="-|>", mutation_scale=14, lw=2.3, color=ORANGE))
    az = np.linspace(0, gamma, 100)
    ax.plot(0.40 * np.sin(az), 0.40 * np.cos(az), color=PRIMARY, lw=2)
    ax.text(0, 1.18, "北（y）", ha="center")
    ax.text(1.16, 0, "东（x）", va="center")
    ax.text(*(horizontal * 1.04), "太阳方向的\n水平投影", color=ORANGE, ha="center", va="center")
    ax.text(0.42, 0.18, "方位角 $\\gamma_s$\n从正北顺时针", color=PRIMARY, ha="center")
    ax.set_title("水平面：确定太阳所在方位")
    ax.set_xlim(-1.18, 1.30)
    ax.set_ylim(-1.15, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    ax.add_patch(FancyArrowPatch((-1.0, 0), (1.12, 0), arrowstyle="-|>", mutation_scale=12, color=DARK))
    ax.add_patch(FancyArrowPatch((0, -0.12), (0, 1.12), arrowstyle="-|>", mutation_scale=12, color=DARK))
    endpoint = np.array([math.cos(alpha), math.sin(alpha)])
    ax.add_patch(FancyArrowPatch((0, 0), tuple(endpoint), arrowstyle="-|>", mutation_scale=14, lw=2.3, color=ORANGE))
    aa = np.linspace(0, alpha, 80)
    ax.plot(0.45 * np.cos(aa), 0.45 * np.sin(aa), color=TEAL, lw=2)
    ax.plot([endpoint[0], endpoint[0]], [0, endpoint[1]], color=BASELINE, ls="--", lw=1)
    ax.text(1.14, 0, "地平面", va="center")
    ax.text(0, 1.17, "天顶（z）", ha="center")
    ax.text(*(endpoint * 1.12), "太阳方向 $\\boldsymbol{s}$", color=ORANGE, ha="center")
    ax.text(0.46, 0.18, "高度角 $\\alpha_s$", color=TEAL, ha="center")
    ax.text(0.70, -0.18, "水平投影", color=BASELINE, ha="center")
    ax.set_title("铅垂面：确定太阳离地高度")
    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(-0.25, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(axes[0], "（a）")
    panel_label(axes[1], "（b）")
    save_figure(fig, "q1_solar_geometry_cn")


def figure_solar_states() -> None:
    data = solar_states()
    times = [9.0, 10.5, 12.0, 13.5, 15.0]
    altitude = data.pivot(index="月份", columns="时刻", values="高度角").values
    dni = data.pivot(index="月份", columns="时刻", values="DNI").values
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), constrained_layout=True)
    for ax, matrix, title, label, cmap, fmt in [
        (axes[0], altitude, "太阳高度角", "角度（°）", "YlOrBr", ".0f"),
        (axes[1], dni, "法向直接辐射辐照度", "DNI（kW/m²）", "Blues", ".2f"),
    ]:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        for row in range(12):
            for col in range(5):
                rgba = image.cmap(image.norm(matrix[row, col]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    col,
                    row,
                    format(matrix[row, col], fmt),
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if luminance < 0.52 else DARK,
                )
        ax.set_xticks(range(5), ["9:00", "10:30", "12:00", "13:30", "15:00"])
        ax.set_yticks(range(12), [f"{m}月" for m in range(1, 13)])
        ax.set_xlabel("当地时间")
        ax.set_ylabel("代表日期（每月21日）")
        ax.set_title(title)
        cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label(label)
    panel_label(axes[0], "（a）")
    panel_label(axes[1], "（b）")
    save_figure(fig, "q1_solar_states_cn")


def figure_reflection_geometry() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.65), constrained_layout=True)
    center = np.array([0.0, 0.0])
    sun = np.array([-0.72, 0.69])
    sun /= np.linalg.norm(sun)
    target = np.array([0.86, 0.51])
    target /= np.linalg.norm(target)
    normal = sun + target
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    incident = -sun
    reflected = incident - 2 * np.dot(incident, normal) * normal
    assert np.linalg.norm(reflected - target) < 1e-12
    mirror = np.vstack((-0.52 * tangent, 0.52 * tangent))
    ax1.plot(mirror[:, 0], mirror[:, 1], color=DARK, lw=1.8, marker="o",
             markersize=3.2, markerfacecolor="white")
    source = 1.03 * sun
    ax1.add_patch(FancyArrowPatch(tuple(source), tuple(center), arrowstyle="-|>",
                                  mutation_scale=13, color=SCHEMATIC_GRAY, lw=1.7))
    ax1.add_patch(FancyArrowPatch(tuple(center), tuple(target * 1.02), arrowstyle="-|>",
                                  mutation_scale=13, color=SCHEMATIC_BLUE, lw=1.7))
    ax1.add_patch(FancyArrowPatch(tuple(center), tuple(normal * 0.86), arrowstyle="-|>",
                                  mutation_scale=12, color=DARK, lw=1.5))
    ax1.plot([0, source[0]], [0, source[1]], color=SCHEMATIC_GRAY, lw=0.8, ls=":")
    theta_s = math.atan2(sun[1], sun[0])
    theta_n = math.atan2(normal[1], normal[0])
    theta_t = math.atan2(target[1], target[0])
    ax1.add_patch(Arc((0, 0), 0.68, 0.68, theta1=math.degrees(theta_n),
                      theta2=math.degrees(theta_s), color=SCHEMATIC_GRAY, lw=1.2))
    ax1.add_patch(Arc((0, 0), 0.83, 0.83, theta1=math.degrees(theta_t),
                      theta2=math.degrees(theta_n), color=SCHEMATIC_BLUE, lw=1.2))
    ax1.text(-0.49, 0.78, "太阳中心光线\n传播方向 $-\\boldsymbol{s}$", color=DARK, ha="center")
    ax1.text(0.73, 0.48, "反射方向 $\\boldsymbol{t}_i$", color=DARK, ha="center")
    ax1.text(*(normal * 0.98), "法向 $\\boldsymbol{n}_i$", color=DARK, ha="center")
    ax1.text(-0.13, 0.36, "$\\theta_i$", color=DARK)
    ax1.text(0.27, 0.33, "$\\theta_r$", color=DARK)
    ax1.text(0, -0.58, "$\\theta_i=\\theta_r$，且 $\\mathcal{R}(-\\boldsymbol{s},\\boldsymbol{n}_i)=\\boldsymbol{t}_i$",
             ha="center", fontsize=8.2)
    ax1.set_title("反射定律确定唯一法向")
    ax1.set_xlim(-1.15, 1.2)
    ax1.set_ylim(-0.75, 1.1)
    ax1.set_aspect("equal")
    ax1.axis("off")

    mirror_polygon = Polygon(
        [(-0.92, -0.25), (0.65, -0.25), (0.92, 0.20), (-0.65, 0.20)],
        closed=True,
        facecolor=SCHEMATIC_LIGHT,
        edgecolor=DARK,
        lw=1.4,
    )
    ax2.add_patch(mirror_polygon)
    center = np.array([0.0, -0.025])
    vectors = [
        (np.array([0.95, 0.0]), "宽向 $\\boldsymbol{e}_{w,i}$", SCHEMATIC_BLUE, (0.57, -0.15)),
        (np.array([-0.28, 0.72]), "高向 $\\boldsymbol{e}_{h,i}$", SCHEMATIC_GRAY, (-0.54, 0.58)),
        (np.array([0.10, 0.95]), "法向 $\\boldsymbol{n}_i$", DARK, (0.30, 0.84)),
    ]
    for vector, label, color, label_pos in vectors:
        ax2.add_patch(FancyArrowPatch(tuple(center), tuple(center + vector), arrowstyle="-|>", mutation_scale=13, lw=1.6, color=color))
        ax2.text(*label_pos, label, color=DARK, ha="center")
    ax2.text(0, -0.66, "$u,v\\in[-3,3]$ m 参数化有限矩形", ha="center")
    ax2.set_title("局部坐标限定真实镜面边界")
    ax2.set_xlim(-1.15, 1.18)
    ax2.set_ylim(-0.78, 1.10)
    ax2.set_aspect("equal")
    ax2.axis("off")
    panel_label(ax1, "（a）")
    panel_label(ax2, "（b）")
    save_figure(fig, "q1_reflection_geometry_cn")


def figure_single_mirror_tracking() -> None:
    positions = pd.read_csv(
        ROOT / "workspace" / "data_clean" / "Q1" / "heliostat_positions.csv"
    )
    mirror = positions.loc[positions["heliostat_id"] == 1].iloc[0]
    center = np.array([mirror["x_m"], mirror["y_m"], mirror["z_m"]], dtype=float)
    receiver = np.array([0.0, 0.0, 76.0])
    target = receiver - center
    target /= np.linalg.norm(target)
    state = solar_states()
    state = state[state["月份"] == 3].sort_values("时刻")
    normal_azimuth = []
    mirror_tilt = []
    for row in state.itertuples(index=False):
        altitude = math.radians(row.高度角)
        azimuth = math.radians(row.方位角)
        sun = np.array(
            [
                math.cos(altitude) * math.sin(azimuth),
                math.cos(altitude) * math.cos(azimuth),
                math.sin(altitude),
            ]
        )
        normal = sun + target
        normal /= np.linalg.norm(normal)
        normal_azimuth.append(
            math.degrees(math.atan2(normal[0], normal[1])) % 360.0
        )
        mirror_tilt.append(math.degrees(math.acos(normal[2])))

    times = state["时刻"].to_numpy()
    labels = ["9:00", "10:30", "12:00", "13:30", "15:00"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        times,
        normal_azimuth,
        color=PRIMARY,
        marker="o",
        markersize=5,
        linewidth=2,
    )
    for x, value in zip(times, normal_azimuth):
        ax.annotate(
            f"{value:.1f}°",
            (x, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(times, labels)
    ax.set_ylim(0, 360)
    ax.set_yticks(
        [0, 90, 180, 270, 360],
        ["北 0°", "东 90°", "南 180°", "西 270°", "北 360°"],
    )
    ax.set_xlabel("当地时间")
    ax.set_ylabel("法向方位角 $\\beta_1$")
    ax.set_title("绕竖直轴调整法向方位")
    clean_axis(ax)

    ax = axes[1]
    ax.plot(
        times,
        mirror_tilt,
        color=ORANGE,
        marker="s",
        markersize=5,
        linewidth=2,
    )
    for x, value in zip(times, mirror_tilt):
        ax.annotate(
            f"{value:.1f}°",
            (x, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(times, labels)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("当地时间")
    ax.set_ylabel("镜面倾角 $\\tau_1$")
    ax.set_title("绕水平轴调整镜面倾斜程度")
    clean_axis(ax)
    panel_label(axes[0], "（a）")
    panel_label(axes[1], "（b）")
    fig.suptitle(
        "第1面镜（x=107.25 m，y=11.664 m）在3月21日的中心瞄准姿态",
        fontsize=10,
    )
    save_figure(fig, "q1_single_mirror_tracking_cn")


def draw_mirror(ax: plt.Axes, center: tuple[float, float], angle: float, color: str = SCHEMATIC_BLUE) -> None:
    direction = np.array([math.cos(angle), math.sin(angle)])
    segment = np.vstack((np.array(center) - 0.45 * direction, np.array(center) + 0.45 * direction))
    ax.plot(segment[:, 0], segment[:, 1], color=color, lw=1.6, marker="o",
            markersize=3.2, markerfacecolor="white", markeredgewidth=0.9)


def figure_shadow_blocking() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), constrained_layout=True)
    ax = axes[0]
    draw_mirror(ax, (0, -0.25), 0.08)
    draw_mirror(ax, (-0.58, 0.52), 0.08, BASELINE)
    tower = Rectangle((0.62, 0.10), 0.20, 1.20, facecolor="#D9D9D9", edgecolor=DARK)
    ax.add_patch(tower)
    ax.add_patch(FancyArrowPatch((-1.18, 1.32), (-0.58, 0.52), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=SCHEMATIC_GRAY))
    ax.plot([-0.58, 0.0], [0.52, -0.25], color=SCHEMATIC_GRAY, lw=1.0, ls="--", alpha=0.75)
    ax.scatter([-0.58], [0.52], s=42, marker="x", color=DARK, zorder=5)
    ax.text(-1.17, 1.43, "来自太阳的入射光", color=DARK)
    ax.text(-0.98, 0.37, "先与邻镜相交", color=DARK)
    ax.text(0.72, 1.36, "塔身也作\n入射相交判定", ha="center", color=DARK, fontsize=8)
    ax.text(0, -0.48, "目标镜", ha="center")
    ax.set_title("入射阴影：邻镜或塔身截断光线")

    ax = axes[1]
    target_center = np.array([-0.75, -0.55])
    receiver_center = np.array([0.92, 1.16])
    outgoing = receiver_center - target_center
    outgoing /= np.linalg.norm(outgoing)
    incoming = np.array([0.46, -0.89])
    incoming /= np.linalg.norm(incoming)
    normal = incoming - outgoing
    normal /= np.linalg.norm(normal)
    reflected = incoming - 2 * np.dot(incoming, normal) * normal
    assert np.linalg.norm(reflected - outgoing) < 1e-12
    tangent = np.array([-normal[1], normal[0]])
    mirror_angle = math.atan2(tangent[1], tangent[0])
    draw_mirror(ax, tuple(target_center), mirror_angle)
    blocker = target_center + 0.72 * outgoing
    draw_mirror(ax, tuple(blocker), mirror_angle + 0.22, BASELINE)
    receiver = Rectangle((0.82, 0.75), 0.20, 0.85, facecolor=SCHEMATIC_LIGHT, edgecolor=DARK)
    ax.add_patch(receiver)
    source = target_center - 0.95 * incoming
    ax.add_patch(FancyArrowPatch(tuple(source), tuple(target_center), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=SCHEMATIC_GRAY))
    ax.add_patch(FancyArrowPatch(tuple(target_center), tuple(blocker), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=SCHEMATIC_BLUE))
    ax.plot([blocker[0], receiver_center[0]], [blocker[1], receiver_center[1]], color=SCHEMATIC_BLUE, lw=1.0, ls="--", alpha=0.75)
    ax.scatter(*blocker, s=42, marker="x", color=DARK, zorder=5)
    ax.text(-1.15, 0.43, "入射光", color=DARK)
    ax.text(-0.20, -0.05, "反射光", color=DARK)
    ax.text(0.04, 0.42, "反射后先与邻镜相交", color=DARK, ha="center")
    ax.text(0.92, 1.68, "接收器", ha="center")
    ax.set_title("遮挡：反射光到达接收器之前损失")

    for idx, ax in enumerate(axes):
        panel_label(ax, f"（{'ab'[idx]}）")
        ax.set_xlim(-1.35, 1.25)
        ax.set_ylim(-0.8, 1.85)
        ax.set_aspect("equal")
        ax.axis("off")
    save_figure(fig, "q1_shadow_blocking_cn")


def figure_suncone_receiver() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), constrained_layout=True)
    ax = axes[0]
    disk_center = np.array([-0.92, 0.68])
    disk_radius = 0.27
    vertex = np.array([1.02, 0.39])
    ax.add_patch(Circle(tuple(disk_center), 0.27, facecolor=SCHEMATIC_LIGHT,
                        edgecolor=DARK, lw=1.4))
    source_points = [disk_center + np.array([0.0, offset])
                     for offset in np.linspace(-disk_radius, disk_radius, 7)]
    for source_point in source_points:
        ax.add_patch(FancyArrowPatch(tuple(source_point), tuple(vertex),
                                    arrowstyle="-|>", mutation_scale=8,
                                    color=SCHEMATIC_GRAY, lw=0.9, alpha=0.82))
    upper_edge = disk_center + np.array([0.0, disk_radius])
    ax.plot([disk_center[0], vertex[0]], [disk_center[1], vertex[1]],
            color=DARK, lw=1.9)
    ax.plot([upper_edge[0], vertex[0]], [upper_edge[1], vertex[1]],
            color=SCHEMATIC_BLUE, lw=1.4)
    center_angle = math.atan2(*(disk_center - vertex)[::-1])
    edge_angle = math.atan2(*(upper_edge - vertex)[::-1])
    ax.add_patch(Arc(tuple(vertex), 0.62, 0.62,
                     theta1=math.degrees(edge_angle),
                     theta2=math.degrees(center_angle), color=SCHEMATIC_BLUE, lw=1.5))
    label_angle = (center_angle + edge_angle) / 2
    label_pos = vertex + 0.42 * np.array([math.cos(label_angle), math.sin(label_angle)])
    ax.text(label_pos[0], label_pos[1] + 0.04, "$\\theta_\\odot$", color=DARK,
            ha="center", va="bottom")
    ax.scatter(*vertex, s=24, facecolor="white", edgecolor=DARK, lw=1.2, zorder=5)
    ax.text(-0.92, 1.10, "太阳视圆盘", ha="center")
    ax.text(0.18, -0.30,
            "视半径：$\\theta_\\odot=4.65$ mrad $\\approx0.266^\\circ$\n"
            "视直径：$2\\theta_\\odot\\approx0.533^\\circ$",
            ha="center", va="center", color=DARK, fontsize=8.3)
    ax.text(0.64, 0.94, "圆盘边缘方向", color=DARK, ha="center", fontsize=8)
    ax.text(0.02, 0.44, "圆盘中心方向 $\\boldsymbol{s}$", color=DARK,
            ha="center", fontsize=8)
    ax.text(1.02, 0.20, "观测点（定日镜）", color=DARK, ha="right", fontsize=7.6)
    ax.text(0.98, -0.58, "角度为示意放大，非实际比例", ha="right",
            color=DARK, fontsize=7.6)
    ax.set_title("太阳视圆盘对应的入射方向集合")

    ax = axes[1]
    center = np.array([-0.72, -0.52])
    receiver_x = 0.90
    receiver_bottom, receiver_top = -0.10, 1.02
    target = np.array([receiver_x, 0.48])
    central_out = target - center
    central_out /= np.linalg.norm(central_out)
    central_in = np.array([0.52, -0.854])
    central_in /= np.linalg.norm(central_in)
    normal = central_in - central_out
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    mirror = np.vstack((center - 0.40 * tangent, center + 0.40 * tangent))
    ax.plot(mirror[:, 0], mirror[:, 1], color=DARK, lw=1.8, marker="o",
            markersize=3.2, markerfacecolor="white")
    ax.add_patch(Rectangle((receiver_x, receiver_bottom), 0.27,
                           receiver_top - receiver_bottom, facecolor=SCHEMATIC_LIGHT,
                           edgecolor=DARK, lw=1.5))
    ax.text(0.98, 1.18, "有限圆柱侧面\n$72\\leq z\\leq80$ m", ha="center", color=DARK)
    offsets = np.linspace(-0.30, 0.30, 7)
    perturbations = np.linspace(-0.18, 0.18, 7)
    hit_count = 0
    for offset, delta in zip(offsets, perturbations):
        point = center + offset * tangent
        rotation = np.array([[math.cos(delta), -math.sin(delta)],
                             [math.sin(delta), math.cos(delta)]])
        incoming = rotation @ central_in
        reflected = incoming - 2 * np.dot(incoming, normal) * normal
        assert abs(np.linalg.norm(reflected) - 1.0) < 1e-12
        source = point - 0.48 * incoming
        ax.add_patch(FancyArrowPatch(tuple(source), tuple(point), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color=SCHEMATIC_GRAY, alpha=0.78))
        lam = (receiver_x - point[0]) / reflected[0]
        y_hit = point[1] + lam * reflected[1]
        hit = lam > 0 and receiver_bottom <= y_hit <= receiver_top
        hit_count += int(hit)
        endpoint = point + (lam if hit else min(max(lam, 0.9), 1.35)) * reflected
        color = SCHEMATIC_BLUE if hit else DARK
        ax.add_patch(FancyArrowPatch(tuple(point), tuple(endpoint), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.05, color=color,
                                     linestyle="-" if hit else "--", alpha=0.88))
    ax.text(0.12, 0.79, f"命中：{hit_count}条", color=DARK)
    ax.text(0.18, -0.43, f"未命中：{len(offsets)-hit_count}条", color=DARK)
    ax.text(-0.73, -0.88, "有限镜面采样点", ha="center")
    ax.set_title("有限接收器产生截断损失")
    for ax in axes:
        ax.set_xlim(-1.35, 1.65)
        ax.set_ylim(-0.95, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
    save_axis_figure(fig, axes[0], "q1_sundisk_angle_cn")
    save_axis_figure(fig, axes[1], "q1_receiver_truncation_cn")
    plt.close(fig)


def read_state_rows(month: int, solar_time: float) -> pd.DataFrame:
    path = ROUND_DIR / "tables" / "q1_main_per_heliostat_state.csv.gz"
    chunks = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for chunk in pd.read_csv(stream, chunksize=20000):
            selected = chunk[(chunk["month"] == month) & (chunk["solar_time"] == solar_time)]
            if not selected.empty:
                chunks.append(selected)
    return pd.concat(chunks, ignore_index=True)


def figure_field_spatial() -> None:
    positions = pd.read_csv(ROOT / "workspace" / "data_clean" / "Q1" / "heliostat_positions.csv")
    state_path = ROUND_DIR / "tables" / "q1_main_per_heliostat_state.csv.gz"
    annual = (
        pd.read_csv(state_path, usecols=["heliostat_id", "eta_total"])
        .groupby("heliostat_id", as_index=False)["eta_total"]
        .mean()
    )
    annual = positions.merge(annual, on="heliostat_id")
    cases = [(1, 9.0, "1月21日 9:00"), (3, 12.0, "3月21日 12:00"), (6, 12.0, "6月21日 12:00")]
    datasets = []
    for month, solar_time, label in cases:
        state = read_state_rows(month, solar_time)
        datasets.append((positions.merge(state, on="heliostat_id"), label))
    all_values = np.concatenate(
        [annual["eta_total"].to_numpy()]
        + [frame["eta_total"].to_numpy() for frame, _ in datasets]
    )
    norm = colors.Normalize(vmin=np.quantile(all_values, 0.02), vmax=np.quantile(all_values, 0.98))
    fig = plt.figure(figsize=(7.1, 6.45), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=(1.0, 1.0))
    axes = [
        fig.add_subplot(grid[0, :3]),
        fig.add_subplot(grid[0, 3:]),
        fig.add_subplot(grid[1, :2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:]),
    ]
    ax = axes[0]
    ax.scatter(positions["x_m"], positions["y_m"], s=4, color=BASELINE, alpha=0.75)
    ax.scatter([0], [0], s=50, marker="^", color=RED, label="吸收塔")
    ax.add_patch(Circle((0, 0), 100, fill=False, color=RED, ls="--", lw=1, label="塔周禁布区边界"))
    ax.set_title("镜场平面布置（1745面）")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        framealpha=0.96,
        fontsize=7.2,
    )
    efficiency_panels = [(annual, "60个规定状态平均", "年均综合光学效率")] + [
        (frame, label, "综合光学效率") for frame, label in datasets
    ]
    scatters = []
    for idx, (frame, label, metric) in enumerate(efficiency_panels, start=1):
        ax = axes[idx]
        scatter = ax.scatter(frame["x_m"], frame["y_m"], c=frame["eta_total"], s=5, cmap="viridis", norm=norm)
        scatters.append(scatter)
        ax.scatter([0], [0], s=34, marker="^", color=RED)
        ax.set_title(f"{label}\n{metric}")
    for idx, ax in enumerate(axes):
        panel_label(ax, f"（{'abcde'[idx]}）")
        ax.set_aspect("equal")
        ax.set_xlabel("东向坐标 x（m）")
        ax.set_ylabel("北向坐标 y（m）")
        ax.grid(color=GRID, lw=0.4, alpha=0.5)
    cbar = fig.colorbar(scatters[-1], ax=axes[1:], shrink=0.88, pad=0.018)
    cbar.set_label("综合光学效率")
    save_figure(fig, "q1_field_spatial_cn")


def figure_monthly_results() -> None:
    data = pd.read_csv(ROUND_DIR / "tables" / "q1_main_monthly.csv")
    month = data["month"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), constrained_layout=True)
    ax = axes[0]
    series = [
        ("eta_cos", "余弦效率", SCI_BLUE, "o"),
        ("eta_sb", "阴影遮挡效率", SCI_VERMILION, "s"),
        ("eta_trunc", "截断效率", SCI_GREEN, "^"),
        ("eta_total", "综合光学效率", SCI_GRAPHITE, "D"),
    ]
    for column, label, color, marker in series:
        ax.plot(month, data[column], color=color, marker=marker, markersize=3,
                markeredgewidth=0.55, lw=1.3, label=label)
    ax.set_xticks(month)
    ax.set_xlabel("月份")
    ax.set_ylabel("月平均效率")
    ax.set_ylim(0.48, 0.97)
    ax.legend(frameon=False, ncol=2, loc="lower center")
    ax.set_title("月平均光学效率分解")
    clean_axis(ax)
    ax = axes[1]
    ax.plot(month, data["unit_area_power_kw_m2"], color=SCI_BLUE, marker="o",
            markersize=3.2, markeredgewidth=0.55, lw=1.4)
    min_idx = int(data["unit_area_power_kw_m2"].idxmin())
    max_idx = int(data["unit_area_power_kw_m2"].idxmax())
    for idx, align in [(min_idx, "left"), (max_idx, "center")]:
        row = data.loc[idx]
        ax.annotate(
            f"{int(row['month'])}月：{row['unit_area_power_kw_m2']:.4f}",
            (row["month"], row["unit_area_power_kw_m2"]),
            xytext=(4, 10),
            textcoords="offset points",
            ha=align,
            color=DARK,
        )
    ax.set_xticks(month)
    ax.set_xlabel("月份")
    ax.set_ylabel("单位面积输出热功率（kW/m²）")
    ax.set_ylim(0.40, 0.68)
    ax.set_title("单位镜面面积月平均输出热功率")
    clean_axis(ax)
    panel_label(axes[0], "（a）")
    panel_label(axes[1], "（b）")
    save_figure(fig, "q1_monthly_results_cn")


def figure_convergence_robustness() -> None:
    with (ROUND_DIR / "metrics" / "q1_convergence.json").open(encoding="utf-8") as stream:
        convergence = json.load(stream)
    with (ROUND_DIR / "metrics" / "q1_sensitivity.json").open(encoding="utf-8") as stream:
        sensitivity = json.load(stream)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5), constrained_layout=True)
    ax = axes[0, 0]
    resolutions = ["32×16", "64×32"]
    powers = [convergence["coarse"]["annual"]["field_power_mw"], convergence["fine"]["annual"]["field_power_mw"]]
    changes = [(powers[0] / powers[1] - 1) * 100, 0.0]
    ax.plot(resolutions, changes, marker="o", markersize=3.2,
            color=SCI_BLUE, lw=1.35)
    ax.axhline(0.3, color=SCI_VERMILION, ls="--", lw=0.8, label="预设阈值 0.3%")
    for x, value in zip(resolutions, changes):
        ax.text(x, value + 0.018, f"{value:.4f}%", ha="center", fontsize=8)
    ax.set_ylabel("相对正式分辨率的功率变化")
    ax.set_title("嵌套分辨率收敛")
    ax.set_ylim(-0.04, 0.34)
    ax.legend(frameon=False)
    clean_axis(ax)

    ax = axes[0, 1]
    seeds = [2023, 2024, 2025, 2026]
    seed_power = convergence["fine_seed_dispersion"]["annual_field_power_mw_values"]
    seed_delta = (np.asarray(seed_power) / np.mean(seed_power) - 1) * 100
    ax.plot(seeds, seed_delta, marker="s", markersize=3.1,
            color=SCI_GREEN, lw=1.25)
    ax.axhline(0, color=SCI_GRAPHITE, ls="--", lw=0.8, label="四种子均值")
    ax.set_xticks(seeds)
    ax.set_xlabel("固定扰动种子")
    ax.set_ylabel("相对四种子均值的功率变化")
    ax.set_title("随机化低差异采样离散度")
    ax.legend(frameon=False)
    ax.set_ylim(-0.025, 0.025)
    clean_axis(ax)

    ax = axes[1, 0]
    angle = sensitivity["pillbox_half_angle"]
    x = [-5, 0, 5]
    y = [angle["minus_5_field_power_mw"], angle["core_field_power_mw"], angle["plus_5_field_power_mw"]]
    y_delta = [(value / y[1] - 1) * 100 for value in y]
    ax.plot(x, y_delta, marker="o", markersize=3.2,
            color=SCI_GREEN, lw=1.35)
    ax.scatter([0], [0], s=25, color=SCI_BLUE, zorder=4, label="批准口径")
    ax.axhline(1.0, color=SCI_VERMILION, ls="--", lw=0.8)
    ax.axhline(-1.0, color=SCI_VERMILION, ls="--", lw=0.8, label="返回决策阈值 ±1%")
    ax.set_xticks(x, ["−5%", "基准", "+5%"])
    ax.set_xlabel("太阳盘半角扰动")
    ax.set_ylabel("相对基准功率变化")
    ax.set_title("太阳盘半角敏感性")
    ax.legend(frameon=False)
    clean_axis(ax)

    ax = axes[1, 1]
    weighting = sensitivity["month_weighting"]
    weight_delta = [0, (weighting["month_day_weighted_field_power_mw"] / weighting["equal_month_field_power_mw"] - 1) * 100]
    bars = ax.bar(
        ["十二个月等权", "按各月天数加权"],
        weight_delta,
        color=[SCI_BLUE, SCI_GRAPHITE],
        width=0.58,
    )
    ax.bar_label(bars, labels=[f"{value:.4f}%" for value in weight_delta], padding=3, fontsize=8)
    ax.axhline(1.0, color=SCI_VERMILION, ls="--", lw=0.8, label="返回决策阈值 1%")
    ax.set_ylabel("相对等月权重的功率变化")
    ax.set_title("月份权重敏感性")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(frameon=False)
    clean_axis(ax)
    for idx, ax in enumerate(axes.flat):
        panel_label(ax, f"（{'abcd'[idx]}）")
    save_figure(fig, "q1_convergence_robustness_cn")


def main() -> None:
    configure_style()
    figure_solar_geometry()
    figure_solar_states()
    figure_reflection_geometry()
    figure_single_mirror_tracking()
    figure_shadow_blocking()
    figure_suncone_receiver()
    figure_field_spatial()
    figure_monthly_results()
    figure_convergence_robustness()
    outputs = sorted(path.name for path in PAPER_FIGURES.glob("q1_*_cn.*"))
    print(f"已生成 {len(outputs)} 个图形文件：")
    for name in outputs:
        print(name)


if __name__ == "__main__":
    main()
