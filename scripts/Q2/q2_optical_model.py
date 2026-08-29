#!/usr/bin/env python3
"""Q2 approved M2 design and B2 baseline with a shared coupled ray tracer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MATPLOTLIB_CONFIG = ROOT / "workspace" / "analysis" / ".mplconfig"
MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG))

import numpy as np
from scipy.spatial import cKDTree
import tensorflow as tf


sys.path.insert(0, str(ROOT / "scripts" / "Q1"))
import q1_optical_model as q1  # noqa: E402


FIELD_RADIUS = 350.0
EXCLUSION_RADIUS = 100.0
TARGET_POWER_MW = 60.0
SEARCH_REPAIR_TARGET_MW = 61.2
SEEDS = q1.SEEDS


@dataclass(frozen=True)
class Design:
    design_id: str
    method_id: str
    role: str
    tower_x_m: float
    tower_y_m: float
    width_m: float
    height_m: float
    installation_height_m: float
    lattice_angle_deg: float = 0.0
    spacing_factor: float = 1.0
    phase_x_fraction: float = 0.0
    phase_y_fraction: float = 0.0


@dataclass
class TraceResult:
    eta_sb: np.ndarray
    eta_trunc: np.ndarray
    joint_received: np.ndarray


def output_dirs(profile: str) -> tuple[Path, Path, Path, Path]:
    if profile == "formal":
        root = ROOT / "results" / "Q2" / "experiments" / "round1"
    else:
        root = ROOT / "workspace" / "analysis" / "Q2" / "smoke_run"
    table = root / "tables"
    metric = root / "metrics"
    figure = root / "figures"
    for path in (table, metric, figure):
        path.mkdir(parents=True, exist_ok=True)
    return root, table, metric, figure


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hexagonal_layout(design: Design) -> np.ndarray:
    tower = np.array([design.tower_x_m, design.tower_y_m], dtype=np.float64)
    spacing = (design.width_m + 5.0) * design.spacing_factor
    row_spacing = math.sqrt(3.0) * spacing / 2.0
    limit = FIELD_RADIUS + np.linalg.norm(tower) + spacing
    rows = np.arange(-limit, limit + row_spacing, row_spacing)
    angle = math.radians(design.lattice_angle_deg)
    rotation = np.array(
        [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    result: list[np.ndarray] = []
    for row_index, base_y in enumerate(rows):
        y_value = base_y + design.phase_y_fraction * row_spacing
        columns = np.arange(-limit, limit + spacing, spacing)
        columns += (row_index % 2) * spacing / 2.0 + design.phase_x_fraction * spacing
        relative = np.column_stack((columns, np.full_like(columns, y_value))) @ rotation
        absolute = relative + tower
        legal = (
            (np.linalg.norm(absolute, axis=1) <= FIELD_RADIUS + 1e-9)
            & (np.linalg.norm(relative, axis=1) >= EXCLUSION_RADIUS - 1e-9)
        )
        result.append(absolute[legal])
    return np.vstack(result)


def geometry_checks(points_xy: np.ndarray, design: Design) -> dict[str, Any]:
    tower = np.array([design.tower_x_m, design.tower_y_m])
    nearest = cKDTree(points_xy).query(points_xy, k=2)[0][:, 1]
    required_spacing = design.width_m + 5.0
    checks = {
        "mirror_count": len(points_xy),
        "inside_field_count": int(
            np.sum(np.linalg.norm(points_xy, axis=1) <= FIELD_RADIUS + 1e-9)
        ),
        "outside_exclusion_count": int(
            np.sum(np.linalg.norm(points_xy - tower, axis=1) >= EXCLUSION_RADIUS - 1e-9)
        ),
        "minimum_spacing_m": float(np.min(nearest)),
        "required_spacing_m": required_spacing,
        "ground_clearance_m": design.installation_height_m - design.height_m / 2.0,
        "tower_inside_field": bool(np.linalg.norm(tower) <= FIELD_RADIUS + 1e-9),
    }
    checks["status"] = "PASS" if (
        checks["inside_field_count"] == len(points_xy)
        and checks["outside_exclusion_count"] == len(points_xy)
        and checks["minimum_spacing_m"] + 1e-8 >= required_spacing
        and checks["ground_clearance_m"] > 0.0
        and checks["tower_inside_field"]
        and 2.0 <= design.height_m <= design.width_m <= 8.0
        and 2.0 <= design.installation_height_m <= 6.0
    ) else "FAIL"
    return checks


def optical_centers(points_xy: np.ndarray, design: Design) -> np.ndarray:
    tower = np.array([design.tower_x_m, design.tower_y_m])
    relative = points_xy - tower
    return np.column_stack(
        (relative, np.full(len(points_xy), design.installation_height_m, dtype=np.float64))
    )


def surface_uv_all(
    count: int,
    seed: int,
    heliostat_ids: np.ndarray,
    width_m: float,
    height_m: float,
) -> np.ndarray:
    samples = np.empty((len(heliostat_ids), count, 2), dtype=np.float64)
    scale = np.array([width_m, height_m], dtype=np.float64)
    for row, heliostat_id in enumerate(heliostat_ids):
        samples[row] = (q1.sobol_unit(count, 2, seed + int(heliostat_id) * 17) - 0.5) * scale
    return samples


@tf.function(reduce_retracing=True)
def trace_dynamic_tensors(
    points: np.ndarray,
    sun_directions: np.ndarray,
    target_normals: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_normals: np.ndarray,
    candidate_width_axes: np.ndarray,
    candidate_height_axes: np.ndarray,
    candidate_valid: np.ndarray,
    half_width: float,
    half_height: float,
) -> tuple[Any, Any]:
    dtype = tf.float32
    p = tf.cast(points, dtype)
    sun = tf.cast(sun_directions, dtype)
    target_n = tf.cast(target_normals, dtype)
    cc = tf.cast(candidate_centers, dtype)
    cn = tf.cast(candidate_normals, dtype)
    cw = tf.cast(candidate_width_axes, dtype)
    ch = tf.cast(candidate_height_axes, dtype)
    candidate_mask = tf.cast(candidate_valid, tf.bool)
    eps = tf.constant(1e-6, dtype=dtype)
    edge_w = tf.cast(half_width + 1e-5, dtype)
    edge_h = tf.cast(half_height + 1e-5, dtype)

    relative = p[:, :, None, :] - cc[:, None, :, :]
    numerator = -tf.einsum("bski,bki->bsk", relative, cn)
    denominator_in = tf.einsum("ui,bki->buk", sun, cn)
    safe_in = tf.where(tf.abs(denominator_in) > eps, denominator_in, tf.ones_like(denominator_in))
    lambda_in = numerator[:, None, :, :] / safe_in[:, :, None, :]
    base_w = tf.einsum("bski,bki->bsk", relative, cw)
    base_h = tf.einsum("bski,bki->bsk", relative, ch)
    direction_w_in = tf.einsum("ui,bki->buk", sun, cw)
    direction_h_in = tf.einsum("ui,bki->buk", sun, ch)
    hit_w_in = base_w[:, None, :, :] + lambda_in * direction_w_in[:, :, None, :]
    hit_h_in = base_h[:, None, :, :] + lambda_in * direction_h_in[:, :, None, :]
    blocked_in = tf.reduce_any(
        candidate_mask[:, None, None, :]
        & (tf.abs(denominator_in)[:, :, None, :] > eps)
        & (lambda_in > eps)
        & (tf.abs(hit_w_in) <= edge_w)
        & (tf.abs(hit_h_in) <= edge_h),
        axis=-1,
    )

    # Optical coordinates are tower-relative, so the tower-body cylinder is
    # centered on the z axis for every candidate tower location.
    tower_a = sun[:, 0] ** 2 + sun[:, 1] ** 2
    tower_b = 2.0 * (
        p[:, None, :, 0] * sun[None, :, None, 0]
        + p[:, None, :, 1] * sun[None, :, None, 1]
    )
    tower_c = (
        p[:, None, :, 0] ** 2
        + p[:, None, :, 1] ** 2
        - q1.TOWER_RADIUS**2
    )
    tower_disc = tower_b**2 - 4.0 * tower_a[None, :, None] * tower_c
    tower_root_term = tf.sqrt(tf.maximum(tower_disc, 0.0))
    tower_denominator = 2.0 * tower_a[None, :, None]
    tower_root_a = (-tower_b - tower_root_term) / tower_denominator
    tower_root_b = (-tower_b + tower_root_term) / tower_denominator
    tower_roots = tf.stack((tower_root_a, tower_root_b), axis=-1)
    tower_z = (
        p[:, None, :, None, 2]
        + tower_roots * sun[None, :, None, None, 2]
    )
    tower_hit = tf.reduce_any(
        (tower_disc[..., None] >= 0.0)
        & (tower_roots > eps)
        & (tower_z >= q1.TOWER_Z_MIN)
        & (tower_z < q1.TOWER_Z_MAX),
        axis=-1,
    )
    blocked_in = blocked_in | tower_hit

    sun_dot_normal = tf.einsum("ui,bi->bu", sun, target_n)
    outgoing = -sun[None, :, :] + 2.0 * sun_dot_normal[:, :, None] * target_n[:, None, :]
    outgoing /= tf.linalg.norm(outgoing, axis=-1, keepdims=True)
    dx, dy = outgoing[:, :, 0], outgoing[:, :, 1]
    coefficient_a = dx**2 + dy**2
    coefficient_b = 2.0 * (
        p[:, None, :, 0] * dx[:, :, None] + p[:, None, :, 1] * dy[:, :, None]
    )
    coefficient_c = p[:, None, :, 0] ** 2 + p[:, None, :, 1] ** 2 - q1.RECEIVER_RADIUS**2
    discriminant = coefficient_b**2 - 4.0 * coefficient_a[:, :, None] * coefficient_c
    root_term = tf.sqrt(tf.maximum(discriminant, 0.0))
    root_a = (-coefficient_b - root_term) / (2.0 * coefficient_a[:, :, None])
    root_b = (-coefficient_b + root_term) / (2.0 * coefficient_a[:, :, None])
    infinity = tf.constant(np.inf, dtype=dtype)
    root_a = tf.where((discriminant >= 0.0) & (root_a > eps), root_a, infinity)
    root_b = tf.where((discriminant >= 0.0) & (root_b > eps), root_b, infinity)
    receiver_lambda = tf.minimum(root_a, root_b)
    receiver_z = p[:, None, :, 2] + receiver_lambda * outgoing[:, :, None, 2]
    receiver_hit = (
        tf.math.is_finite(receiver_lambda)
        & (receiver_z >= q1.RECEIVER_Z_MIN)
        & (receiver_z <= q1.RECEIVER_Z_MAX)
    )
    plane_lambda = tf.where(
        outgoing[:, :, None, 2] > eps,
        (q1.RECEIVER_Z_MAX - p[:, None, :, 2]) / outgoing[:, :, None, 2],
        infinity,
    )
    max_lambda = tf.where(receiver_hit, receiver_lambda, plane_lambda)

    denominator_out = tf.einsum("bui,bki->buk", outgoing, cn)
    safe_out = tf.where(
        tf.abs(denominator_out) > eps, denominator_out, tf.ones_like(denominator_out)
    )
    lambda_out = numerator[:, None, :, :] / safe_out[:, :, None, :]
    direction_w_out = tf.einsum("bui,bki->buk", outgoing, cw)
    direction_h_out = tf.einsum("bui,bki->buk", outgoing, ch)
    hit_w_out = base_w[:, None, :, :] + lambda_out * direction_w_out[:, :, None, :]
    hit_h_out = base_h[:, None, :, :] + lambda_out * direction_h_out[:, :, None, :]
    blocked_out = tf.reduce_any(
        candidate_mask[:, None, None, :]
        & (tf.abs(denominator_out)[:, :, None, :] > eps)
        & (lambda_out > eps)
        & (lambda_out < max_lambda[:, :, :, None] - eps)
        & (tf.abs(hit_w_out) <= edge_w)
        & (tf.abs(hit_h_out) <= edge_h),
        axis=-1,
    )
    visible = ~blocked_in & ~blocked_out
    received = visible & receiver_hit
    return (
        tf.reduce_sum(tf.cast(visible, tf.int64), axis=(1, 2)),
        tf.reduce_sum(tf.cast(received, tf.int64), axis=(1, 2)),
    )


def trace_dynamic_batch(*args: Any) -> tuple[np.ndarray, np.ndarray]:
    return tuple(value.numpy() for value in trace_dynamic_tensors(*args))


def ray_clear_numpy(
    points: np.ndarray,
    direction: np.ndarray,
    candidate_indices: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    width_axes: np.ndarray,
    height_axes: np.ndarray,
    half_width: float,
    half_height: float,
    max_lambda: np.ndarray | None = None,
) -> np.ndarray:
    blocked = np.zeros(len(points), dtype=bool)
    for index in candidate_indices:
        denominator = float(np.dot(direction, normals[index]))
        if abs(denominator) <= q1.EPS64:
            continue
        lam = (centers[index] - points) @ normals[index] / denominator
        eligible = lam > q1.EPS64
        if max_lambda is not None:
            eligible &= lam < max_lambda - q1.EPS64
        if not np.any(eligible):
            continue
        intersection = points[eligible] + lam[eligible, None] * direction
        relative = intersection - centers[index]
        inside = (
            np.abs(relative @ width_axes[index]) <= half_width + 1e-10
        ) & (
            np.abs(relative @ height_axes[index]) <= half_height + 1e-10
        )
        blocked[np.flatnonzero(eligible)[inside]] = True
    return ~blocked


def trace_numpy_target(
    points: np.ndarray,
    sun_directions: np.ndarray,
    target_normal: np.ndarray,
    candidate_indices: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    width_axes: np.ndarray,
    height_axes: np.ndarray,
    half_width: float,
    half_height: float,
) -> tuple[int, int]:
    visible_count = 0
    received_count = 0
    for sun_direction in sun_directions:
        incoming_clear = ~q1.tower_blocks_numpy(points, sun_direction) & ray_clear_numpy(
            points, sun_direction, candidate_indices, centers, normals,
            width_axes, height_axes, half_width, half_height,
        )
        outgoing = q1.reflect_directions(sun_direction[None, :], target_normal[None, :])[0]
        hit, receiver_lambda = q1.receiver_hits_numpy(points[None, :, :], outgoing[None, None, :])
        plane_lambda = np.full(len(points), np.inf)
        if outgoing[2] > q1.EPS64:
            plane_lambda = (q1.RECEIVER_Z_MAX - points[:, 2]) / outgoing[2]
        maximum = np.where(hit[0, 0], receiver_lambda[0, 0], plane_lambda)
        outgoing_clear = ray_clear_numpy(
            points, outgoing, candidate_indices, centers, normals,
            width_axes, height_axes, half_width, half_height, maximum,
        )
        visible = incoming_clear & outgoing_clear
        visible_count += int(np.sum(visible))
        received_count += int(np.sum(visible & hit[0, 0]))
    return visible_count, received_count


def trace_full_field(
    heliostat_ids: np.ndarray,
    centers: np.ndarray,
    design: Design,
    states: list[q1.SolarState],
    candidate_indices: np.ndarray,
    candidate_valid: np.ndarray,
    surface_count: int,
    sun_count: int,
    seed: int,
    batch_size: int,
    half_angle: float = q1.SUN_HALF_ANGLE,
) -> TraceResult:
    eta_sb = np.empty((len(states), len(centers)), dtype=np.float64)
    eta_trunc = np.empty_like(eta_sb)
    joint_received = np.empty_like(eta_sb)
    uv = surface_uv_all(surface_count, seed, heliostat_ids, design.width_m, design.height_m)
    total = surface_count * sun_count
    for state in states:
        sun = np.asarray(state.sun_vector, dtype=np.float64)
        _, normals, width_axes, height_axes = q1.mirror_frames(centers, sun)
        directions = q1.sun_disk_directions(
            sun, sun_count, half_angle, seed + 100_000 + state.state_index * 101
        )
        points = q1.points_from_uv(centers, width_axes, height_axes, uv)
        for start in range(0, len(centers), batch_size):
            stop = min(start + batch_size, len(centers))
            batch = np.arange(start, stop)
            candidates = candidate_indices[batch]
            visible, received = trace_dynamic_batch(
                points[batch], directions, normals[batch], centers[candidates],
                normals[candidates], width_axes[candidates], height_axes[candidates],
                candidate_valid[batch], design.width_m / 2.0, design.height_m / 2.0,
            )
            eta_sb[state.state_index, batch] = visible / total
            eta_trunc[state.state_index, batch] = np.divide(
                received, visible, out=np.zeros_like(received, dtype=np.float64), where=visible > 0
            )
            joint_received[state.state_index, batch] = received / total
    return TraceResult(eta_sb, eta_trunc, joint_received)


def deterministic_components(
    centers: np.ndarray, states: list[q1.SolarState]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta_cos = np.empty((len(states), len(centers)), dtype=np.float64)
    for state in states:
        sun = np.asarray(state.sun_vector, dtype=np.float64)
        _, normals, _, _ = q1.mirror_frames(centers, sun)
        eta_cos[state.state_index] = normals @ sun
    return (
        eta_cos,
        q1.atmospheric_efficiency(centers),
        np.asarray([state.dni_kw_m2 for state in states], dtype=np.float64),
    )


def complete_efficiencies(
    trace: TraceResult, eta_cos: np.ndarray, eta_at: np.ndarray
) -> dict[str, np.ndarray]:
    eta_total = trace.eta_sb * eta_cos * eta_at[None, :] * trace.eta_trunc * q1.REFLECTIVITY
    return {
        "eta_sb": trace.eta_sb,
        "eta_cos": eta_cos,
        "eta_at": np.broadcast_to(eta_at[None, :], eta_cos.shape),
        "eta_trunc": trace.eta_trunc,
        "eta_total": eta_total,
        "joint_received": trace.joint_received,
    }


def aggregate(
    components: dict[str, np.ndarray],
    states: list[q1.SolarState],
    dni: np.ndarray,
    area_m2: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    state_mean = {
        name: values.mean(axis=1) for name, values in components.items() if name.startswith("eta_")
    }
    unit_power = dni * state_mean["eta_total"]
    field_power_mw = unit_power * components["eta_total"].shape[1] * area_m2 / 1000.0
    monthly: list[dict[str, float]] = []
    for month in range(1, 13):
        indices = np.asarray([index for index, state in enumerate(states) if state.month == month])
        if len(indices) == 0:
            continue
        monthly.append({
            "month": month,
            "eta_total": float(np.mean(state_mean["eta_total"][indices])),
            "eta_cos": float(np.mean(state_mean["eta_cos"][indices])),
            "eta_sb": float(np.mean(state_mean["eta_sb"][indices])),
            "eta_trunc": float(np.mean(state_mean["eta_trunc"][indices])),
            "unit_area_power_kw_m2": float(np.mean(unit_power[indices])),
            "field_power_mw": float(np.mean(field_power_mw[indices])),
        })
    annual = {
        "eta_total": float(np.mean(state_mean["eta_total"])),
        "eta_cos": float(np.mean(state_mean["eta_cos"])),
        "eta_sb": float(np.mean(state_mean["eta_sb"])),
        "eta_trunc": float(np.mean(state_mean["eta_trunc"])),
        "field_power_mw": float(np.mean(field_power_mw)),
        "unit_area_power_kw_m2": float(np.mean(unit_power)),
    }
    return monthly, annual


def evaluate_resolution(
    points_xy: np.ndarray,
    design: Design,
    states: list[q1.SolarState],
    surface_count: int,
    sun_count: int,
    seeds: tuple[int, ...],
    batch_size: int,
    half_angle: float = q1.SUN_HALF_ANGLE,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    started = time.perf_counter()
    centers = optical_centers(points_xy, design)
    ids = np.arange(1, len(centers) + 1, dtype=np.int64)
    candidates, candidate_valid = q1.build_candidates(centers)
    eta_cos, eta_at, dni = deterministic_components(centers, states)
    traces: list[TraceResult] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        print(
            f"{design.method_id} {surface_count}x{sun_count} seed={seed} started",
            flush=True,
        )
        seed_started = time.perf_counter()
        trace = trace_full_field(
            ids, centers, design, states, candidates, candidate_valid,
            surface_count, sun_count, seed, batch_size, half_angle,
        )
        components = complete_efficiencies(trace, eta_cos, eta_at)
        _, annual = aggregate(components, states, dni, design.width_m * design.height_m)
        seed_summaries.append({
            "seed": seed,
            "surface_samples": surface_count,
            "sun_samples": sun_count,
            "annual": annual,
            "max_joint_factorization_residual": float(
                np.max(np.abs(trace.eta_sb * trace.eta_trunc - trace.joint_received))
            ),
            "runtime_seconds": time.perf_counter() - seed_started,
        })
        print(
            f"{design.method_id} {surface_count}x{sun_count} seed={seed} "
            f"power={annual['field_power_mw']:.6f} MW completed",
            flush=True,
        )
        traces.append(trace)
    eta_sb = np.mean(np.stack([trace.eta_sb for trace in traces]), axis=0)
    joint = np.mean(np.stack([trace.joint_received for trace in traces]), axis=0)
    eta_trunc = np.divide(joint, eta_sb, out=np.zeros_like(joint), where=eta_sb > 0)
    pooled = complete_efficiencies(TraceResult(eta_sb, eta_trunc, joint), eta_cos, eta_at)
    return pooled, seed_summaries, time.perf_counter() - started


def select_main_layout(
    design: Design,
    source_points: np.ndarray,
    states: list[q1.SolarState],
    surface_count: int,
    sun_count: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    components, seed_summaries, runtime = evaluate_resolution(
        source_points, design, states, surface_count, sun_count, (SEEDS[0],), batch_size
    )
    dni = np.asarray([state.dni_kw_m2 for state in states], dtype=np.float64)
    area = design.width_m * design.height_m
    mirror_mean_power_kw = np.mean(dni[:, None] * area * components["eta_total"], axis=0)
    ranking = np.argsort(mirror_mean_power_kw)[::-1]
    cumulative = np.cumsum(mirror_mean_power_kw[ranking])
    selected_count = min(
        len(ranking),
        int(np.searchsorted(cumulative, SEARCH_REPAIR_TARGET_MW * 1000.0) + 1),
    )
    selected_indices = np.sort(ranking[:selected_count])
    selected = source_points[selected_indices]
    return selected, {
        "schema_version": 1,
        "source_design": asdict(design),
        "source_mirror_count": len(source_points),
        "quick_surface_samples": surface_count,
        "quick_sun_samples": sun_count,
        "quick_seed": SEEDS[0],
        "search_repair_target_mw": SEARCH_REPAIR_TARGET_MW,
        "source_quick_power_mw": float(np.sum(mirror_mean_power_kw) / 1000.0),
        "selected_additive_power_before_re_evaluation_mw": float(
            np.sum(mirror_mean_power_kw[selected_indices]) / 1000.0
        ),
        "selected_mirror_count": len(selected),
        "deleted_mirror_count": len(source_points) - len(selected),
        "selection_rule": "按耦合快速层逐镜60状态年均功率降序，保留达到61.2 MW的最小前缀",
        "selected_geometry": geometry_checks(selected, design),
        "quick_seed_summary": seed_summaries,
        "runtime_seconds": runtime,
        "formal_re_evaluation_required": True,
    }


def convergence_metrics(
    method_id: str,
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    states: list[q1.SolarState],
    area: float,
    fine_seed_summaries: list[dict[str, Any]],
    coarse_samples: tuple[int, int],
    fine_samples: tuple[int, int],
) -> dict[str, Any]:
    dni = np.asarray([state.dni_kw_m2 for state in states], dtype=np.float64)
    coarse_monthly, coarse_annual = aggregate(coarse, states, dni, area)
    fine_monthly, fine_annual = aggregate(fine, states, dni, area)
    eta_delta = abs(fine_annual["eta_total"] - coarse_annual["eta_total"])
    power_delta = abs(fine_annual["field_power_mw"] - coarse_annual["field_power_mw"]) / fine_annual["field_power_mw"]
    monthly_delta = [
        abs(fine_row["unit_area_power_kw_m2"] - coarse_row["unit_area_power_kw_m2"])
        / fine_row["unit_area_power_kw_m2"]
        for coarse_row, fine_row in zip(coarse_monthly, fine_monthly)
    ]
    seed_powers = [row["annual"]["field_power_mw"] for row in fine_seed_summaries]
    passed = eta_delta <= 0.002 and power_delta <= 0.003 and max(monthly_delta) <= 0.005
    return {
        "method_id": method_id,
        "coarse": {"surface_samples": coarse_samples[0], "sun_samples": coarse_samples[1], "annual": coarse_annual},
        "fine": {"surface_samples": fine_samples[0], "sun_samples": fine_samples[1], "annual": fine_annual},
        "annual_eta_total_abs_delta": float(eta_delta),
        "annual_field_power_relative_delta": float(power_delta),
        "monthly_unit_power_relative_delta": [float(value) for value in monthly_delta],
        "max_monthly_unit_power_relative_delta": float(max(monthly_delta)),
        "fine_seed_field_power_mw": seed_powers,
        "fine_seed_field_power_mw_std": float(np.std(seed_powers)),
        "fine_seed_field_power_mw_range": float(np.ptp(seed_powers)),
        "thresholds": {
            "annual_eta_total_abs": 0.002,
            "annual_field_power_relative": 0.003,
            "monthly_unit_power_relative": 0.005,
        },
        "status": "PASS" if passed else "CONDITIONAL",
    }


def layout_rows(points: np.ndarray, design: Design) -> list[dict[str, Any]]:
    return [
        {
            "heliostat_id": index,
            "x_m": float(point[0]),
            "y_m": float(point[1]),
            "z_m": design.installation_height_m,
            "width_m": design.width_m,
            "height_m": design.height_m,
        }
        for index, point in enumerate(points, start=1)
    ]


def run_validation(
    main_design: Design,
    main_points: np.ndarray,
    baseline_design: Design,
    baseline_points: np.ndarray,
) -> dict[str, Any]:
    states = q1.build_solar_states()
    main_geometry = geometry_checks(main_points, main_design)
    baseline_geometry = geometry_checks(baseline_points, baseline_design)
    q1_ids, q1_centers = q1.load_centers()
    q1_candidates, q1_valid = q1.build_candidates(q1_centers)
    state = states[12]
    sun = np.asarray(state.sun_vector)
    _, normals, width_axes, height_axes = q1.mirror_frames(q1_centers, sun)
    uv = q1.surface_uv_all(8, SEEDS[0], q1_ids)
    points = q1.points_from_uv(q1_centers, width_axes, height_axes, uv)
    directions = q1.sun_disk_directions(sun, 4, q1.SUN_HALF_ANGLE, SEEDS[0] + 100_000)
    selected = np.array([0, len(q1_centers) // 2, len(q1_centers) - 1])
    candidates = q1_candidates[selected]
    dynamic_counts = trace_dynamic_batch(
        points[selected], directions, normals[selected], q1_centers[candidates],
        normals[candidates], width_axes[candidates], height_axes[candidates],
        q1_valid[selected], 3.0, 3.0,
    )
    q1_counts = q1._tensorflow_trace_batch(
        points[selected], directions, normals[selected], q1_centers[candidates],
        normals[candidates], width_axes[candidates], height_axes[candidates], q1_valid[selected],
    )[:2]

    centers = optical_centers(main_points, main_design)
    local_candidates, local_valid = q1.build_candidates(centers)
    _, local_normals, local_width_axes, local_height_axes = q1.mirror_frames(centers, sun)
    local_ids = np.arange(1, len(centers) + 1)
    local_uv = surface_uv_all(8, SEEDS[0], local_ids, main_design.width_m, main_design.height_m)
    local_points = q1.points_from_uv(centers, local_width_axes, local_height_axes, local_uv)
    local_selected = np.array([0, len(centers) // 2, len(centers) - 1])
    local_candidate_batch = local_candidates[local_selected]
    tf_visible, tf_received = trace_dynamic_batch(
        local_points[local_selected], directions, local_normals[local_selected],
        centers[local_candidate_batch], local_normals[local_candidate_batch],
        local_width_axes[local_candidate_batch], local_height_axes[local_candidate_batch],
        local_valid[local_selected], main_design.width_m / 2.0, main_design.height_m / 2.0,
    )
    numpy_counts = []
    for index in local_selected:
        active = local_candidates[index, local_valid[index]]
        numpy_counts.append(trace_numpy_target(
            local_points[index], directions, local_normals[index], active, centers,
            local_normals, local_width_axes, local_height_axes,
            main_design.width_m / 2.0, main_design.height_m / 2.0,
        ))
    numpy_counts_array = np.asarray(numpy_counts)
    tensorflow_counts_array = np.column_stack((tf_visible, tf_received))
    reflection_residual = 0.0
    center_hits: list[bool] = []
    for current in states:
        current_sun = np.asarray(current.sun_vector)
        target, current_normals, _, _ = q1.mirror_frames(centers, current_sun)
        outgoing = q1.reflect_directions(np.broadcast_to(current_sun, current_normals.shape), current_normals)
        reflection_residual = max(
            reflection_residual, float(np.max(np.linalg.norm(outgoing - target, axis=1)))
        )
        hits, _ = q1.receiver_hits_numpy(centers[:, None, :], outgoing[:, None, :])
        center_hits.extend(hits[:, 0, 0].tolist())
    validation = {
        "schema_version": 1,
        "solar_state_count": len(states),
        "main_geometry": main_geometry,
        "baseline_geometry": baseline_geometry,
        "dynamic_6m_regression": {
            "dynamic_visible": dynamic_counts[0].tolist(),
            "q1_visible": q1_counts[0].tolist(),
            "dynamic_received": dynamic_counts[1].tolist(),
            "q1_received": q1_counts[1].tolist(),
            "exact_match": bool(
                np.array_equal(dynamic_counts[0], q1_counts[0])
                and np.array_equal(dynamic_counts[1], q1_counts[1])
            ),
        },
        "dynamic_7m_tensorflow_vs_numpy": {
            "tensorflow": tensorflow_counts_array.tolist(),
            "numpy": numpy_counts_array.tolist(),
            "exact_match": bool(np.array_equal(tensorflow_counts_array, numpy_counts_array)),
        },
        "max_sun_vector_norm_error": float(
            max(abs(np.linalg.norm(state.sun_vector) - 1.0) for state in states)
        ),
        "max_center_reflection_residual": reflection_residual,
        "center_ray_receiver_hit_rate": float(np.mean(center_hits)),
        "tower_body_model": {
            "radius_m": q1.TOWER_RADIUS,
            "z_interval_m": [q1.TOWER_Z_MIN, q1.TOWER_Z_MAX],
            "receiver_excluded_from_incoming_shadow": True,
        },
    }
    passed = (
        main_geometry["status"] == "PASS"
        and baseline_geometry["status"] == "PASS"
        and validation["dynamic_6m_regression"]["exact_match"]
        and validation["dynamic_7m_tensorflow_vs_numpy"]["exact_match"]
        and validation["max_sun_vector_norm_error"] <= 1e-12
        and reflection_residual <= 1e-12
        and validation["center_ray_receiver_hit_rate"] == 1.0
    )
    validation["status"] = "PASS" if passed else "FAIL"
    return validation


def make_figures(
    main_points: np.ndarray,
    baseline_points: np.ndarray,
    main_design: Design,
    baseline_design: Design,
    main_monthly: list[dict[str, float]],
    baseline_monthly: list[dict[str, float]],
    figure_dir: Path,
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["AR PL SungtiL GB", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharex=True, sharey=True)
    for axis, points, design, title in (
        (axes[0], baseline_points, baseline_design, "B2 基线镜场"),
        (axes[1], main_points, main_design, "M2 主方法镜场"),
    ):
        axis.scatter(points[:, 0], points[:, 1], s=1.2, alpha=0.75)
        axis.scatter([design.tower_x_m], [design.tower_y_m], marker="*", s=90, color="#C43C39")
        axis.set_title(title)
        axis.set_xlabel("东西坐标 / m")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("南北坐标 / m")
    fig.tight_layout()
    layout_path = figure_dir / "q2_layout_comparison_diagnostic.png"
    fig.savefig(layout_path, dpi=180)
    plt.close(fig)

    months = np.arange(1, 13)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    axes[0].plot(months, [row["field_power_mw"] for row in main_monthly], "o-", label="M2")
    axes[0].plot(months, [row["field_power_mw"] for row in baseline_monthly], "s--", label="B2")
    axes[0].set_ylabel("月平均输出热功率 / MW")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(months, [row["unit_area_power_kw_m2"] for row in main_monthly], "o-", label="M2")
    axes[1].plot(months, [row["unit_area_power_kw_m2"] for row in baseline_monthly], "s--", label="B2")
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("单位面积功率 / (kW/m^2)")
    axes[1].set_xticks(months)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    monthly_path = figure_dir / "q2_monthly_comparison_diagnostic.png"
    fig.savefig(monthly_path, dpi=180)
    plt.close(fig)
    return [str(layout_path.relative_to(ROOT)), str(monthly_path.relative_to(ROOT))]


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    overall_started = time.perf_counter()
    round_dir, table_dir, metric_dir, figure_dir = output_dirs(args.profile)
    states = q1.build_solar_states()
    baseline_design = Design(
        "B2_fixed_7m_centered_hex", "B2", "usable_baseline", 0.0, 0.0, 7.0, 7.0, 4.0
    )
    main_design = Design(
        "M2_south40_spacing101", "M2", "main_candidate", 0.0, -40.0,
        7.0, 7.0, 4.0, spacing_factor=1.01,
    )
    baseline_points = hexagonal_layout(baseline_design)
    source_main_points = hexagonal_layout(main_design)
    validation = run_validation(main_design, source_main_points, baseline_design, baseline_points)
    write_json(metric_dir / "q2_validation.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError("Q2 validation failed")

    if args.profile == "formal":
        quick_samples = (8, 4)
        coarse_samples = (16, 8)
        fine_samples = (32, 16)
        formal_seeds = SEEDS
    else:
        quick_samples = (4, 2)
        coarse_samples = (4, 2)
        fine_samples = (8, 4)
        formal_seeds = (SEEDS[0],)
    main_points, design_selection = select_main_layout(
        main_design, source_main_points, states, *quick_samples, args.batch_size
    )
    write_json(metric_dir / "q2_design_selection.json", design_selection)

    method_results: dict[str, dict[str, Any]] = {}
    for design, points in ((main_design, main_points), (baseline_design, baseline_points)):
        coarse, coarse_seeds, coarse_runtime = evaluate_resolution(
            points, design, states, *coarse_samples, formal_seeds, args.batch_size
        )
        fine, fine_seeds, fine_runtime = evaluate_resolution(
            points, design, states, *fine_samples, formal_seeds, args.batch_size
        )
        dni = np.asarray([state.dni_kw_m2 for state in states], dtype=np.float64)
        monthly, annual = aggregate(fine, states, dni, design.width_m * design.height_m)
        convergence = convergence_metrics(
            design.method_id, coarse, fine, states, design.width_m * design.height_m,
            fine_seeds, coarse_samples, fine_samples,
        )
        method_results[design.method_id] = {
            "design": design,
            "points": points,
            "fine": fine,
            "monthly": monthly,
            "annual": annual,
            "convergence": convergence,
            "coarse_seed_summaries": coarse_seeds,
            "fine_seed_summaries": fine_seeds,
            "runtime_seconds": coarse_runtime + fine_runtime,
        }
        prefix = "q2_main" if design.method_id == "M2" else "q2_baseline"
        write_csv(table_dir / f"{prefix}_layout.csv", layout_rows(points, design))
        write_csv(table_dir / f"{prefix}_monthly.csv", monthly)
        write_csv(table_dir / f"{prefix}_annual.csv", [annual])

    convergence = {
        "schema_version": 1,
        "methods": {
            method_id: method_results[method_id]["convergence"] for method_id in ("M2", "B2")
        },
    }
    convergence["status"] = "PASS" if all(
        value["status"] == "PASS" for value in convergence["methods"].values()
    ) else "CONDITIONAL"
    write_json(metric_dir / "q2_convergence.json", convergence)

    main_annual = method_results["M2"]["annual"]
    baseline_annual = method_results["B2"]["annual"]
    main_eta = method_results["M2"]["fine"]["eta_total"]
    baseline_eta = method_results["B2"]["fine"]["eta_total"]
    absolute_gain = main_annual["unit_area_power_kw_m2"] - baseline_annual["unit_area_power_kw_m2"]
    comparison = {
        "schema_version": 1,
        "main_method": "M2",
        "baseline_method": "B2",
        "annual": {
            "main": main_annual,
            "baseline": baseline_annual,
            "unit_area_absolute_gain_kw_m2": absolute_gain,
            "unit_area_relative_gain": absolute_gain / baseline_annual["unit_area_power_kw_m2"],
            "main_power_margin_mw": main_annual["field_power_mw"] - TARGET_POWER_MW,
        },
        "output_degeneracy": {
            "main_eta_total_unique_rounded_6dp": int(len(np.unique(np.round(main_eta, 6)))),
            "baseline_eta_total_unique_rounded_6dp": int(len(np.unique(np.round(baseline_eta, 6)))),
            "main_eta_total_std": float(np.std(main_eta)),
            "baseline_eta_total_std": float(np.std(baseline_eta)),
            "main_zero_mass": float(np.mean(main_eta <= 1e-12)),
            "baseline_zero_mass": float(np.mean(baseline_eta <= 1e-12)),
        },
    }
    write_json(metric_dir / "q2_method_comparison.json", comparison)
    figure_files = make_figures(
        main_points, baseline_points, main_design, baseline_design,
        method_results["M2"]["monthly"], method_results["B2"]["monthly"], figure_dir,
    )

    power_failure = main_annual["field_power_mw"] < TARGET_POWER_MW
    convergence_failure = convergence["status"] != "PASS"
    warnings: list[str] = []
    physical_gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
    if not physical_gpus:
        warnings.append("TensorFlow未识别到物理GPU，正式计算在CPU上执行")
    if power_failure:
        warnings.append("M2正式年平均功率低于60 MW，必须返回设计修复判断点")
    if convergence_failure:
        warnings.append("至少一种方法未通过粗细分辨率收敛阈值")
    method_status = "success" if not power_failure and not convergence_failure else "conditional"
    summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round1" if args.profile == "formal" else "smoke",
        "implementation_target": "python",
        "profile": args.profile,
        "random_seed": list(formal_seeds),
        "approved_decision_id": "q2_method_choice",
        "methods": [
            {
                "method_id": "M2",
                "role": "main_candidate",
                "script": "scripts/Q2/q2_optical_model.py",
                "status": method_status,
                "execution_time_seconds": method_results["M2"]["runtime_seconds"],
                "input_files": [
                    "workspace/methods/Q2/q2_method_card.md",
                    "workspace/methods/Q2/q2_decisions.jsonl",
                ],
                "output_files": [
                    str((table_dir / "q2_main_layout.csv").relative_to(ROOT)),
                    str((table_dir / "q2_main_monthly.csv").relative_to(ROOT)),
                    str((table_dir / "q2_main_annual.csv").relative_to(ROOT)),
                    str((metric_dir / "q2_design_selection.json").relative_to(ROOT)),
                    str((metric_dir / "q2_convergence.json").relative_to(ROOT)),
                ],
                "figure_files": figure_files,
                "metrics_summary": main_annual,
                "warnings": warnings,
                "errors": [],
            },
            {
                "method_id": "B2",
                "role": "usable_baseline",
                "script": "scripts/Q2/q2_optical_model.py",
                "status": "success" if method_results["B2"]["convergence"]["status"] == "PASS" else "conditional",
                "execution_time_seconds": method_results["B2"]["runtime_seconds"],
                "input_files": [],
                "output_files": [
                    str((table_dir / "q2_baseline_layout.csv").relative_to(ROOT)),
                    str((table_dir / "q2_baseline_monthly.csv").relative_to(ROOT)),
                    str((table_dir / "q2_baseline_annual.csv").relative_to(ROOT)),
                    str((metric_dir / "q2_convergence.json").relative_to(ROOT)),
                ],
                "figure_files": figure_files,
                "metrics_summary": baseline_annual,
                "warnings": [],
                "errors": [],
            },
        ],
        "comparison": comparison,
        "output_degeneracy": comparison["output_degeneracy"],
        "validation": validation,
        "fallback_trigger": {
            "fallback_id": None,
            "condition": "M2正式功率低于60 MW、粗细收敛失败或正式单位面积优势不超过数值与扰动不确定性时，返回人类判断点；B2不得自动替代M2",
            "observed": bool(power_failure or convergence_failure),
            "evidence": str((metric_dir / "q2_convergence.json").relative_to(ROOT)),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "tensorflow": tf.__version__,
            "physical_gpus": physical_gpus,
            "batch_size": args.batch_size,
        },
        "warnings": warnings,
        "errors": [],
        "execution_time_seconds": time.perf_counter() - overall_started,
    }
    write_json(round_dir / "run_summary.json", summary)
    return summary


def render_existing(profile: str) -> dict[str, Any]:
    _, table_dir, _, figure_dir = output_dirs(profile)
    main_design = Design(
        "M2_south40_spacing101", "M2", "main_candidate", 0.0, -40.0,
        7.0, 7.0, 4.0, spacing_factor=1.01,
    )
    baseline_design = Design(
        "B2_fixed_7m_centered_hex", "B2", "usable_baseline", 0.0, 0.0, 7.0, 7.0, 4.0
    )
    main_points = np.loadtxt(
        table_dir / "q2_main_layout.csv", delimiter=",", skiprows=1, usecols=(1, 2)
    )
    baseline_points = np.loadtxt(
        table_dir / "q2_baseline_layout.csv", delimiter=",", skiprows=1, usecols=(1, 2)
    )

    def read_monthly(path: Path) -> list[dict[str, float]]:
        with path.open(encoding="utf-8") as stream:
            return [
                {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]

    files = make_figures(
        main_points,
        baseline_points,
        main_design,
        baseline_design,
        read_monthly(table_dir / "q2_main_monthly.csv"),
        read_monthly(table_dir / "q2_baseline_monthly.csv"),
        figure_dir,
    )
    return {"profile": profile, "figure_files": files, "status": "PASS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "all", "render"), nargs="?", default="validate")
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = Design(
        "B2_fixed_7m_centered_hex", "B2", "usable_baseline", 0.0, 0.0, 7.0, 7.0, 4.0
    )
    main_design = Design(
        "M2_south40_spacing101", "M2", "main_candidate", 0.0, -40.0,
        7.0, 7.0, 4.0, spacing_factor=1.01,
    )
    if args.command == "validate":
        round_dir, _, metric_dir, _ = output_dirs(args.profile)
        result = run_validation(
            main_design, hexagonal_layout(main_design), baseline, hexagonal_layout(baseline)
        )
        write_json(metric_dir / "q2_validation.json", result)
        result["output_root"] = str(round_dir.relative_to(ROOT))
    elif args.command == "all":
        result = run_all(args)
    else:
        result = render_existing(args.profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
