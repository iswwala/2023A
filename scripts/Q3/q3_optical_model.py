#!/usr/bin/env python3
"""Heterogeneous-heliostat optical evaluator for Q3."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q1"))
sys.path.insert(0, str(ROOT / "scripts/Q2"))
import q1_optical_model as q1  # noqa: E402
import q2_optical_model as q2  # noqa: E402

SEEDS = q1.SEEDS


@dataclass(frozen=True)
class HeterogeneousDesign:
    design_id: str
    method_id: str
    role: str
    tower_x_m: float
    tower_y_m: float


@dataclass
class TraceResult:
    eta_sb: np.ndarray
    eta_trunc: np.ndarray
    joint_received: np.ndarray


def geometry_checks(
    points_xy: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    installation_heights: np.ndarray,
    design: HeterogeneousDesign,
) -> dict[str, Any]:
    n = len(points_xy)
    if not (len(widths) == len(heights) == len(installation_heights) == n):
        raise ValueError("Q3 geometry arrays must have identical lengths")
    tower = np.asarray([design.tower_x_m, design.tower_y_m])
    tree = cKDTree(points_xy)
    pairs = np.asarray(list(tree.query_pairs(14.0)), dtype=np.int64)
    if len(pairs):
        distances = np.linalg.norm(points_xy[pairs[:, 0]] - points_xy[pairs[:, 1]], axis=1)
        required = (widths[pairs[:, 0]] + widths[pairs[:, 1]]) / 2.0 + 5.0
        slack = distances - required
        minimum_slack = float(np.min(slack))
        violations = int(np.sum(slack < -1e-7))
    else:
        minimum_slack = float("inf")
        violations = 0
    clearance = installation_heights - heights / 2.0
    result = {
        "mirror_count": n,
        "inside_field_count": int(np.sum(np.linalg.norm(points_xy, axis=1) <= 350.0 + 1e-9)),
        "outside_exclusion_count": int(np.sum(np.linalg.norm(points_xy - tower, axis=1) >= 100.0 - 1e-9)),
        "checked_neighbor_pair_count": int(len(pairs)),
        "spacing_violation_count": violations,
        "minimum_spacing_slack_m": minimum_slack,
        "minimum_ground_clearance_m": float(np.min(clearance)),
        "tower_inside_field": bool(np.linalg.norm(tower) <= 350.0 + 1e-9),
        "width_range_m": [float(np.min(widths)), float(np.max(widths))],
        "height_range_m": [float(np.min(heights)), float(np.max(heights))],
        "height_exceeds_width_count": int(np.sum(heights > widths + 1e-9)),
        "installation_height_range_m": [float(np.min(installation_heights)), float(np.max(installation_heights))],
    }
    result["status"] = "PASS" if (
        result["inside_field_count"] == n
        and result["outside_exclusion_count"] == n
        and violations == 0
        and result["minimum_ground_clearance_m"] > 0.0
        and result["tower_inside_field"]
        and np.all((widths >= 2.0) & (widths <= 8.0))
        and np.all((heights >= 2.0) & (heights <= 8.0))
        and result["height_exceeds_width_count"] == 0
        and np.all((installation_heights >= 2.0) & (installation_heights <= 6.0))
    ) else "FAIL"
    return result


def optical_centers(
    points_xy: np.ndarray, installation_heights: np.ndarray, design: HeterogeneousDesign
) -> np.ndarray:
    tower = np.asarray([design.tower_x_m, design.tower_y_m])
    return np.column_stack((points_xy - tower, installation_heights))


def surface_uv_all(
    count: int, seed: int, heliostat_ids: np.ndarray,
    widths: np.ndarray, heights: np.ndarray,
) -> np.ndarray:
    samples = np.empty((len(heliostat_ids), count, 2), dtype=np.float64)
    for row, heliostat_id in enumerate(heliostat_ids):
        unit = q1.sobol_unit(count, 2, seed + int(heliostat_id) * 17) - 0.5
        samples[row] = unit * np.asarray([widths[row], heights[row]])
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
    candidate_half_widths: np.ndarray,
    candidate_half_heights: np.ndarray,
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
    edge_w = tf.cast(candidate_half_widths[:, None, None, :] + 1e-5, dtype)
    edge_h = tf.cast(candidate_half_heights[:, None, None, :] + 1e-5, dtype)
    eps = tf.constant(1e-6, dtype=dtype)

    relative = p[:, :, None, :] - cc[:, None, :, :]
    numerator = -tf.einsum("bski,bki->bsk", relative, cn)
    denominator_in = tf.einsum("ui,bki->buk", sun, cn)
    safe_in = tf.where(tf.abs(denominator_in) > eps, denominator_in, tf.ones_like(denominator_in))
    lambda_in = numerator[:, None, :, :] / safe_in[:, :, None, :]
    base_w = tf.einsum("bski,bki->bsk", relative, cw)
    base_h = tf.einsum("bski,bki->bsk", relative, ch)
    hit_w_in = base_w[:, None, :, :] + lambda_in * tf.einsum("ui,bki->buk", sun, cw)[:, :, None, :]
    hit_h_in = base_h[:, None, :, :] + lambda_in * tf.einsum("ui,bki->buk", sun, ch)[:, :, None, :]
    blocked_in = tf.reduce_any(
        candidate_mask[:, None, None, :]
        & (tf.abs(denominator_in)[:, :, None, :] > eps)
        & (lambda_in > eps)
        & (tf.abs(hit_w_in) <= edge_w)
        & (tf.abs(hit_h_in) <= edge_h), axis=-1,
    )

    tower_a = sun[:, 0] ** 2 + sun[:, 1] ** 2
    tower_b = 2.0 * (p[:, None, :, 0] * sun[None, :, None, 0] + p[:, None, :, 1] * sun[None, :, None, 1])
    tower_c = p[:, None, :, 0] ** 2 + p[:, None, :, 1] ** 2 - q1.TOWER_RADIUS**2
    tower_disc = tower_b**2 - 4.0 * tower_a[None, :, None] * tower_c
    root_term = tf.sqrt(tf.maximum(tower_disc, 0.0))
    denominator = 2.0 * tower_a[None, :, None]
    roots = tf.stack(((-tower_b - root_term) / denominator, (-tower_b + root_term) / denominator), axis=-1)
    tower_z = p[:, None, :, None, 2] + roots * sun[None, :, None, None, 2]
    tower_hit = tf.reduce_any(
        (tower_disc[..., None] >= 0.0) & (roots > eps)
        & (tower_z >= q1.TOWER_Z_MIN) & (tower_z < q1.TOWER_Z_MAX), axis=-1,
    )
    blocked_in = blocked_in | tower_hit

    sun_dot_normal = tf.einsum("ui,bi->bu", sun, target_n)
    outgoing = -sun[None, :, :] + 2.0 * sun_dot_normal[:, :, None] * target_n[:, None, :]
    outgoing /= tf.linalg.norm(outgoing, axis=-1, keepdims=True)
    dx, dy = outgoing[:, :, 0], outgoing[:, :, 1]
    coefficient_a = dx**2 + dy**2
    coefficient_b = 2.0 * (p[:, None, :, 0] * dx[:, :, None] + p[:, None, :, 1] * dy[:, :, None])
    coefficient_c = p[:, None, :, 0] ** 2 + p[:, None, :, 1] ** 2 - q1.RECEIVER_RADIUS**2
    discriminant = coefficient_b**2 - 4.0 * coefficient_a[:, :, None] * coefficient_c
    root_term = tf.sqrt(tf.maximum(discriminant, 0.0))
    infinity = tf.constant(np.inf, dtype=dtype)
    root_a = (-coefficient_b - root_term) / (2.0 * coefficient_a[:, :, None])
    root_b = (-coefficient_b + root_term) / (2.0 * coefficient_a[:, :, None])
    root_a = tf.where((discriminant >= 0.0) & (root_a > eps), root_a, infinity)
    root_b = tf.where((discriminant >= 0.0) & (root_b > eps), root_b, infinity)
    receiver_lambda = tf.minimum(root_a, root_b)
    receiver_z = p[:, None, :, 2] + receiver_lambda * outgoing[:, :, None, 2]
    receiver_hit = tf.math.is_finite(receiver_lambda) & (receiver_z >= q1.RECEIVER_Z_MIN) & (receiver_z <= q1.RECEIVER_Z_MAX)
    plane_lambda = tf.where(
        outgoing[:, :, None, 2] > eps,
        (q1.RECEIVER_Z_MAX - p[:, None, :, 2]) / outgoing[:, :, None, 2], infinity,
    )
    max_lambda = tf.where(receiver_hit, receiver_lambda, plane_lambda)

    denominator_out = tf.einsum("bui,bki->buk", outgoing, cn)
    safe_out = tf.where(tf.abs(denominator_out) > eps, denominator_out, tf.ones_like(denominator_out))
    lambda_out = numerator[:, None, :, :] / safe_out[:, :, None, :]
    hit_w_out = base_w[:, None, :, :] + lambda_out * tf.einsum("bui,bki->buk", outgoing, cw)[:, :, None, :]
    hit_h_out = base_h[:, None, :, :] + lambda_out * tf.einsum("bui,bki->buk", outgoing, ch)[:, :, None, :]
    blocked_out = tf.reduce_any(
        candidate_mask[:, None, None, :]
        & (tf.abs(denominator_out)[:, :, None, :] > eps)
        & (lambda_out > eps)
        & (lambda_out < max_lambda[:, :, :, None] - eps)
        & (tf.abs(hit_w_out) <= edge_w)
        & (tf.abs(hit_h_out) <= edge_h), axis=-1,
    )
    visible = ~blocked_in & ~blocked_out
    received = visible & receiver_hit
    return (
        tf.reduce_sum(tf.cast(visible, tf.int64), axis=(1, 2)),
        tf.reduce_sum(tf.cast(received, tf.int64), axis=(1, 2)),
    )


def trace_full_field(
    centers: np.ndarray, widths: np.ndarray, heights: np.ndarray,
    states: list[q1.SolarState], surface_count: int, sun_count: int,
    seed: int, batch_size: int,
) -> TraceResult:
    n = len(centers)
    eta_sb = np.empty((len(states), n), dtype=np.float64)
    eta_trunc = np.empty_like(eta_sb)
    joint = np.empty_like(eta_sb)
    ids = np.arange(1, n + 1, dtype=np.int64)
    uv = surface_uv_all(surface_count, seed, ids, widths, heights)
    candidate_indices, candidate_valid = q1.build_candidates(centers)
    total = surface_count * sun_count
    for state in states:
        sun = np.asarray(state.sun_vector, dtype=np.float64)
        _, normals, width_axes, height_axes = q1.mirror_frames(centers, sun)
        directions = q1.sun_disk_directions(sun, sun_count, q1.SUN_HALF_ANGLE, seed + 100_000 + state.state_index * 101)
        points = q1.points_from_uv(centers, width_axes, height_axes, uv)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = np.arange(start, stop)
            candidates = candidate_indices[batch]
            visible, received = tuple(value.numpy() for value in trace_dynamic_tensors(
                points[batch], directions, normals[batch], centers[candidates],
                normals[candidates], width_axes[candidates], height_axes[candidates],
                candidate_valid[batch], widths[candidates] / 2.0, heights[candidates] / 2.0,
            ))
            eta_sb[state.state_index, batch] = visible / total
            eta_trunc[state.state_index, batch] = np.divide(received, visible, out=np.zeros_like(received, dtype=float), where=visible > 0)
            joint[state.state_index, batch] = received / total
    return TraceResult(eta_sb, eta_trunc, joint)


def complete_efficiencies(trace: TraceResult, centers: np.ndarray, states: list[q1.SolarState]) -> dict[str, np.ndarray]:
    eta_cos, eta_at, _ = q2.deterministic_components(centers, states)
    eta_total = trace.eta_sb * eta_cos * eta_at[None, :] * trace.eta_trunc * q1.REFLECTIVITY
    return {
        "eta_sb": trace.eta_sb, "eta_cos": eta_cos,
        "eta_at": np.broadcast_to(eta_at[None, :], eta_cos.shape),
        "eta_trunc": trace.eta_trunc, "eta_total": eta_total,
        "joint_received": trace.joint_received,
    }


def aggregate(
    components: dict[str, np.ndarray], states: list[q1.SolarState], areas: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    dni = np.asarray([state.dni_kw_m2 for state in states])
    total_area = float(np.sum(areas))
    state_mean = {
        name: np.sum(values * areas[None, :], axis=1) / total_area
        for name, values in components.items() if name.startswith("eta_")
    }
    unit_power = dni * state_mean["eta_total"]
    field_power = unit_power * total_area / 1000.0
    monthly = []
    for month in range(1, 13):
        indices = np.asarray([i for i, state in enumerate(states) if state.month == month])
        monthly.append({
            "month": month,
            **{name: float(np.mean(values[indices])) for name, values in state_mean.items()},
            "unit_area_power_kw_m2": float(np.mean(unit_power[indices])),
            "field_power_mw": float(np.mean(field_power[indices])),
        })
    annual = {
        **{name: float(np.mean(values)) for name, values in state_mean.items()},
        "field_power_mw": float(np.mean(field_power)),
        "unit_area_power_kw_m2": float(np.mean(unit_power)),
        "total_area_m2": total_area,
    }
    return monthly, annual


def evaluate_resolution(
    points_xy: np.ndarray, widths: np.ndarray, heights: np.ndarray,
    installation_heights: np.ndarray, design: HeterogeneousDesign,
    states: list[q1.SolarState], surface_count: int, sun_count: int,
    seeds: tuple[int, ...], batch_size: int = 64,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    started = time.perf_counter()
    centers = optical_centers(points_xy, installation_heights, design)
    traces, summaries = [], []
    areas = widths * heights
    for seed in seeds:
        trace = trace_full_field(centers, widths, heights, states, surface_count, sun_count, seed, batch_size)
        components = complete_efficiencies(trace, centers, states)
        _, annual = aggregate(components, states, areas)
        summaries.append({"seed": seed, "annual": annual})
        traces.append(trace)
    eta_sb = np.mean(np.stack([x.eta_sb for x in traces]), axis=0)
    joint = np.mean(np.stack([x.joint_received for x in traces]), axis=0)
    eta_trunc = np.divide(joint, eta_sb, out=np.zeros_like(joint), where=eta_sb > 0)
    pooled = complete_efficiencies(TraceResult(eta_sb, eta_trunc, joint), centers, states)
    return pooled, summaries, time.perf_counter() - started
