#!/usr/bin/env python3
"""Risk-screen parameterized Q2 layouts without touching frozen Q1 outputs."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "Q1"))
import q1_optical_model as q1  # noqa: E402


OUTPUT = ROOT / "workspace" / "methods" / "Q2" / "probes" / "risk_probe_raw_metrics.json"
FIELD_RADIUS = 350.0
EXCLUSION_RADIUS = 100.0
TARGET_POWER_MW = 60.0
REPAIR_TARGET_MW = 61.2
SCREEN_STATE_INDICES = (0, 2, 4, 15, 17, 19, 30, 32, 34, 45, 47, 49)


def hexagonal_layout(
    width: float,
    tower_xy: tuple[float, float],
    angle_deg: float = 0.0,
    spacing_factor: float = 1.0,
    phase_x_fraction: float = 0.0,
    phase_y_fraction: float = 0.0,
) -> np.ndarray:
    spacing = (width + 5.0) * spacing_factor
    row_spacing = math.sqrt(3.0) * spacing / 2.0
    limit = FIELD_RADIUS + np.linalg.norm(tower_xy) + spacing
    rows = np.arange(-limit, limit + row_spacing, row_spacing)
    points: list[np.ndarray] = []
    angle = math.radians(angle_deg)
    rotation = np.array(
        [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    for row_index, base_y_value in enumerate(rows):
        y_value = base_y_value + phase_y_fraction * row_spacing
        columns = np.arange(-limit, limit + spacing, spacing)
        columns += (row_index % 2) * spacing / 2.0 + phase_x_fraction * spacing
        relative = np.column_stack((columns, np.full_like(columns, y_value))) @ rotation
        absolute = relative + np.asarray(tower_xy)
        legal = (
            (np.linalg.norm(absolute, axis=1) <= FIELD_RADIUS + 1e-9)
            & (np.linalg.norm(relative, axis=1) >= EXCLUSION_RADIUS - 1e-9)
        )
        points.append(absolute[legal])
    return np.vstack(points)


def midpoint_uv(width: float, height: float, side: int = 5) -> np.ndarray:
    u = ((np.arange(side) + 0.5) / side - 0.5) * width
    v = ((np.arange(side) + 0.5) / side - 0.5) * height
    grid_u, grid_v = np.meshgrid(u, v, indexing="xy")
    return np.column_stack((grid_u.ravel(), grid_v.ravel()))


@tf.function(reduce_retracing=True)
def trace_center_visibility(
    points: np.ndarray,
    sun_direction: np.ndarray,
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
    sun = tf.cast(sun_direction, dtype)
    target_n = tf.cast(target_normals, dtype)
    candidate_c = tf.cast(candidate_centers, dtype)
    candidate_n = tf.cast(candidate_normals, dtype)
    candidate_w = tf.cast(candidate_width_axes, dtype)
    candidate_h = tf.cast(candidate_height_axes, dtype)
    candidate_mask = tf.cast(candidate_valid, tf.bool)
    eps = tf.constant(1e-6, dtype=dtype)
    edge_w = tf.cast(half_width + 1e-5, dtype)
    edge_h = tf.cast(half_height + 1e-5, dtype)

    relative = p[:, :, None, :] - candidate_c[:, None, :, :]
    numerator = -tf.einsum("bski,bki->bsk", relative, candidate_n)
    denominator_in = tf.einsum("ui,bki->buk", sun, candidate_n)
    safe_in = tf.where(tf.abs(denominator_in) > eps, denominator_in, tf.ones_like(denominator_in))
    lambda_in = numerator[:, None, :, :] / safe_in[:, :, None, :]
    base_w = tf.einsum("bski,bki->bsk", relative, candidate_w)
    base_h = tf.einsum("bski,bki->bsk", relative, candidate_h)
    direction_w_in = tf.einsum("ui,bki->buk", sun, candidate_w)
    direction_h_in = tf.einsum("ui,bki->buk", sun, candidate_h)
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

    denominator_out = tf.einsum("bui,bki->buk", outgoing, candidate_n)
    safe_out = tf.where(tf.abs(denominator_out) > eps, denominator_out, tf.ones_like(denominator_out))
    lambda_out = numerator[:, None, :, :] / safe_out[:, :, None, :]
    direction_w_out = tf.einsum("bui,bki->buk", outgoing, candidate_w)
    direction_h_out = tf.einsum("bui,bki->buk", outgoing, candidate_h)
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
    visible_count = tf.reduce_sum(tf.cast(visible, tf.int64), axis=(1, 2))
    return visible_count, visible


def evaluate_design(
    points_xy: np.ndarray,
    tower_xy: tuple[float, float],
    width: float,
    height: float,
    installation_height: float,
    states: list[q1.SolarState],
    state_indices: tuple[int, ...] | None,
    batch_size: int = 128,
) -> dict[str, Any]:
    started = time.perf_counter()
    relative_xy = points_xy - np.asarray(tower_xy)
    centers = np.column_stack((relative_xy, np.full(len(points_xy), installation_height)))
    ids = np.arange(1, len(centers) + 1, dtype=np.int64)
    candidate_indices, candidate_valid = q1.build_candidates(centers)
    active_states = states if state_indices is None else [states[index] for index in state_indices]
    area = width * height
    uv = midpoint_uv(width, height)
    eta_cos_rows: list[np.ndarray] = []
    eta_at = q1.atmospheric_efficiency(centers)
    eta_sb_rows: list[np.ndarray] = []
    eta_trunc_rows: list[np.ndarray] = []
    total_rows: list[np.ndarray] = []
    power_rows: list[np.ndarray] = []

    for state in active_states:
        sun = np.asarray(state.sun_vector, dtype=np.float64)
        _, normals, width_axes, height_axes = q1.mirror_frames(centers, sun)
        points = q1.points_from_uv(centers, width_axes, height_axes, uv)
        sun_directions = q1.sun_disk_directions(
            sun, 16, q1.SUN_HALF_ANGLE, q1.SEEDS[0] + 300_000 + state.state_index * 101
        )
        outgoing = q1.reflect_directions(
            np.broadcast_to(sun_directions[None, :, :], (len(centers), 16, 3)),
            np.broadcast_to(normals[:, None, :], (len(centers), 16, 3)),
        )
        eta_sb = np.empty(len(centers), dtype=np.float64)
        eta_trunc = np.empty(len(centers), dtype=np.float64)
        for start in range(0, len(centers), batch_size):
            stop = min(start + batch_size, len(centers))
            batch = np.arange(start, stop)
            candidates = candidate_indices[batch]
            visible_count_tensor, visible_tensor = trace_center_visibility(
                points[batch],
                sun[None, :],
                normals[batch],
                centers[candidates],
                normals[candidates],
                width_axes[candidates],
                height_axes[candidates],
                candidate_valid[batch],
                width / 2.0,
                height / 2.0,
            )
            visible_count = visible_count_tensor.numpy()
            center_visible = visible_tensor.numpy()[:, 0, :]
            receiver_hit, _ = q1.receiver_hits_numpy(points[batch], outgoing[batch])
            received_count = np.sum(receiver_hit & center_visible[:, None, :], axis=(1, 2))
            eta_sb[batch] = visible_count / len(uv)
            denominator = visible_count * len(sun_directions)
            eta_trunc[batch] = np.divide(
                received_count,
                denominator,
                out=np.zeros_like(received_count, dtype=np.float64),
                where=denominator > 0,
            )
        eta_cos = normals @ sun
        eta_total = eta_sb * eta_cos * eta_at * eta_trunc * q1.REFLECTIVITY
        mirror_power = state.dni_kw_m2 * area * eta_total
        eta_cos_rows.append(eta_cos)
        eta_sb_rows.append(eta_sb)
        eta_trunc_rows.append(eta_trunc)
        total_rows.append(eta_total)
        power_rows.append(mirror_power)

    eta_total_matrix = np.asarray(total_rows)
    power_matrix = np.asarray(power_rows)
    annual_power_mw = float(np.mean(np.sum(power_matrix, axis=1)) / 1000.0)
    return {
        "mirror_count": len(centers),
        "total_area_m2": len(centers) * area,
        "eta_total": float(np.mean(eta_total_matrix)),
        "eta_cos": float(np.mean(eta_cos_rows)),
        "eta_sb": float(np.mean(eta_sb_rows)),
        "eta_trunc": float(np.mean(eta_trunc_rows)),
        "field_power_mw": annual_power_mw,
        "unit_area_power_kw_m2": annual_power_mw * 1000.0 / (len(centers) * area),
        "mirror_mean_power_kw": np.mean(power_matrix, axis=0).tolist(),
        "finite_output_rate": float(np.mean(np.isfinite(eta_total_matrix))),
        "eta_total_unique_rounded_6dp": int(len(np.unique(np.round(eta_total_matrix, 6)))),
        "eta_total_std": float(np.std(eta_total_matrix)),
        "runtime_seconds": time.perf_counter() - started,
        "state_count": len(active_states),
        "candidate_width_max": int(candidate_indices.shape[1]),
        "ids_checksum": int(np.sum(ids)),
    }


def geometry_checks(
    points: np.ndarray,
    tower_xy: tuple[float, float],
    width: float,
    height: float,
    installation_height: float,
) -> dict[str, Any]:
    nearest = cKDTree(points).query(points, k=2)[0][:, 1]
    return {
        "inside_field_count": int(np.sum(np.linalg.norm(points, axis=1) <= FIELD_RADIUS + 1e-9)),
        "outside_exclusion_count": int(
            np.sum(np.linalg.norm(points - np.asarray(tower_xy), axis=1) >= EXCLUSION_RADIUS - 1e-9)
        ),
        "minimum_spacing_m": float(np.min(nearest)),
        "required_spacing_m": width + 5.0,
        "ground_clearance_m": installation_height - height / 2.0,
        "tower_inside_field": bool(np.linalg.norm(tower_xy) <= FIELD_RADIUS),
    }


def design_record(
    design_id: str,
    width: float,
    height: float,
    installation_height: float,
    tower_xy: tuple[float, float] = (0.0, 0.0),
    angle_deg: float = 0.0,
    spacing_factor: float = 1.0,
    phase_x_fraction: float = 0.0,
    phase_y_fraction: float = 0.0,
) -> dict[str, Any]:
    return {
        "design_id": design_id,
        "width_m": width,
        "height_m": height,
        "installation_height_m": installation_height,
        "tower_xy_m": list(tower_xy),
        "lattice_angle_deg": angle_deg,
        "spacing_factor": spacing_factor,
        "phase_x_fraction": phase_x_fraction,
        "phase_y_fraction": phase_y_fraction,
    }


def main() -> None:
    overall_started = time.perf_counter()
    states = q1.build_solar_states()
    baseline = design_record("B2_fixed_7m_centered_hex", 7.0, 7.0, 4.0)
    candidates = [
        design_record("M2_w6_h6_center", 6.0, 6.0, 3.5),
        design_record("M2_w7_h6_center", 7.0, 6.0, 3.5),
        design_record("M2_w7_h7_center", 7.0, 7.0, 4.0),
        design_record("M2_w8_h6_center", 8.0, 6.0, 3.5),
        design_record("M2_w8_h7_center", 8.0, 7.0, 4.0),
        design_record("M2_w8_h8_center", 8.0, 8.0, 4.5),
        design_record("M2_w7_h7_north40", 7.0, 7.0, 4.0, (0.0, 40.0)),
        design_record("M2_w7_h7_south40", 7.0, 7.0, 4.0, (0.0, -40.0)),
        design_record("M2_w7_h7_east40", 7.0, 7.0, 4.0, (40.0, 0.0)),
        design_record("M2_w7_h7_angle15", 7.0, 7.0, 4.0, angle_deg=15.0),
        design_record("M2_w7_h7_spacing102", 7.0, 7.0, 4.0, spacing_factor=1.02),
        design_record("M2_w7_h7_south30", 7.0, 7.0, 4.0, (0.0, -30.0)),
        design_record("M2_w7_h7_south35", 7.0, 7.0, 4.0, (0.0, -35.0)),
        design_record("M2_w7_h7_south45", 7.0, 7.0, 4.0, (0.0, -45.0)),
        design_record("M2_w7_h7_south50", 7.0, 7.0, 4.0, (0.0, -50.0)),
        design_record("M2_w7_h7_south40_west5", 7.0, 7.0, 4.0, (-5.0, -40.0)),
        design_record("M2_w7_h7_south40_east5", 7.0, 7.0, 4.0, (5.0, -40.0)),
        design_record("M2_w7_h7_south40_angle_m5", 7.0, 7.0, 4.0, (0.0, -40.0), -5.0),
        design_record("M2_w7_h7_south40_angle_p5", 7.0, 7.0, 4.0, (0.0, -40.0), 5.0),
        design_record(
            "M2_w7_h7_south40_spacing1005",
            7.0,
            7.0,
            4.0,
            (0.0, -40.0),
            spacing_factor=1.005,
        ),
        design_record(
            "M2_w7_h7_south40_spacing101",
            7.0,
            7.0,
            4.0,
            (0.0, -40.0),
            spacing_factor=1.01,
        ),
        design_record(
            "M2_w7_h7_south40_phase_x_m025",
            7.0,
            7.0,
            4.0,
            (0.0, -40.0),
            phase_x_fraction=-0.25,
        ),
        design_record(
            "M2_w7_h7_south40_phase_x_p025",
            7.0,
            7.0,
            4.0,
            (0.0, -40.0),
            phase_x_fraction=0.25,
        ),
        design_record(
            "M2_w7_h7_south40_phase_y_m025",
            7.0,
            7.0,
            4.0,
            (0.0, -40.0),
            phase_y_fraction=-0.25,
        ),
        design_record(
            "M2_w7_h7_south40_phase_y_p025",
            7.0,
            7.0,
            4.0,
            (0.0, -40.0),
            phase_y_fraction=0.25,
        ),
    ]

    screening: list[dict[str, Any]] = []
    for design in candidates:
        points = hexagonal_layout(
            design["width_m"],
            tuple(design["tower_xy_m"]),
            design["lattice_angle_deg"],
            design["spacing_factor"],
            design["phase_x_fraction"],
            design["phase_y_fraction"],
        )
        metrics = evaluate_design(
            points,
            tuple(design["tower_xy_m"]),
            design["width_m"],
            design["height_m"],
            design["installation_height_m"],
            states,
            SCREEN_STATE_INDICES,
        )
        screening.append(
            {
                **design,
                "geometry": geometry_checks(
                    points,
                    tuple(design["tower_xy_m"]),
                    design["width_m"],
                    design["height_m"],
                    design["installation_height_m"],
                ),
                "screen_metrics": metrics,
            }
        )
        print(design["design_id"], metrics["field_power_mw"], metrics["unit_area_power_kw_m2"], flush=True)

    baseline_points = hexagonal_layout(
        baseline["width_m"],
        tuple(baseline["tower_xy_m"]),
        baseline["lattice_angle_deg"],
        baseline["spacing_factor"],
        baseline["phase_x_fraction"],
        baseline["phase_y_fraction"],
    )
    baseline_full = evaluate_design(
        baseline_points,
        tuple(baseline["tower_xy_m"]),
        baseline["width_m"],
        baseline["height_m"],
        baseline["installation_height_m"],
        states,
        None,
    )
    baseline_record = {
        **baseline,
        "geometry": geometry_checks(
            baseline_points,
            tuple(baseline["tower_xy_m"]),
            baseline["width_m"],
            baseline["height_m"],
            baseline["installation_height_m"],
        ),
        "full_metrics": baseline_full,
    }

    eligible = [row for row in screening if row["screen_metrics"]["field_power_mw"] >= 59.5]
    if len(eligible) < 6:
        eligible = sorted(
            screening,
            key=lambda row: row["screen_metrics"]["field_power_mw"],
            reverse=True,
        )[:6]
    ranked = sorted(
        eligible,
        key=lambda row: row["screen_metrics"]["unit_area_power_kw_m2"],
        reverse=True,
    )
    full_rechecks: list[dict[str, Any]] = []
    for row in ranked[:6]:
        points = hexagonal_layout(
            row["width_m"],
            tuple(row["tower_xy_m"]),
            row["lattice_angle_deg"],
            row["spacing_factor"],
            row["phase_x_fraction"],
            row["phase_y_fraction"],
        )
        full_metrics = evaluate_design(
            points,
            tuple(row["tower_xy_m"]),
            row["width_m"],
            row["height_m"],
            row["installation_height_m"],
            states,
            None,
        )
        full_rechecks.append({"design_id": row["design_id"], "full_metrics": full_metrics})

    best = max(
        full_rechecks,
        key=lambda row: (
            row["full_metrics"]["field_power_mw"] >= TARGET_POWER_MW,
            row["full_metrics"]["unit_area_power_kw_m2"],
        ),
    )
    best_design = next(row for row in screening if row["design_id"] == best["design_id"])
    best_points = hexagonal_layout(
        best_design["width_m"],
        tuple(best_design["tower_xy_m"]),
        best_design["lattice_angle_deg"],
        best_design["spacing_factor"],
        best_design["phase_x_fraction"],
        best_design["phase_y_fraction"],
    )
    mirror_power = np.asarray(best["full_metrics"]["mirror_mean_power_kw"])
    ranked_indices = np.argsort(mirror_power)[::-1]
    selected_count = min(
        len(ranked_indices),
        int(np.searchsorted(np.cumsum(mirror_power[ranked_indices]), REPAIR_TARGET_MW * 1000.0) + 1),
    )
    selected_indices = np.sort(ranked_indices[:selected_count])
    repaired_points = best_points[selected_indices]
    repaired_metrics = evaluate_design(
        repaired_points,
        tuple(best_design["tower_xy_m"]),
        best_design["width_m"],
        best_design["height_m"],
        best_design["installation_height_m"],
        states,
        None,
    )

    screen_scores = np.asarray([row["screen_metrics"]["unit_area_power_kw_m2"] for row in screening])
    screen_powers = np.asarray([row["screen_metrics"]["field_power_mw"] for row in screening])
    screen_by_id = {row["design_id"]: row["screen_metrics"] for row in screening}
    recheck_ids = [row["design_id"] for row in full_rechecks]
    recheck_screen_power = np.asarray(
        [screen_by_id[design_id]["field_power_mw"] for design_id in recheck_ids]
    )
    recheck_full_power = np.asarray(
        [row["full_metrics"]["field_power_mw"] for row in full_rechecks]
    )
    recheck_screen_score = np.asarray(
        [screen_by_id[design_id]["unit_area_power_kw_m2"] for design_id in recheck_ids]
    )
    recheck_full_score = np.asarray(
        [row["full_metrics"]["unit_area_power_kw_m2"] for row in full_rechecks]
    )
    power_rho = float(spearmanr(recheck_screen_power, recheck_full_power).statistic)
    score_rho = float(spearmanr(recheck_screen_score, recheck_full_score).statistic)
    relative_power_error = (recheck_screen_power - recheck_full_power) / recheck_full_power
    south40_screen = screen_by_id["M2_w7_h7_south40"]
    local_rows = [
        row
        for row in screening
        if row["design_id"].startswith("M2_w7_h7_south")
    ]
    local_perturbation_summary = {
        "reference_design_id": "M2_w7_h7_south40",
        "candidate_count": len(local_rows),
        "power_delta_mw_range": [
            float(
                min(
                    row["screen_metrics"]["field_power_mw"]
                    - south40_screen["field_power_mw"]
                    for row in local_rows
                )
            ),
            float(
                max(
                    row["screen_metrics"]["field_power_mw"]
                    - south40_screen["field_power_mw"]
                    for row in local_rows
                )
            ),
        ],
        "unit_area_delta_kw_m2_range": [
            float(
                min(
                    row["screen_metrics"]["unit_area_power_kw_m2"]
                    - south40_screen["unit_area_power_kw_m2"]
                    for row in local_rows
                )
            ),
            float(
                max(
                    row["screen_metrics"]["unit_area_power_kw_m2"]
                    - south40_screen["unit_area_power_kw_m2"]
                    for row in local_rows
                )
            ),
        ],
    }
    result = {
        "schema_version": 1,
        "question_id": "Q2",
        "probe_scope": {
            "screen_states": list(SCREEN_STATE_INDICES),
            "screen_state_count": len(SCREEN_STATE_INDICES),
            "full_state_count": len(states),
            "layout_family": "center-constrained hexagonal candidate lattice",
            "evaluator": "5x5 mirror midpoint center-sun visibility plus 16-direction pillbox truncation",
            "backend": "tensorflow",
            "physical_gpu_visible": bool(tf.config.list_physical_devices("GPU")),
        },
        "baseline": baseline_record,
        "screening_candidates": screening,
        "screening_summary": {
            "candidate_count": len(screening),
            "unique_scores_rounded_6dp": int(len(np.unique(np.round(screen_scores, 6)))),
            "score_coefficient_of_variation": float(np.std(screen_scores) / np.mean(screen_scores)),
            "power_range_mw": [float(np.min(screen_powers)), float(np.max(screen_powers))],
            "screen_feasible_count": int(np.sum(screen_powers >= TARGET_POWER_MW)),
        },
        "full_rechecks": full_rechecks,
        "ranking_stability": {
            "rechecked_candidate_count": len(full_rechecks),
            "screen_vs_full_power_spearman_rho": power_rho,
            "screen_vs_full_unit_area_spearman_rho": score_rho,
            "screen_vs_full_relative_power_error_range": [
                float(np.min(relative_power_error)),
                float(np.max(relative_power_error)),
            ],
            "screen_vs_full_relative_power_error_mean": float(np.mean(relative_power_error)),
            "screen_top_design_id": recheck_ids[int(np.argmax(recheck_screen_score))],
            "full_top_design_id": recheck_ids[int(np.argmax(recheck_full_score))],
        },
        "local_perturbation_summary": local_perturbation_summary,
        "repair": {
            "source_design_id": best_design["design_id"],
            "pre_repair_mirror_count": len(best_points),
            "selected_mirror_count": len(repaired_points),
            "selection_rule": "rank full-60-state mean mirror power and retain the smallest prefix reaching 61.2 MW before interaction re-evaluation",
            "geometry": geometry_checks(
                repaired_points,
                tuple(best_design["tower_xy_m"]),
                best_design["width_m"],
                best_design["height_m"],
                best_design["installation_height_m"],
            ),
            "full_metrics_after_re_evaluation": repaired_metrics,
        },
        "runtime_seconds": time.perf_counter() - overall_started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_power_mw": baseline_full["field_power_mw"],
        "screening": result["screening_summary"],
        "repaired_power_mw": repaired_metrics["field_power_mw"],
        "repaired_unit_power_kw_m2": repaired_metrics["unit_area_power_kw_m2"],
        "ranking_stability": result["ranking_stability"],
        "runtime_seconds": result["runtime_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
