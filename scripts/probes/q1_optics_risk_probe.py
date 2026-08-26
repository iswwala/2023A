#!/usr/bin/env python3
"""Risk probe for Q1 optical-model candidates, not the final solver."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "workspace" / "data_clean" / "Q1" / "heliostat_positions.csv"
OUTPUT = ROOT / "workspace" / "methods" / "Q1" / "probes" / "risk_probe_raw_metrics.json"

LATITUDE = math.radians(39.4)
MIRROR_WIDTH = 6.0
MIRROR_HEIGHT = 6.0
RECEIVER_CENTER = np.array([0.0, 0.0, 80.0])
RECEIVER_RADIUS = 3.5
RECEIVER_Z_MIN = 76.0
RECEIVER_Z_MAX = 84.0
SUN_HALF_ANGLE = 4.65e-3
NEIGHBOR_RADIUS = 60.0
EPS = 1e-9

DAYS = {1: -59, 2: -28, 3: 0, 4: 31, 5: 61, 6: 92,
        7: 122, 8: 153, 9: 184, 10: 214, 11: 245, 12: 275}
PROBE_TIMES = [(1, 9.0), (3, 12.0), (6, 9.0), (6, 12.0), (9, 15.0), (12, 15.0)]


@dataclass
class SolarState:
    month: int
    solar_time: float
    declination_rad: float
    hour_angle_rad: float
    altitude_rad: float
    azimuth_rad: float
    dni_kw_m2: float
    sun_vector: list[float]


def load_centers() -> np.ndarray:
    rows = []
    with INPUT.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append((float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))
    return np.asarray(rows, dtype=float)


def solar_state(month: int, solar_time: float) -> SolarState:
    day = DAYS[month]
    declination = math.asin(
        math.sin(2 * math.pi * day / 365) * math.sin(math.radians(23.45))
    )
    hour_angle = math.pi / 12 * (solar_time - 12)
    sin_altitude = (
        math.cos(declination) * math.cos(LATITUDE) * math.cos(hour_angle)
        + math.sin(declination) * math.sin(LATITUDE)
    )
    altitude = math.asin(max(-1.0, min(1.0, sin_altitude)))
    cos_azimuth = (
        math.sin(declination) - math.sin(altitude) * math.sin(LATITUDE)
    ) / (math.cos(altitude) * math.cos(LATITUDE))
    base_azimuth = math.acos(max(-1.0, min(1.0, cos_azimuth)))
    azimuth = base_azimuth if hour_angle <= 0 else 2 * math.pi - base_azimuth
    sun = np.array([
        math.cos(altitude) * math.sin(azimuth),
        math.cos(altitude) * math.cos(azimuth),
        math.sin(altitude),
    ])
    a = 0.4237 - 0.00821 * (6 - 3.0) ** 2
    b = 0.5055 + 0.00595 * (6.5 - 3.0) ** 2
    c = 0.2711 + 0.01858 * (2.5 - 3.0) ** 2
    dni = 1.366 * (a + b * math.exp(-c / math.sin(altitude)))
    return SolarState(
        month=month,
        solar_time=solar_time,
        declination_rad=declination,
        hour_angle_rad=hour_angle,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        dni_kw_m2=dni,
        sun_vector=sun.tolist(),
    )


def mirror_frames(centers: np.ndarray, sun: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = RECEIVER_CENTER - centers
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    normals = target + sun[None, :]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    width_axes = np.column_stack((-normals[:, 1], normals[:, 0], np.zeros(len(normals))))
    width_axes /= np.linalg.norm(width_axes, axis=1, keepdims=True)
    height_axes = np.cross(normals, width_axes)
    height_axes /= np.linalg.norm(height_axes, axis=1, keepdims=True)
    return target, normals, width_axes, height_axes


def reflect(propagation: np.ndarray, normal: np.ndarray) -> np.ndarray:
    reflected = propagation - 2.0 * np.dot(propagation, normal) * normal
    return reflected / np.linalg.norm(reflected)


def sobol_surface(center: np.ndarray, width_axis: np.ndarray, height_axis: np.ndarray,
                  count: int, seed: int) -> np.ndarray:
    exponent = int(round(math.log2(count)))
    if 2 ** exponent != count:
        raise ValueError("Sobol count must be a power of two")
    unit = qmc.Sobol(2, scramble=True, seed=seed).random_base2(exponent)
    uv = (unit - 0.5) * np.array([MIRROR_WIDTH, MIRROR_HEIGHT])
    return center + uv[:, :1] * width_axis + uv[:, 1:] * height_axis


def midpoint_grid(center: np.ndarray, width_axis: np.ndarray, height_axis: np.ndarray,
                  side: int) -> np.ndarray:
    coords = (np.arange(side) + 0.5) / side - 0.5
    u, v = np.meshgrid(coords * MIRROR_WIDTH, coords * MIRROR_HEIGHT, indexing="xy")
    return center + u.ravel()[:, None] * width_axis + v.ravel()[:, None] * height_axis


def sun_disk_directions(sun: np.ndarray, count: int, half_angle: float, seed: int) -> np.ndarray:
    exponent = int(round(math.log2(count)))
    if 2 ** exponent != count:
        raise ValueError("Sun-ray count must be a power of two")
    unit = qmc.Sobol(2, scramble=True, seed=seed).random_base2(exponent)
    reference = np.array([0.0, 0.0, 1.0]) if abs(sun[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    axis_a = np.cross(sun, reference)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(sun, axis_a)
    cos_theta = 1.0 - unit[:, 0] * (1.0 - math.cos(half_angle))
    theta = np.arccos(cos_theta)
    psi = 2 * math.pi * unit[:, 1]
    directions = (
        np.cos(theta)[:, None] * sun
        + np.sin(theta)[:, None]
        * (np.cos(psi)[:, None] * axis_a + np.sin(psi)[:, None] * axis_b)
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def receiver_hits(points: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = direction[0], direction[1]
    a = dx * dx + dy * dy
    b = 2.0 * (points[:, 0] * dx + points[:, 1] * dy)
    c = points[:, 0] ** 2 + points[:, 1] ** 2 - RECEIVER_RADIUS ** 2
    discriminant = b * b - 4.0 * a * c
    roots = np.full((len(points), 2), np.inf)
    valid = discriminant >= 0
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    roots[valid, 0] = (-b[valid] - sqrt_disc[valid]) / (2.0 * a)
    roots[valid, 1] = (-b[valid] + sqrt_disc[valid]) / (2.0 * a)
    roots[roots <= EPS] = np.inf
    lam = np.min(roots, axis=1)
    z = points[:, 2] + lam * direction[2]
    hit = np.isfinite(lam) & (z >= RECEIVER_Z_MIN) & (z <= RECEIVER_Z_MAX)
    return hit, lam


def ray_clear(points: np.ndarray, direction: np.ndarray, candidate_indices: np.ndarray,
              centers: np.ndarray, normals: np.ndarray, width_axes: np.ndarray,
              height_axes: np.ndarray, max_lambda: np.ndarray | None = None) -> np.ndarray:
    blocked = np.zeros(len(points), dtype=bool)
    for index in candidate_indices:
        normal = normals[index]
        denominator = float(np.dot(direction, normal))
        if abs(denominator) <= EPS:
            continue
        lam = (centers[index] - points) @ normal / denominator
        eligible = lam > EPS
        if max_lambda is not None:
            eligible &= lam < max_lambda - EPS
        if not np.any(eligible):
            continue
        intersection = points[eligible] + lam[eligible, None] * direction
        relative = intersection - centers[index]
        within_width = np.abs(relative @ width_axes[index]) <= MIRROR_WIDTH / 2 + 1e-10
        within_height = np.abs(relative @ height_axes[index]) <= MIRROR_HEIGHT / 2 + 1e-10
        eligible_rows = np.flatnonzero(eligible)
        blocked[eligible_rows[within_width & within_height]] = True
        if np.all(blocked):
            break
    return ~blocked


def atmospheric_efficiency(center: np.ndarray) -> float:
    distance = float(np.linalg.norm(RECEIVER_CENTER - center))
    return 0.99321 - 0.0001176 * distance + 1.97e-8 * distance * distance


def evaluate_main(index: int, centers: np.ndarray, frames: tuple[np.ndarray, ...],
                  tree: cKDTree, sun: np.ndarray, surface_count: int = 64,
                  sun_count: int = 32, half_angle: float = SUN_HALF_ANGLE,
                  seed: int = 2023, use_all_candidates: bool = False) -> dict:
    target, normals, width_axes, height_axes = frames
    candidates = np.arange(len(centers)) if use_all_candidates else np.asarray(
        tree.query_ball_point(centers[index, :2], NEIGHBOR_RADIUS), dtype=int
    )
    candidates = candidates[candidates != index]
    points = sobol_surface(centers[index], width_axes[index], height_axes[index], surface_count, seed + index)
    sun_directions = sun_disk_directions(sun, sun_count, half_angle, seed + 10000 + index)
    visible_count = 0
    received_count = 0
    for sun_direction in sun_directions:
        incoming_clear = ray_clear(
            points, sun_direction, candidates, centers, normals, width_axes, height_axes
        )
        outgoing = reflect(-sun_direction, normals[index])
        max_lambda = np.full(len(points), np.inf)
        if outgoing[2] > EPS:
            max_lambda = (RECEIVER_Z_MAX - points[:, 2]) / outgoing[2]
        outgoing_clear = ray_clear(
            points, outgoing, candidates, centers, normals, width_axes, height_axes, max_lambda
        )
        visible = incoming_clear & outgoing_clear
        hit, _ = receiver_hits(points, outgoing)
        visible_count += int(np.sum(visible))
        received_count += int(np.sum(visible & hit))

    total_rays = surface_count * sun_count
    eta_sb = visible_count / total_rays
    eta_trunc = received_count / visible_count if visible_count else 0.0
    eta_cos = float(np.dot(normals[index], sun))
    eta_at = atmospheric_efficiency(centers[index])
    eta_total = eta_sb * eta_cos * eta_at * eta_trunc * 0.92
    return {
        "eta_sb": eta_sb,
        "eta_cos": eta_cos,
        "eta_at": eta_at,
        "eta_trunc": eta_trunc,
        "eta_total": eta_total,
        "joint_visible_and_received": received_count / total_rays,
        "candidate_count": int(len(candidates)),
    }


def evaluate_baseline(index: int, centers: np.ndarray, frames: tuple[np.ndarray, ...],
                      tree: cKDTree, sun: np.ndarray, grid_side: int = 5,
                      sun_count: int = 16, half_angle: float = SUN_HALF_ANGLE,
                      seed: int = 2023) -> dict:
    target, normals, width_axes, height_axes = frames
    candidates = np.asarray(tree.query_ball_point(centers[index, :2], NEIGHBOR_RADIUS), dtype=int)
    candidates = candidates[candidates != index]
    points = midpoint_grid(centers[index], width_axes[index], height_axes[index], grid_side)
    incoming_clear = ray_clear(points, sun, candidates, centers, normals, width_axes, height_axes)
    outgoing_center = reflect(-sun, normals[index])
    max_lambda = (RECEIVER_Z_MAX - points[:, 2]) / outgoing_center[2]
    outgoing_clear = ray_clear(
        points, outgoing_center, candidates, centers, normals, width_axes, height_axes, max_lambda
    )
    visible = incoming_clear & outgoing_clear
    eta_sb = float(np.mean(visible))

    sun_directions = sun_disk_directions(sun, sun_count, half_angle, seed + 20000 + index)
    received = 0
    denominator = int(np.sum(visible)) * sun_count
    for sun_direction in sun_directions:
        outgoing = reflect(-sun_direction, normals[index])
        hit, _ = receiver_hits(points, outgoing)
        received += int(np.sum(hit & visible))
    eta_trunc = received / denominator if denominator else 0.0
    eta_cos = float(np.dot(normals[index], sun))
    eta_at = atmospheric_efficiency(centers[index])
    eta_total = eta_sb * eta_cos * eta_at * eta_trunc * 0.92
    return {
        "eta_sb": eta_sb,
        "eta_cos": eta_cos,
        "eta_at": eta_at,
        "eta_trunc": eta_trunc,
        "eta_total": eta_total,
        "joint_visible_and_received": eta_sb * eta_trunc,
        "candidate_count": int(len(candidates)),
    }


def stratified_indices(centers: np.ndarray) -> list[int]:
    radii = np.linalg.norm(centers[:, :2], axis=1)
    angles = (np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) + 360) % 360
    radius_edges = np.quantile(radii, [0.0, 1 / 3, 2 / 3, 1.0])
    selected = []
    for radial_index in range(3):
        radial_mask = (radii >= radius_edges[radial_index]) & (
            radii <= radius_edges[radial_index + 1] if radial_index == 2
            else radii < radius_edges[radial_index + 1]
        )
        radial_target = (radius_edges[radial_index] + radius_edges[radial_index + 1]) / 2
        for sector in range(12):
            lower, upper = sector * 30, (sector + 1) * 30
            mask = radial_mask & (angles >= lower) & (angles < upper)
            candidates = np.flatnonzero(mask)
            target_angle = lower + 15
            angle_delta = np.abs(((angles[candidates] - target_angle + 180) % 360) - 180)
            score = np.abs(radii[candidates] - radial_target) / 25 + angle_delta / 15
            selected.append(int(candidates[np.argmin(score)]))
    return selected


def summarize(records: list[dict]) -> dict:
    summary = {}
    for field in ["eta_sb", "eta_cos", "eta_at", "eta_trunc", "eta_total"]:
        values = np.asarray([record[field] for record in records])
        summary[field] = {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "unique_rounded_6dp": int(len(np.unique(np.round(values, 6)))),
            "mass_at_zero": float(np.mean(values <= 1e-12)),
            "mass_at_one": float(np.mean(values >= 1 - 1e-12)),
        }
    return summary


def main() -> None:
    centers = load_centers()
    tree = cKDTree(centers[:, :2])
    selected = stratified_indices(centers)
    states = [solar_state(month, solar_time) for month, solar_time in PROBE_TIMES]
    main_records: list[dict] = []
    baseline_records: list[dict] = []
    reflection_residuals = []
    center_receiver_hits = []

    start = time.perf_counter()
    baseline_start = time.perf_counter()
    for state in states:
        sun = np.asarray(state.sun_vector)
        frames = mirror_frames(centers, sun)
        target, normals, _, _ = frames
        for index in selected:
            reflected = reflect(-sun, normals[index])
            reflection_residuals.append(float(np.linalg.norm(reflected - target[index])))
            center_hit, _ = receiver_hits(centers[index:index + 1], reflected)
            center_receiver_hits.append(bool(center_hit[0]))
            record = evaluate_baseline(index, centers, frames, tree, sun)
            record.update({"mirror_index": index, "month": state.month, "solar_time": state.solar_time})
            baseline_records.append(record)
    baseline_runtime = time.perf_counter() - baseline_start

    main_start = time.perf_counter()
    for state in states:
        sun = np.asarray(state.sun_vector)
        frames = mirror_frames(centers, sun)
        for index in selected:
            record = evaluate_main(index, centers, frames, tree, sun)
            record.update({"mirror_index": index, "month": state.month, "solar_time": state.solar_time})
            main_records.append(record)
    main_runtime = time.perf_counter() - main_start

    # Targeted sensitivity uses one mirror per radial band and four seasonal states.
    sensitivity_indices = selected[::12]
    sensitivity_states = states[:2] + states[-2:]
    sensitivity = []
    candidate_pruning_checks = []
    for state in sensitivity_states:
        sun = np.asarray(state.sun_vector)
        frames = mirror_frames(centers, sun)
        for index in sensitivity_indices:
            coarse = evaluate_main(index, centers, frames, tree, sun, surface_count=32, sun_count=16)
            reference = evaluate_main(index, centers, frames, tree, sun, surface_count=128, sun_count=64)
            narrow = evaluate_main(
                index, centers, frames, tree, sun, surface_count=64, sun_count=32,
                half_angle=SUN_HALF_ANGLE * 0.95,
            )
            wide = evaluate_main(
                index, centers, frames, tree, sun, surface_count=64, sun_count=32,
                half_angle=SUN_HALF_ANGLE * 1.05,
            )
            sensitivity.append({
                "mirror_index": index,
                "month": state.month,
                "solar_time": state.solar_time,
                "coarse_eta_total": coarse["eta_total"],
                "reference_eta_total": reference["eta_total"],
                "resolution_abs_delta": abs(coarse["eta_total"] - reference["eta_total"]),
                "half_angle_minus5_eta_total": narrow["eta_total"],
                "half_angle_plus5_eta_total": wide["eta_total"],
                "half_angle_span": abs(narrow["eta_total"] - wide["eta_total"]),
            })

            local = evaluate_main(index, centers, frames, tree, sun, surface_count=32, sun_count=16)
            all_field = evaluate_main(
                index, centers, frames, tree, sun, surface_count=32, sun_count=16,
                use_all_candidates=True,
            )
            candidate_pruning_checks.append({
                "mirror_index": index,
                "month": state.month,
                "solar_time": state.solar_time,
                "eta_sb_abs_delta": abs(local["eta_sb"] - all_field["eta_sb"]),
                "eta_trunc_abs_delta": abs(local["eta_trunc"] - all_field["eta_trunc"]),
            })

    total_runtime = time.perf_counter() - start
    sensitivity_resolution = [row["resolution_abs_delta"] for row in sensitivity]
    sensitivity_angle = [row["half_angle_span"] for row in sensitivity]
    pruning_sb = [row["eta_sb_abs_delta"] for row in candidate_pruning_checks]
    pruning_trunc = [row["eta_trunc_abs_delta"] for row in candidate_pruning_checks]
    pair_count = len(selected) * len(states)
    full_pair_count = len(centers) * 60

    result = {
        "schema_version": 1,
        "probe_kind": "method_screening_only",
        "input_rows": len(centers),
        "target_selection": {
            "rule": "3 radial quantile bands x 12 azimuth sectors; one central representative per stratum",
            "target_count": len(selected),
            "target_indices_zero_based": selected,
            "all_1745_mirrors_used_as_occluders": True,
        },
        "solar_states": [asdict(state) for state in states],
        "physics_invariants": {
            "max_sun_vector_norm_error": max(abs(np.linalg.norm(state.sun_vector) - 1) for state in states),
            "max_reflection_direction_residual": max(reflection_residuals),
            "central_ray_receiver_hit_rate": float(np.mean(center_receiver_hits)),
            "all_altitudes_positive": all(state.altitude_rad > 0 for state in states),
            "all_dni_positive": all(state.dni_kw_m2 > 0 for state in states),
            "max_joint_factorization_residual_main": max(
                abs(record["eta_sb"] * record["eta_trunc"] - record["joint_visible_and_received"])
                for record in main_records
            ),
        },
        "M1_coupled_qmc_ray_trace": {
            "surface_samples": 64,
            "sun_samples": 32,
            "records": len(main_records),
            "runtime_seconds": main_runtime,
            "summary": summarize(main_records),
            "mean_candidate_count": float(np.mean([row["candidate_count"] for row in main_records])),
            "estimated_full_60_state_runtime_seconds_same_resolution_cpu": main_runtime * full_pair_count / pair_count,
        },
        "B1_factorized_grid_ray_trace": {
            "surface_grid": "5x5 midpoint",
            "sun_samples": 16,
            "records": len(baseline_records),
            "runtime_seconds": baseline_runtime,
            "summary": summarize(baseline_records),
            "mean_candidate_count": float(np.mean([row["candidate_count"] for row in baseline_records])),
            "estimated_full_60_state_runtime_seconds_same_resolution_cpu": baseline_runtime * full_pair_count / pair_count,
        },
        "method_difference": {
            "mean_eta_total_abs_delta": float(np.mean(np.abs(
                np.asarray([row["eta_total"] for row in main_records])
                - np.asarray([row["eta_total"] for row in baseline_records])
            ))),
            "max_eta_total_abs_delta": float(np.max(np.abs(
                np.asarray([row["eta_total"] for row in main_records])
                - np.asarray([row["eta_total"] for row in baseline_records])
            ))),
        },
        "sensitivity": {
            "cases": sensitivity,
            "resolution_abs_delta_mean": float(np.mean(sensitivity_resolution)),
            "resolution_abs_delta_max": float(np.max(sensitivity_resolution)),
            "half_angle_plus_minus5_span_mean": float(np.mean(sensitivity_angle)),
            "half_angle_plus_minus5_span_max": float(np.max(sensitivity_angle)),
        },
        "candidate_pruning_validation": {
            "radius_m": NEIGHBOR_RADIUS,
            "cases": candidate_pruning_checks,
            "max_eta_sb_abs_delta_vs_all_field": float(np.max(pruning_sb)),
            "max_eta_trunc_abs_delta_vs_all_field": float(np.max(pruning_trunc)),
        },
        "total_probe_runtime_seconds": total_runtime,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "main_runtime_seconds": main_runtime,
        "baseline_runtime_seconds": baseline_runtime,
        "total_probe_runtime_seconds": total_runtime,
        "main_eta_total": result["M1_coupled_qmc_ray_trace"]["summary"]["eta_total"],
        "baseline_eta_total": result["B1_factorized_grid_ray_trace"]["summary"]["eta_total"],
        "max_reflection_residual": result["physics_invariants"]["max_reflection_direction_residual"],
        "max_resolution_delta": result["sensitivity"]["resolution_abs_delta_max"],
        "max_pruning_sb_delta": result["candidate_pruning_validation"]["max_eta_sb_abs_delta_vs_all_field"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
