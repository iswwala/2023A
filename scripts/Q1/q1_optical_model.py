#!/usr/bin/env python3
"""Q1 heliostat-field optical model: approved M1 and usable B1 baseline."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "workspace" / "data_clean" / "Q1" / "heliostat_positions.csv"
ROUND_DIR = ROOT / "results" / "Q1" / "experiments" / "round1"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
FIGURE_DIR = ROUND_DIR / "figures"

LATITUDE = math.radians(39.4)
MIRROR_WIDTH = 6.0
MIRROR_HEIGHT = 6.0
MIRROR_AREA = 36.0
REFLECTIVITY = 0.92
RECEIVER_CENTER = np.array([0.0, 0.0, 80.0], dtype=np.float64)
RECEIVER_RADIUS = 3.5
RECEIVER_Z_MIN = 76.0
RECEIVER_Z_MAX = 84.0
SUN_HALF_ANGLE = 4.65e-3
NEIGHBOR_RADIUS = 60.0
EPS64 = 1e-9
SEEDS = (2023, 2024, 2025, 2026)
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
DAYS_FROM_EQUINOX = (-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275)
SOLAR_TIMES = (9.0, 10.5, 12.0, 13.5, 15.0)


@dataclass(frozen=True)
class SolarState:
    state_index: int
    month: int
    solar_time: float
    declination_rad: float
    hour_angle_rad: float
    altitude_rad: float
    azimuth_rad: float
    dni_kw_m2: float
    sun_vector: tuple[float, float, float]


@dataclass
class TraceResult:
    eta_sb: np.ndarray
    eta_trunc: np.ndarray
    joint_received: np.ndarray


def ensure_output_dirs() -> None:
    for path in (TABLE_DIR, METRIC_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_centers() -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    rows: list[tuple[float, float, float]] = []
    with INPUT.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            ids.append(int(row["heliostat_id"]))
            rows.append((float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))
    centers = np.asarray(rows, dtype=np.float64)
    heliostat_ids = np.asarray(ids, dtype=np.int64)
    if centers.shape != (1745, 3):
        raise ValueError(f"Expected 1745 heliostats, got {centers.shape}")
    if len(np.unique(heliostat_ids)) != len(heliostat_ids):
        raise ValueError("heliostat_id must be unique")
    if not np.allclose(centers[:, 2], 4.0):
        raise ValueError("Q1 requires all installation heights to equal 4 m")
    return heliostat_ids, centers


def build_solar_states() -> list[SolarState]:
    states: list[SolarState] = []
    state_index = 0
    for month, day in enumerate(DAYS_FROM_EQUINOX, start=1):
        declination = math.asin(
            math.sin(2.0 * math.pi * day / 365.0) * math.sin(math.radians(23.45))
        )
        for solar_time in SOLAR_TIMES:
            hour_angle = math.pi / 12.0 * (solar_time - 12.0)
            sin_altitude = (
                math.cos(declination) * math.cos(LATITUDE) * math.cos(hour_angle)
                + math.sin(declination) * math.sin(LATITUDE)
            )
            altitude = math.asin(float(np.clip(sin_altitude, -1.0, 1.0)))
            cos_azimuth = (
                math.sin(declination) - math.sin(altitude) * math.sin(LATITUDE)
            ) / (math.cos(altitude) * math.cos(LATITUDE))
            base_azimuth = math.acos(float(np.clip(cos_azimuth, -1.0, 1.0)))
            azimuth = base_azimuth if hour_angle <= 0 else 2.0 * math.pi - base_azimuth
            sun = np.array(
                [
                    math.cos(altitude) * math.sin(azimuth),
                    math.cos(altitude) * math.cos(azimuth),
                    math.sin(altitude),
                ],
                dtype=np.float64,
            )
            a = 0.4237 - 0.00821 * (6.0 - 3.0) ** 2
            b = 0.5055 + 0.00595 * (6.5 - 3.0) ** 2
            c = 0.2711 + 0.01858 * (2.5 - 3.0) ** 2
            dni = 1.366 * (a + b * math.exp(-c / math.sin(altitude)))
            states.append(
                SolarState(
                    state_index=state_index,
                    month=month,
                    solar_time=solar_time,
                    declination_rad=declination,
                    hour_angle_rad=hour_angle,
                    altitude_rad=altitude,
                    azimuth_rad=azimuth,
                    dni_kw_m2=dni,
                    sun_vector=tuple(float(value) for value in sun),
                )
            )
            state_index += 1
    return states


def mirror_frames(
    centers: np.ndarray, sun: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = RECEIVER_CENTER - centers
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    normals = target + sun[None, :]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    width_axes = np.column_stack((-normals[:, 1], normals[:, 0], np.zeros(len(normals))))
    width_axes /= np.linalg.norm(width_axes, axis=1, keepdims=True)
    height_axes = np.cross(normals, width_axes)
    height_axes /= np.linalg.norm(height_axes, axis=1, keepdims=True)
    return target, normals, width_axes, height_axes


def reflect_directions(sun_directions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    # sun_directions point from the field toward the sun; propagation is their negative.
    dot = np.einsum("...j,...j->...", sun_directions, normals)
    outgoing = -sun_directions + 2.0 * dot[..., None] * normals
    return outgoing / np.linalg.norm(outgoing, axis=-1, keepdims=True)


def atmospheric_efficiency(centers: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(RECEIVER_CENTER - centers, axis=1)
    if np.any(distance > 1000.0):
        raise ValueError("Atmospheric-efficiency formula is only valid for d_HR <= 1000 m")
    return 0.99321 - 0.0001176 * distance + 1.97e-8 * distance**2


def sobol_unit(count: int, dimension: int, seed: int) -> np.ndarray:
    exponent = int(round(math.log2(count)))
    if 2**exponent != count:
        raise ValueError("Sobol sample count must be a power of two")
    return qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(exponent)


def surface_uv_all(count: int, seed: int, heliostat_ids: np.ndarray) -> np.ndarray:
    samples = np.empty((len(heliostat_ids), count, 2), dtype=np.float64)
    for row, heliostat_id in enumerate(heliostat_ids):
        unit = sobol_unit(count, 2, seed + int(heliostat_id) * 17)
        samples[row] = (unit - 0.5) * np.array([MIRROR_WIDTH, MIRROR_HEIGHT])
    return samples


def midpoint_uv(side: int) -> np.ndarray:
    coordinates = ((np.arange(side) + 0.5) / side - 0.5) * MIRROR_WIDTH
    u, v = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack((u.ravel(), v.ravel()))


def sun_disk_directions(sun: np.ndarray, count: int, half_angle: float, seed: int) -> np.ndarray:
    unit = sobol_unit(count, 2, seed)
    reference = np.array([0.0, 0.0, 1.0]) if abs(sun[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    axis_a = np.cross(sun, reference)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(sun, axis_a)
    cos_theta = 1.0 - unit[:, 0] * (1.0 - math.cos(half_angle))
    theta = np.arccos(cos_theta)
    psi = 2.0 * math.pi * unit[:, 1]
    directions = (
        np.cos(theta)[:, None] * sun
        + np.sin(theta)[:, None]
        * (np.cos(psi)[:, None] * axis_a + np.sin(psi)[:, None] * axis_b)
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def build_candidates(centers: np.ndarray, radius: float = NEIGHBOR_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(centers[:, :2])
    lists: list[np.ndarray] = []
    for index, center in enumerate(centers):
        candidates = np.asarray(tree.query_ball_point(center[:2], radius), dtype=np.int64)
        lists.append(candidates[candidates != index])
    width = max(len(values) for values in lists)
    padded = np.zeros((len(centers), width), dtype=np.int64)
    valid = np.zeros((len(centers), width), dtype=bool)
    for index, values in enumerate(lists):
        padded[index, : len(values)] = values
        valid[index, : len(values)] = True
    return padded, valid


def points_from_uv(
    centers: np.ndarray, width_axes: np.ndarray, height_axes: np.ndarray, uv: np.ndarray
) -> np.ndarray:
    if uv.ndim == 2:
        return (
            centers[:, None, :]
            + uv[None, :, :1] * width_axes[:, None, :]
            + uv[None, :, 1:] * height_axes[:, None, :]
        )
    return (
        centers[:, None, :]
        + uv[:, :, :1] * width_axes[:, None, :]
        + uv[:, :, 1:] * height_axes[:, None, :]
    )


def receiver_hits_numpy(points: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intersect BxS points and BxU directions with the finite cylindrical side."""
    dx = directions[:, :, 0]
    dy = directions[:, :, 1]
    a = dx**2 + dy**2
    b = 2.0 * (
        points[:, None, :, 0] * dx[:, :, None]
        + points[:, None, :, 1] * dy[:, :, None]
    )
    c = points[:, None, :, 0] ** 2 + points[:, None, :, 1] ** 2 - RECEIVER_RADIUS**2
    discriminant = b**2 - 4.0 * a[:, :, None] * c
    square_root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = 2.0 * a[:, :, None]
    roots = np.stack(((-b - square_root) / denominator, (-b + square_root) / denominator), axis=-1)
    roots[(discriminant < 0.0)[..., None].repeat(2, axis=-1)] = np.inf
    roots[roots <= EPS64] = np.inf
    lam = np.min(roots, axis=-1)
    z = points[:, None, :, 2] + lam * directions[:, :, None, 2]
    hit = np.isfinite(lam) & (z >= RECEIVER_Z_MIN) & (z <= RECEIVER_Z_MAX)
    return hit, lam


def ray_clear_numpy(
    points: np.ndarray,
    direction: np.ndarray,
    candidate_indices: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    width_axes: np.ndarray,
    height_axes: np.ndarray,
    max_lambda: np.ndarray | None = None,
) -> np.ndarray:
    blocked = np.zeros(len(points), dtype=bool)
    for index in candidate_indices:
        denominator = float(np.dot(direction, normals[index]))
        if abs(denominator) <= EPS64:
            continue
        lam = (centers[index] - points) @ normals[index] / denominator
        eligible = lam > EPS64
        if max_lambda is not None:
            eligible &= lam < max_lambda - EPS64
        if not np.any(eligible):
            continue
        intersection = points[eligible] + lam[eligible, None] * direction
        relative = intersection - centers[index]
        inside = (
            np.abs(relative @ width_axes[index]) <= MIRROR_WIDTH / 2.0 + 1e-10
        ) & (
            np.abs(relative @ height_axes[index]) <= MIRROR_HEIGHT / 2.0 + 1e-10
        )
        blocked[np.flatnonzero(eligible)[inside]] = True
        if np.all(blocked):
            break
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
) -> tuple[int, int, int]:
    visible_count = 0
    received_count = 0
    total = len(points) * len(sun_directions)
    for sun_direction in sun_directions:
        incoming_clear = ray_clear_numpy(
            points, sun_direction, candidate_indices, centers, normals, width_axes, height_axes
        )
        outgoing = reflect_directions(sun_direction[None, :], target_normal[None, :])[0]
        hit, receiver_lambda = receiver_hits_numpy(points[None, :, :], outgoing[None, None, :])
        plane_lambda = np.full(len(points), np.inf)
        if outgoing[2] > EPS64:
            plane_lambda = (RECEIVER_Z_MAX - points[:, 2]) / outgoing[2]
        max_lambda = np.where(hit[0, 0], receiver_lambda[0, 0], plane_lambda)
        outgoing_clear = ray_clear_numpy(
            points,
            outgoing,
            candidate_indices,
            centers,
            normals,
            width_axes,
            height_axes,
            max_lambda,
        )
        visible = incoming_clear & outgoing_clear
        visible_count += int(np.sum(visible))
        received_count += int(np.sum(visible & hit[0, 0]))
    return visible_count, received_count, total


@tf.function(reduce_retracing=True)
def _tensorflow_trace_tensors(
    points: np.ndarray,
    sun_directions: np.ndarray,
    target_normals: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_normals: np.ndarray,
    candidate_width_axes: np.ndarray,
    candidate_height_axes: np.ndarray,
    candidate_valid: np.ndarray,
) -> tuple[Any, Any, Any]:
    dtype = tf.float32
    p = tf.cast(points, dtype)
    sun = tf.cast(sun_directions, dtype)
    target_n = tf.cast(target_normals, dtype)
    cc = tf.cast(candidate_centers, dtype)
    cn = tf.cast(candidate_normals, dtype)
    cw = tf.cast(candidate_width_axes, dtype)
    ch = tf.cast(candidate_height_axes, dtype)
    cmask = tf.cast(candidate_valid, tf.bool)
    eps = tf.constant(1e-6, dtype=dtype)
    edge = tf.constant(3.0 + 1e-5, dtype=dtype)

    relative = p[:, :, None, :] - cc[:, None, :, :]
    numerator = -tf.einsum("bski,bki->bsk", relative, cn)
    denominator_in = tf.einsum("ui,bki->buk", sun, cn)
    safe_in = tf.where(tf.abs(denominator_in) > eps, denominator_in, tf.ones_like(denominator_in))
    lam_in = numerator[:, None, :, :] / safe_in[:, :, None, :]
    base_w = tf.einsum("bski,bki->bsk", relative, cw)
    base_h = tf.einsum("bski,bki->bsk", relative, ch)
    dir_w_in = tf.einsum("ui,bki->buk", sun, cw)
    dir_h_in = tf.einsum("ui,bki->buk", sun, ch)
    hit_w_in = base_w[:, None, :, :] + lam_in * dir_w_in[:, :, None, :]
    hit_h_in = base_h[:, None, :, :] + lam_in * dir_h_in[:, :, None, :]
    valid_in = (
        cmask[:, None, None, :]
        & (tf.abs(denominator_in)[:, :, None, :] > eps)
        & (lam_in > eps)
        & (tf.abs(hit_w_in) <= edge)
        & (tf.abs(hit_h_in) <= edge)
    )
    blocked_in = tf.reduce_any(valid_in, axis=-1)

    sun_dot_normal = tf.einsum("ui,bi->bu", sun, target_n)
    outgoing = -sun[None, :, :] + 2.0 * sun_dot_normal[:, :, None] * target_n[:, None, :]
    outgoing /= tf.linalg.norm(outgoing, axis=-1, keepdims=True)

    dx, dy = outgoing[:, :, 0], outgoing[:, :, 1]
    a = dx**2 + dy**2
    b = 2.0 * (
        p[:, None, :, 0] * dx[:, :, None] + p[:, None, :, 1] * dy[:, :, None]
    )
    c = p[:, None, :, 0] ** 2 + p[:, None, :, 1] ** 2 - RECEIVER_RADIUS**2
    discriminant = b**2 - 4.0 * a[:, :, None] * c
    root_term = tf.sqrt(tf.maximum(discriminant, 0.0))
    root_a = (-b - root_term) / (2.0 * a[:, :, None])
    root_b = (-b + root_term) / (2.0 * a[:, :, None])
    inf = tf.constant(np.inf, dtype=dtype)
    root_a = tf.where((discriminant >= 0.0) & (root_a > eps), root_a, inf)
    root_b = tf.where((discriminant >= 0.0) & (root_b > eps), root_b, inf)
    receiver_lambda = tf.minimum(root_a, root_b)
    receiver_z = p[:, None, :, 2] + receiver_lambda * outgoing[:, :, None, 2]
    receiver_hit = (
        tf.math.is_finite(receiver_lambda)
        & (receiver_z >= RECEIVER_Z_MIN)
        & (receiver_z <= RECEIVER_Z_MAX)
    )
    plane_lambda = tf.where(
        outgoing[:, :, None, 2] > eps,
        (RECEIVER_Z_MAX - p[:, None, :, 2]) / outgoing[:, :, None, 2],
        inf,
    )
    max_lambda = tf.where(receiver_hit, receiver_lambda, plane_lambda)

    denominator_out = tf.einsum("bui,bki->buk", outgoing, cn)
    safe_out = tf.where(tf.abs(denominator_out) > eps, denominator_out, tf.ones_like(denominator_out))
    lam_out = numerator[:, None, :, :] / safe_out[:, :, None, :]
    dir_w_out = tf.einsum("bui,bki->buk", outgoing, cw)
    dir_h_out = tf.einsum("bui,bki->buk", outgoing, ch)
    hit_w_out = base_w[:, None, :, :] + lam_out * dir_w_out[:, :, None, :]
    hit_h_out = base_h[:, None, :, :] + lam_out * dir_h_out[:, :, None, :]
    valid_out = (
        cmask[:, None, None, :]
        & (tf.abs(denominator_out)[:, :, None, :] > eps)
        & (lam_out > eps)
        & (lam_out < max_lambda[:, :, :, None] - eps)
        & (tf.abs(hit_w_out) <= edge)
        & (tf.abs(hit_h_out) <= edge)
    )
    blocked_out = tf.reduce_any(valid_out, axis=-1)
    visible = ~blocked_in & ~blocked_out
    received = visible & receiver_hit
    visible_count = tf.reduce_sum(tf.cast(visible, tf.int64), axis=(1, 2))
    received_count = tf.reduce_sum(tf.cast(received, tf.int64), axis=(1, 2))
    return visible_count, received_count, visible


def _tensorflow_trace_batch(
    points: np.ndarray,
    sun_directions: np.ndarray,
    target_normals: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_normals: np.ndarray,
    candidate_width_axes: np.ndarray,
    candidate_height_axes: np.ndarray,
    candidate_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = _tensorflow_trace_tensors(
        points,
        sun_directions,
        target_normals,
        candidate_centers,
        candidate_normals,
        candidate_width_axes,
        candidate_height_axes,
        candidate_valid,
    )
    return tuple(value.numpy() for value in tensors)


def trace_full_field(
    heliostat_ids: np.ndarray,
    centers: np.ndarray,
    states: list[SolarState],
    candidate_indices: np.ndarray,
    candidate_valid: np.ndarray,
    surface_count: int,
    sun_count: int,
    seed: int,
    half_angle: float,
    backend: str,
    batch_size: int,
) -> TraceResult:
    state_count, mirror_count = len(states), len(centers)
    eta_sb = np.empty((state_count, mirror_count), dtype=np.float64)
    eta_trunc = np.empty_like(eta_sb)
    joint_received = np.empty_like(eta_sb)
    uv = surface_uv_all(surface_count, seed, heliostat_ids)

    for state in states:
        sun = np.asarray(state.sun_vector, dtype=np.float64)
        _, normals, width_axes, height_axes = mirror_frames(centers, sun)
        sun_directions = sun_disk_directions(
            sun, sun_count, half_angle, seed + 100_000 + state.state_index * 101
        )
        points_all = points_from_uv(centers, width_axes, height_axes, uv)
        for start in range(0, mirror_count, batch_size):
            stop = min(start + batch_size, mirror_count)
            batch = np.arange(start, stop)
            if backend == "tensorflow":
                candidates = candidate_indices[batch]
                visible, received, _ = _tensorflow_trace_batch(
                    points_all[batch],
                    sun_directions,
                    normals[batch],
                    centers[candidates],
                    normals[candidates],
                    width_axes[candidates],
                    height_axes[candidates],
                    candidate_valid[batch],
                )
                total = surface_count * sun_count
                eta_sb[state.state_index, batch] = visible / total
                eta_trunc[state.state_index, batch] = np.divide(
                    received,
                    visible,
                    out=np.zeros_like(received, dtype=np.float64),
                    where=visible > 0,
                )
                joint_received[state.state_index, batch] = received / total
            else:
                for index in batch:
                    candidates = candidate_indices[index, candidate_valid[index]]
                    visible, received, total = trace_numpy_target(
                        points_all[index],
                        sun_directions,
                        normals[index],
                        candidates,
                        centers,
                        normals,
                        width_axes,
                        height_axes,
                    )
                    eta_sb[state.state_index, index] = visible / total
                    eta_trunc[state.state_index, index] = received / visible if visible else 0.0
                    joint_received[state.state_index, index] = received / total
    return TraceResult(eta_sb=eta_sb, eta_trunc=eta_trunc, joint_received=joint_received)


def deterministic_components(
    centers: np.ndarray, states: list[SolarState]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta_cos = np.empty((len(states), len(centers)), dtype=np.float64)
    for state in states:
        sun = np.asarray(state.sun_vector)
        _, normals, _, _ = mirror_frames(centers, sun)
        eta_cos[state.state_index] = normals @ sun
    eta_at = atmospheric_efficiency(centers)
    dni = np.asarray([state.dni_kw_m2 for state in states], dtype=np.float64)
    return eta_cos, eta_at, dni


def complete_efficiencies(
    trace: TraceResult, eta_cos: np.ndarray, eta_at: np.ndarray
) -> dict[str, np.ndarray]:
    eta_total = trace.eta_sb * eta_cos * eta_at[None, :] * trace.eta_trunc * REFLECTIVITY
    return {
        "eta_sb": trace.eta_sb,
        "eta_cos": eta_cos,
        "eta_at": np.broadcast_to(eta_at[None, :], eta_cos.shape),
        "eta_trunc": trace.eta_trunc,
        "eta_total": eta_total,
        "joint_received": trace.joint_received,
    }


def aggregate(
    components: dict[str, np.ndarray], states: list[SolarState], dni: np.ndarray
) -> tuple[list[dict[str, float]], dict[str, float], np.ndarray]:
    state_mean = {name: values.mean(axis=1) for name, values in components.items() if name.startswith("eta_")}
    unit_power = dni * state_mean["eta_total"]
    field_power_kw = unit_power * len(components["eta_total"][0]) * MIRROR_AREA
    monthly: list[dict[str, float]] = []
    for month in range(1, 13):
        indices = np.asarray([index for index, state in enumerate(states) if state.month == month])
        monthly.append(
            {
                "month": month,
                "eta_total": float(np.mean(state_mean["eta_total"][indices])),
                "eta_cos": float(np.mean(state_mean["eta_cos"][indices])),
                "eta_sb": float(np.mean(state_mean["eta_sb"][indices])),
                "eta_trunc": float(np.mean(state_mean["eta_trunc"][indices])),
                "unit_area_power_kw_m2": float(np.mean(unit_power[indices])),
                "field_power_mw": float(np.mean(field_power_kw[indices]) / 1000.0),
            }
        )
    annual = {
        "eta_total": float(np.mean(state_mean["eta_total"])),
        "eta_cos": float(np.mean(state_mean["eta_cos"])),
        "eta_sb": float(np.mean(state_mean["eta_sb"])),
        "eta_trunc": float(np.mean(state_mean["eta_trunc"])),
        "field_power_mw": float(np.mean(field_power_kw) / 1000.0),
        "unit_area_power_kw_m2": float(np.mean(unit_power)),
    }
    return monthly, annual, field_power_kw


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_per_state_gzip(
    path: Path,
    heliostat_ids: np.ndarray,
    states: list[SolarState],
    components: dict[str, np.ndarray],
    dni: np.ndarray,
) -> None:
    fields = [
        "heliostat_id",
        "month",
        "solar_time",
        "dni_kw_m2",
        "eta_sb",
        "eta_cos",
        "eta_at",
        "eta_trunc",
        "eta_total",
        "power_kw",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for state in states:
            index = state.state_index
            for mirror_index, heliostat_id in enumerate(heliostat_ids):
                eta_total = float(components["eta_total"][index, mirror_index])
                writer.writerow(
                    {
                        "heliostat_id": int(heliostat_id),
                        "month": state.month,
                        "solar_time": state.solar_time,
                        "dni_kw_m2": float(dni[index]),
                        "eta_sb": float(components["eta_sb"][index, mirror_index]),
                        "eta_cos": float(components["eta_cos"][index, mirror_index]),
                        "eta_at": float(components["eta_at"][index, mirror_index]),
                        "eta_trunc": float(components["eta_trunc"][index, mirror_index]),
                        "eta_total": eta_total,
                        "power_kw": float(dni[index] * MIRROR_AREA * eta_total),
                    }
                )


def run_validation(
    heliostat_ids: np.ndarray,
    centers: np.ndarray,
    states: list[SolarState],
    candidate_indices: np.ndarray,
    candidate_valid: np.ndarray,
) -> dict[str, Any]:
    sun_norm_error = max(abs(np.linalg.norm(state.sun_vector) - 1.0) for state in states)
    reflection_residual = 0.0
    center_hits: list[bool] = []
    for state in states:
        sun = np.asarray(state.sun_vector)
        target, normals, _, _ = mirror_frames(centers, sun)
        outgoing = reflect_directions(
            np.broadcast_to(sun, normals.shape), normals
        )
        reflection_residual = max(reflection_residual, float(np.max(np.linalg.norm(outgoing - target, axis=1))))
        hits, _ = receiver_hits_numpy(centers[:, None, :], outgoing[:, None, :])
        center_hits.extend(hits[:, 0, 0].tolist())

    selected_mirrors = (0, len(centers) // 2, len(centers) - 1)
    state = states[12]
    sun = np.asarray(state.sun_vector)
    _, normals, width_axes, height_axes = mirror_frames(centers, sun)
    uv = surface_uv_all(8, SEEDS[0], heliostat_ids)
    points_all = points_from_uv(centers, width_axes, height_axes, uv)
    sun_directions = sun_disk_directions(sun, 4, SUN_HALF_ANGLE, SEEDS[0] + 100_000)
    tf_visible, tf_received, _ = _tensorflow_trace_batch(
        points_all[list(selected_mirrors)],
        sun_directions,
        normals[list(selected_mirrors)],
        centers[candidate_indices[list(selected_mirrors)]],
        normals[candidate_indices[list(selected_mirrors)]],
        width_axes[candidate_indices[list(selected_mirrors)]],
        height_axes[candidate_indices[list(selected_mirrors)]],
        candidate_valid[list(selected_mirrors)],
    )
    np_counts = []
    for index in selected_mirrors:
        candidates = candidate_indices[index, candidate_valid[index]]
        np_counts.append(
            trace_numpy_target(
                points_all[index], sun_directions, normals[index], candidates,
                centers, normals, width_axes, height_axes,
            )[:2]
        )
    np_counts_array = np.asarray(np_counts, dtype=np.int64)
    tf_counts_array = np.column_stack((tf_visible, tf_received))

    # Validate local pruning against all mirrors on deliberately small but stratified cases.
    pruning_deltas: list[float] = []
    for state_index in (0, 12, 32, 59):
        current = states[state_index]
        sun = np.asarray(current.sun_vector)
        _, normals, width_axes, height_axes = mirror_frames(centers, sun)
        uv_small = surface_uv_all(8, SEEDS[0], heliostat_ids)
        points = points_from_uv(centers, width_axes, height_axes, uv_small)
        directions = sun_disk_directions(sun, 4, SUN_HALF_ANGLE, SEEDS[0] + state_index)
        for index in selected_mirrors:
            local = candidate_indices[index, candidate_valid[index]]
            all_field = np.delete(np.arange(len(centers), dtype=np.int64), index)
            local_counts = trace_numpy_target(
                points[index], directions, normals[index], local,
                centers, normals, width_axes, height_axes,
            )
            all_counts = trace_numpy_target(
                points[index], directions, normals[index], all_field,
                centers, normals, width_axes, height_axes,
            )
            pruning_deltas.append(abs(local_counts[0] / local_counts[2] - all_counts[0] / all_counts[2]))

    validation = {
        "schema_version": 1,
        "solar_state_count": len(states),
        "all_altitudes_positive": all(state.altitude_rad > 0.0 for state in states),
        "all_dni_positive": all(state.dni_kw_m2 > 0.0 for state in states),
        "max_sun_vector_norm_error": float(sun_norm_error),
        "max_center_reflection_residual": float(reflection_residual),
        "center_ray_receiver_hit_rate": float(np.mean(center_hits)),
        "tensorflow_vs_numpy_counts": {
            "numpy": np_counts_array.tolist(),
            "tensorflow": tf_counts_array.tolist(),
            "exact_match": bool(np.array_equal(np_counts_array, tf_counts_array)),
        },
        "candidate_pruning": {
            "radius_m": NEIGHBOR_RADIUS,
            "case_count": len(pruning_deltas),
            "max_eta_sb_abs_delta_vs_all_field": float(max(pruning_deltas)),
            "threshold": 1e-4,
        },
    }
    checks = [
        validation["all_altitudes_positive"],
        validation["all_dni_positive"],
        sun_norm_error <= 1e-12,
        reflection_residual <= 1e-12,
        validation["center_ray_receiver_hit_rate"] == 1.0,
        validation["tensorflow_vs_numpy_counts"]["exact_match"],
        max(pruning_deltas) <= 1e-4,
    ]
    validation["status"] = "PASS" if all(checks) else "FAIL"
    write_json(METRIC_DIR / "q1_validation.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError("Q1 validation failed; see q1_validation.json")
    return validation


def run_baseline(
    heliostat_ids: np.ndarray,
    centers: np.ndarray,
    states: list[SolarState],
    candidate_indices: np.ndarray,
    candidate_valid: np.ndarray,
    eta_cos: np.ndarray,
    eta_at: np.ndarray,
    dni: np.ndarray,
    backend: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]], dict[str, float], float]:
    start_time = time.perf_counter()
    eta_sb = np.empty((len(states), len(centers)), dtype=np.float64)
    eta_trunc = np.empty_like(eta_sb)
    joint_received = np.empty_like(eta_sb)
    uv = midpoint_uv(5)
    for state in states:
        sun = np.asarray(state.sun_vector)
        _, normals, width_axes, height_axes = mirror_frames(centers, sun)
        points = points_from_uv(centers, width_axes, height_axes, uv)
        sun_directions = sun_disk_directions(
            sun, 16, SUN_HALF_ANGLE, SEEDS[0] + 200_000 + state.state_index * 101
        )
        outgoing_directions = reflect_directions(
            np.broadcast_to(sun_directions[None, :, :], (len(centers), 16, 3)),
            np.broadcast_to(normals[:, None, :], (len(centers), 16, 3)),
        )
        for start in range(0, len(centers), batch_size):
            stop = min(start + batch_size, len(centers))
            batch = np.arange(start, stop)
            if backend == "tensorflow":
                candidates = candidate_indices[batch]
                visible_count, _, visible_mask = _tensorflow_trace_batch(
                    points[batch],
                    sun[None, :],
                    normals[batch],
                    centers[candidates],
                    normals[candidates],
                    width_axes[candidates],
                    height_axes[candidates],
                    candidate_valid[batch],
                )
                center_visible = visible_mask[:, 0, :]
            else:
                center_visible = np.empty((len(batch), 25), dtype=bool)
                visible_count = np.empty(len(batch), dtype=np.int64)
                for local_row, index in enumerate(batch):
                    candidates = candidate_indices[index, candidate_valid[index]]
                    incoming_clear = ray_clear_numpy(
                        points[index], sun, candidates, centers, normals, width_axes, height_axes
                    )
                    outgoing_center = reflect_directions(sun[None, :], normals[index][None, :])[0]
                    center_hit, receiver_lambda = receiver_hits_numpy(
                        points[index][None, :, :], outgoing_center[None, None, :]
                    )
                    plane_lambda = (RECEIVER_Z_MAX - points[index, :, 2]) / outgoing_center[2]
                    max_lambda = np.where(center_hit[0, 0], receiver_lambda[0, 0], plane_lambda)
                    outgoing_clear = ray_clear_numpy(
                        points[index], outgoing_center, candidates, centers, normals,
                        width_axes, height_axes, max_lambda,
                    )
                    center_visible[local_row] = incoming_clear & outgoing_clear
                    visible_count[local_row] = int(np.sum(center_visible[local_row]))
            receiver_hit, _ = receiver_hits_numpy(points[batch], outgoing_directions[batch])
            received_count = np.sum(receiver_hit & center_visible[:, None, :], axis=(1, 2))
            eta_sb[state.state_index, batch] = visible_count / 25.0
            denominator = visible_count * 16
            eta_trunc[state.state_index, batch] = np.divide(
                received_count,
                denominator,
                out=np.zeros_like(received_count, dtype=np.float64),
                where=denominator > 0,
            )
            joint_received[state.state_index, batch] = received_count / (25.0 * 16.0)
    trace = TraceResult(
        eta_sb=eta_sb,
        eta_trunc=eta_trunc,
        joint_received=joint_received,
    )
    components = complete_efficiencies(trace, eta_cos, eta_at)
    monthly, annual, _ = aggregate(components, states, dni)
    write_csv(TABLE_DIR / "q1_baseline_monthly.csv", monthly)
    write_csv(TABLE_DIR / "q1_baseline_annual.csv", [annual])
    return components, monthly, annual, time.perf_counter() - start_time


def run_m1_resolution(
    label: str,
    heliostat_ids: np.ndarray,
    centers: np.ndarray,
    states: list[SolarState],
    candidate_indices: np.ndarray,
    candidate_valid: np.ndarray,
    eta_cos: np.ndarray,
    eta_at: np.ndarray,
    dni: np.ndarray,
    surface_count: int,
    sun_count: int,
    half_angle: float,
    backend: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    start_time = time.perf_counter()
    trace_records: list[TraceResult] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_start = time.perf_counter()
        trace = trace_full_field(
            heliostat_ids, centers, states, candidate_indices, candidate_valid,
            surface_count, sun_count, seed, half_angle, backend, batch_size,
        )
        components = complete_efficiencies(trace, eta_cos, eta_at)
        _, annual, _ = aggregate(components, states, dni)
        factorization_residual = float(
            np.max(np.abs(trace.eta_sb * trace.eta_trunc - trace.joint_received))
        )
        seed_summaries.append(
            {
                "seed": seed,
                "resolution": label,
                "annual": annual,
                "max_joint_factorization_residual": factorization_residual,
                "runtime_seconds": time.perf_counter() - seed_start,
            }
        )
        trace_records.append(trace)
    eta_sb_mean = np.mean(np.stack([record.eta_sb for record in trace_records]), axis=0)
    joint_mean = np.mean(np.stack([record.joint_received for record in trace_records]), axis=0)
    eta_trunc_pooled = np.divide(
        joint_mean,
        eta_sb_mean,
        out=np.zeros_like(joint_mean),
        where=eta_sb_mean > 0,
    )
    pooled_trace = TraceResult(
        eta_sb=eta_sb_mean,
        eta_trunc=eta_trunc_pooled,
        joint_received=joint_mean,
    )
    pooled_components = complete_efficiencies(pooled_trace, eta_cos, eta_at)
    return pooled_components, seed_summaries, time.perf_counter() - start_time


def convergence_metrics(
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    states: list[SolarState],
    dni: np.ndarray,
    fine_seed_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    coarse_monthly, coarse_annual, _ = aggregate(coarse, states, dni)
    fine_monthly, fine_annual, _ = aggregate(fine, states, dni)
    annual_eta_delta = abs(fine_annual["eta_total"] - coarse_annual["eta_total"])
    annual_power_relative_delta = abs(
        fine_annual["field_power_mw"] - coarse_annual["field_power_mw"]
    ) / fine_annual["field_power_mw"]
    monthly_power_relative = [
        abs(fine_row["unit_area_power_kw_m2"] - coarse_row["unit_area_power_kw_m2"])
        / fine_row["unit_area_power_kw_m2"]
        for coarse_row, fine_row in zip(coarse_monthly, fine_monthly)
    ]
    seed_annual_eta = [row["annual"]["eta_total"] for row in fine_seed_summaries]
    seed_annual_power = [row["annual"]["field_power_mw"] for row in fine_seed_summaries]
    passed = (
        annual_eta_delta <= 0.002
        and annual_power_relative_delta <= 0.003
        and max(monthly_power_relative) <= 0.005
    )
    return {
        "schema_version": 1,
        "coarse": {"surface_samples": 32, "sun_samples": 16, "annual": coarse_annual},
        "fine": {"surface_samples": 64, "sun_samples": 32, "annual": fine_annual},
        "annual_eta_total_abs_delta": float(annual_eta_delta),
        "annual_field_power_relative_delta": float(annual_power_relative_delta),
        "monthly_unit_power_relative_delta": [float(value) for value in monthly_power_relative],
        "max_monthly_unit_power_relative_delta": float(max(monthly_power_relative)),
        "thresholds": {
            "annual_eta_total_abs": 0.002,
            "annual_field_power_relative": 0.003,
            "monthly_unit_power_relative": 0.005,
        },
        "fine_seed_dispersion": {
            "annual_eta_total_values": seed_annual_eta,
            "annual_eta_total_std": float(np.std(seed_annual_eta)),
            "annual_eta_total_range": float(np.ptp(seed_annual_eta)),
            "annual_field_power_mw_values": seed_annual_power,
            "annual_field_power_mw_std": float(np.std(seed_annual_power)),
            "annual_field_power_mw_range": float(np.ptp(seed_annual_power)),
        },
        "status": "PASS" if passed else "CONDITIONAL",
        "requires_128x64": not passed,
    }


def method_comparison(
    main: dict[str, np.ndarray], baseline: dict[str, np.ndarray], states: list[SolarState], dni: np.ndarray
) -> dict[str, Any]:
    main_monthly, main_annual, _ = aggregate(main, states, dni)
    base_monthly, base_annual, _ = aggregate(baseline, states, dni)
    eta_delta = np.abs(main["eta_total"] - baseline["eta_total"])
    return {
        "schema_version": 1,
        "annual_absolute_differences": {
            key: float(abs(main_annual[key] - base_annual[key])) for key in main_annual
        },
        "monthly": [
            {
                "month": month,
                **{
                    f"{key}_abs_delta": float(abs(main_monthly[month - 1][key] - base_monthly[month - 1][key]))
                    for key in main_monthly[month - 1]
                    if key != "month"
                },
            }
            for month in range(1, 13)
        ],
        "per_heliostat_state_eta_total_abs_delta": {
            "mean": float(np.mean(eta_delta)),
            "p95": float(np.quantile(eta_delta, 0.95)),
            "max": float(np.max(eta_delta)),
            "unique_main_rounded_6dp": int(len(np.unique(np.round(main["eta_total"], 6)))),
            "mass_main_at_zero": float(np.mean(main["eta_total"] <= 1e-12)),
        },
    }


def make_diagnostic_figures(
    main_monthly: list[dict[str, float]],
    baseline_monthly: list[dict[str, float]],
    convergence: dict[str, Any],
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROUND_DIR / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = np.arange(1, 13)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    axes[0].plot(months, [row["eta_total"] for row in main_monthly], "o-", label="M1")
    axes[0].plot(months, [row["eta_total"] for row in baseline_monthly], "s--", label="B1")
    axes[0].set_ylabel("Mean optical efficiency")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(months, [row["unit_area_power_kw_m2"] for row in main_monthly], "o-", label="M1")
    axes[1].plot(months, [row["unit_area_power_kw_m2"] for row in baseline_monthly], "s--", label="B1")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Unit-area power (kW/m2)")
    axes[1].set_xticks(months)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    comparison_path = FIGURE_DIR / "q1_method_comparison_diagnostic.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    values = np.asarray(convergence["monthly_unit_power_relative_delta"]) * 100.0
    ax.bar(months, values, color="#3A6EA5")
    ax.axhline(0.5, color="#B33A3A", linestyle="--", linewidth=1.1, label="0.5% threshold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Coarse-fine relative delta (%)")
    ax.set_xticks(months)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    convergence_path = FIGURE_DIR / "q1_convergence_diagnostic.png"
    fig.savefig(convergence_path, dpi=180)
    plt.close(fig)
    return [str(comparison_path.relative_to(ROOT)), str(convergence_path.relative_to(ROOT))]


def tensorflow_environment() -> dict[str, Any]:
    try:
        import tensorflow as tf

        return {
            "tensorflow": tf.__version__,
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
        }
    except Exception as error:  # pragma: no cover - environment evidence
        return {"tensorflow_error": repr(error), "physical_gpus": [], "logical_gpus": []}


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_output_dirs()
    overall_start = time.perf_counter()
    heliostat_ids, centers = load_centers()
    states = build_solar_states()
    candidate_indices, candidate_valid = build_candidates(centers)
    eta_cos, eta_at, dni = deterministic_components(centers, states)
    validation = run_validation(heliostat_ids, centers, states, candidate_indices, candidate_valid)

    baseline, baseline_monthly, baseline_annual, baseline_runtime = run_baseline(
        heliostat_ids, centers, states, candidate_indices, candidate_valid,
        eta_cos, eta_at, dni, args.backend, args.batch_size,
    )
    coarse, coarse_seed_summaries, coarse_runtime = run_m1_resolution(
        "32x16", heliostat_ids, centers, states, candidate_indices, candidate_valid,
        eta_cos, eta_at, dni, 32, 16, SUN_HALF_ANGLE, args.backend, args.batch_size,
    )
    fine, fine_seed_summaries, fine_runtime = run_m1_resolution(
        "64x32", heliostat_ids, centers, states, candidate_indices, candidate_valid,
        eta_cos, eta_at, dni, 64, 32, SUN_HALF_ANGLE, args.backend, args.batch_size,
    )
    main_monthly, main_annual, _ = aggregate(fine, states, dni)
    write_csv(TABLE_DIR / "q1_main_monthly.csv", main_monthly)
    write_csv(TABLE_DIR / "q1_main_annual.csv", [main_annual])
    write_per_state_gzip(TABLE_DIR / "q1_main_per_heliostat_state.csv.gz", heliostat_ids, states, fine, dni)

    convergence = convergence_metrics(coarse, fine, states, dni, fine_seed_summaries)
    write_json(METRIC_DIR / "q1_convergence.json", convergence)
    comparison = method_comparison(fine, baseline, states, dni)
    write_json(METRIC_DIR / "q1_method_comparison.json", comparison)

    sensitivity_start = time.perf_counter()
    narrow, narrow_seed_summaries, narrow_runtime = run_m1_resolution(
        "64x32_half_angle_minus5", heliostat_ids, centers, states,
        candidate_indices, candidate_valid, eta_cos, eta_at, dni,
        64, 32, SUN_HALF_ANGLE * 0.95, args.backend, args.batch_size,
    )
    wide, wide_seed_summaries, wide_runtime = run_m1_resolution(
        "64x32_half_angle_plus5", heliostat_ids, centers, states,
        candidate_indices, candidate_valid, eta_cos, eta_at, dni,
        64, 32, SUN_HALF_ANGLE * 1.05, args.backend, args.batch_size,
    )
    _, narrow_annual, _ = aggregate(narrow, states, dni)
    _, wide_annual, _ = aggregate(wide, states, dni)

    # Month-day weighting is derived without changing the 60-state optical solution.
    monthly_power = np.asarray([row["field_power_mw"] for row in main_monthly])
    day_weighted_power = float(np.average(monthly_power, weights=MONTH_DAYS))
    equal_weighted_power = main_annual["field_power_mw"]
    narrow_relative = abs(narrow_annual["field_power_mw"] - equal_weighted_power) / equal_weighted_power
    wide_relative = abs(wide_annual["field_power_mw"] - equal_weighted_power) / equal_weighted_power
    weighting_relative = abs(day_weighted_power - equal_weighted_power) / equal_weighted_power
    sensitivity_trigger = max(narrow_relative, wide_relative, weighting_relative) > 0.01
    sensitivity = {
        "schema_version": 1,
        "month_weighting": {
            "equal_month_field_power_mw": equal_weighted_power,
            "month_day_weighted_field_power_mw": day_weighted_power,
            "relative_delta": weighting_relative,
        },
        "pillbox_half_angle": {
            "minus_5_rad": SUN_HALF_ANGLE * 0.95,
            "core_rad": SUN_HALF_ANGLE,
            "plus_5_rad": SUN_HALF_ANGLE * 1.05,
            "minus_5_field_power_mw": narrow_annual["field_power_mw"],
            "core_field_power_mw": equal_weighted_power,
            "plus_5_field_power_mw": wide_annual["field_power_mw"],
            "minus_5_relative_delta": narrow_relative,
            "plus_5_relative_delta": wide_relative,
            "seed_summaries": {
                "minus_5": narrow_seed_summaries,
                "plus_5": wide_seed_summaries,
            },
        },
        "threshold_relative": 0.01,
        "human_assumption_review_triggered": sensitivity_trigger,
        "status": "CONDITIONAL" if sensitivity_trigger else "PASS",
        "runtime_seconds": time.perf_counter() - sensitivity_start,
    }
    write_json(METRIC_DIR / "q1_sensitivity.json", sensitivity)
    figure_files = make_diagnostic_figures(main_monthly, baseline_monthly, convergence)

    warnings: list[str] = []
    if convergence["status"] != "PASS":
        warnings.append("M1 32x16 to 64x32 convergence thresholds were not met; 128x64 is required")
    if sensitivity_trigger:
        warnings.append("A sensitivity scenario changed annual power by more than 1%; return to the human assumption decision point")
    tf_environment = tensorflow_environment()
    if args.backend == "tensorflow" and not tf_environment.get("physical_gpus"):
        warnings.append("TensorFlow backend ran without a visible GPU")

    summary = {
        "schema_version": 1,
        "question": "Q1",
        "round": "round1",
        "implementation_target": "python",
        "random_seed": list(SEEDS),
        "approved_decision_id": "q1_method_choice",
        "methods": [
            {
                "method_id": "M1",
                "role": "main_candidate",
                "script": "scripts/Q1/q1_optical_model.py",
                "status": "success" if convergence["status"] == "PASS" else "conditional",
                "execution_time_seconds": coarse_runtime + fine_runtime + narrow_runtime + wide_runtime,
                "input_files": [str(INPUT.relative_to(ROOT))],
                "output_files": [
                    "results/Q1/experiments/round1/tables/q1_main_monthly.csv",
                    "results/Q1/experiments/round1/tables/q1_main_annual.csv",
                    "results/Q1/experiments/round1/tables/q1_main_per_heliostat_state.csv.gz",
                    "results/Q1/experiments/round1/metrics/q1_convergence.json",
                    "results/Q1/experiments/round1/metrics/q1_sensitivity.json",
                ],
                "figure_files": figure_files,
                "metrics_summary": main_annual,
                "warnings": warnings,
                "errors": [],
            },
            {
                "method_id": "B1",
                "role": "usable_baseline",
                "script": "scripts/Q1/q1_optical_model.py",
                "status": "success",
                "execution_time_seconds": baseline_runtime,
                "input_files": [str(INPUT.relative_to(ROOT))],
                "output_files": [
                    "results/Q1/experiments/round1/tables/q1_baseline_monthly.csv",
                    "results/Q1/experiments/round1/tables/q1_baseline_annual.csv",
                ],
                "figure_files": figure_files[:1],
                "metrics_summary": baseline_annual,
                "warnings": ["B1 remains a low-resolution usable baseline and cannot replace M1"],
                "errors": [],
            },
        ],
        "comparison": comparison,
        "output_degeneracy": comparison["per_heliostat_state_eta_total_abs_delta"],
        "validation": validation,
        "fallback_trigger": {
            "fallback_id": None,
            "condition": "M1未达到收敛时提高同一方法分辨率并返回实验判断点；B1不得自动替代M1",
            "observed": convergence["status"] != "PASS",
            "evidence": "results/Q1/experiments/round1/metrics/q1_convergence.json",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            **tf_environment,
            "backend": args.backend,
            "batch_size": args.batch_size,
            "peak_gpu_memory": "unavailable_from_portable_tensorflow_api",
        },
        "warnings": warnings,
        "errors": [],
        "execution_time_seconds": time.perf_counter() - overall_start,
    }
    write_json(ROUND_DIR / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "all"), nargs="?", default="validate")
    parser.add_argument("--backend", choices=("tensorflow", "numpy"), default="tensorflow")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if args.command == "validate":
        heliostat_ids, centers = load_centers()
        states = build_solar_states()
        candidate_indices, candidate_valid = build_candidates(centers)
        result = run_validation(heliostat_ids, centers, states, candidate_indices, candidate_valid)
    else:
        result = run_all(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
