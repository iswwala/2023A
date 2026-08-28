#!/usr/bin/env python3
"""Freeze signed-off Q1 numerical claims from canonical result artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUND_DIR = ROOT / "results" / "Q1" / "experiments" / "round1"
REPORT_DIR = ROOT / "results" / "Q1" / "reports"
OUTPUT = REPORT_DIR / "frozen_numbers.json"
DECISION_ID = "q1_package_signoff"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    monthly_path = ROUND_DIR / "tables" / "q1_main_monthly.csv"
    annual_path = ROUND_DIR / "tables" / "q1_main_annual.csv"
    baseline_path = ROUND_DIR / "tables" / "q1_baseline_annual.csv"
    convergence_path = ROUND_DIR / "metrics" / "q1_convergence.json"
    sensitivity_path = ROUND_DIR / "metrics" / "q1_sensitivity.json"
    validation_path = ROUND_DIR / "metrics" / "q1_validation.json"
    comparison_path = ROUND_DIR / "metrics" / "q1_method_comparison.json"
    input_path = ROOT / "workspace" / "data_clean" / "Q1" / "heliostat_positions.csv"
    script_path = ROOT / "scripts" / "Q1" / "q1_optical_model.py"
    source_paths = [
        monthly_path,
        annual_path,
        baseline_path,
        convergence_path,
        sensitivity_path,
        validation_path,
        comparison_path,
        input_path,
        script_path,
    ]
    monthly = read_csv(monthly_path)
    annual = read_csv(annual_path)[0]
    baseline = read_csv(baseline_path)[0]
    convergence = read_json(convergence_path)
    sensitivity = read_json(sensitivity_path)
    validation = read_json(validation_path)
    comparison = read_json(comparison_path)
    frozen_at = datetime.now().astimezone().isoformat(timespec="seconds")
    claims: list[dict] = []

    def add(claim_id: str, value, unit: str, source_file: Path, locator: str) -> None:
        claims.append(
            {
                "claim_id": claim_id,
                "value": value,
                "unit": unit,
                "source_file": str(source_file.relative_to(ROOT)),
                "source_locator": locator,
                "frozen_at": frozen_at,
                "frozen_by_skill": "solution-package-builder",
                "decision_id": DECISION_ID,
            }
        )

    add("q1_heliostat_count", len(read_csv(input_path)), "mirrors", input_path, "row_count")
    add("q1_solar_state_count", validation["solar_state_count"], "states", validation_path, "$.solar_state_count")
    add("q1_mirror_width", 6.0, "m", script_path, "MIRROR_WIDTH")
    add("q1_mirror_height", 6.0, "m", script_path, "MIRROR_HEIGHT")
    add("q1_mirror_area", 36.0, "m^2", script_path, "MIRROR_AREA")
    add("q1_total_mirror_area", len(read_csv(input_path)) * 36.0, "m^2", input_path, "row_count * 36 m^2")
    add("q1_receiver_center_height", 76.0, "m", script_path, "RECEIVER_CENTER[2]")
    add("q1_receiver_radius", 3.5, "m", script_path, "RECEIVER_RADIUS")
    add("q1_receiver_z_min", 72.0, "m", script_path, "RECEIVER_Z_MIN")
    add("q1_receiver_z_max", 80.0, "m", script_path, "RECEIVER_Z_MAX")
    add("q1_tower_radius", 3.5, "m", script_path, "TOWER_RADIUS")
    add("q1_tower_z_min", 0.0, "m", script_path, "TOWER_Z_MIN")
    add("q1_tower_z_max", 72.0, "m", script_path, "TOWER_Z_MAX")
    add("q1_receiver_shadow_excluded", True, "bool", validation_path, "$.tower_shadow_geometry.receiver_shadow_excluded")
    add("q1_reflectivity", 0.92, "1", script_path, "REFLECTIVITY")
    add("q1_sun_half_angle", 0.00465, "rad", script_path, "SUN_HALF_ANGLE")

    units = {
        "eta_total": "1",
        "eta_cos": "1",
        "eta_sb": "1",
        "eta_trunc": "1",
        "unit_area_power_kw_m2": "kW/m^2",
        "field_power_mw": "MW",
    }
    for row in monthly:
        month = int(row["month"])
        for field, unit in units.items():
            add(
                f"q1_month_{month:02d}_{field}",
                float(row[field]),
                unit,
                monthly_path,
                f"row[month={month}].{field}",
            )
    for field, unit in units.items():
        add(f"q1_annual_{field}", float(annual[field]), unit, annual_path, f"row[1].{field}")
        add(f"q1_baseline_annual_{field}", float(baseline[field]), unit, baseline_path, f"row[1].{field}")
    for field, value in comparison["annual_absolute_differences"].items():
        add(
            f"q1_main_baseline_abs_delta_{field}",
            value,
            units[field],
            comparison_path,
            f"$.annual_absolute_differences.{field}",
        )

    add("q1_convergence_eta_abs_delta", convergence["annual_eta_total_abs_delta"], "1", convergence_path, "$.annual_eta_total_abs_delta")
    add("q1_convergence_power_relative_delta", convergence["annual_field_power_relative_delta"], "1", convergence_path, "$.annual_field_power_relative_delta")
    add("q1_convergence_monthly_max_relative_delta", convergence["max_monthly_unit_power_relative_delta"], "1", convergence_path, "$.max_monthly_unit_power_relative_delta")
    add("q1_seed_eta_std", convergence["fine_seed_dispersion"]["annual_eta_total_std"], "1", convergence_path, "$.fine_seed_dispersion.annual_eta_total_std")
    add("q1_seed_eta_range", convergence["fine_seed_dispersion"]["annual_eta_total_range"], "1", convergence_path, "$.fine_seed_dispersion.annual_eta_total_range")
    add("q1_seed_power_std", convergence["fine_seed_dispersion"]["annual_field_power_mw_std"], "MW", convergence_path, "$.fine_seed_dispersion.annual_field_power_mw_std")
    add("q1_seed_power_range", convergence["fine_seed_dispersion"]["annual_field_power_mw_range"], "MW", convergence_path, "$.fine_seed_dispersion.annual_field_power_mw_range")
    add("q1_weighting_power_relative_delta", sensitivity["month_weighting"]["relative_delta"], "1", sensitivity_path, "$.month_weighting.relative_delta")
    add("q1_weighting_day_power", sensitivity["month_weighting"]["month_day_weighted_field_power_mw"], "MW", sensitivity_path, "$.month_weighting.month_day_weighted_field_power_mw")
    add("q1_half_angle_minus5_power", sensitivity["pillbox_half_angle"]["minus_5_field_power_mw"], "MW", sensitivity_path, "$.pillbox_half_angle.minus_5_field_power_mw")
    add("q1_half_angle_plus5_power", sensitivity["pillbox_half_angle"]["plus_5_field_power_mw"], "MW", sensitivity_path, "$.pillbox_half_angle.plus_5_field_power_mw")
    add("q1_half_angle_minus5_relative_delta", sensitivity["pillbox_half_angle"]["minus_5_relative_delta"], "1", sensitivity_path, "$.pillbox_half_angle.minus_5_relative_delta")
    add("q1_half_angle_plus5_relative_delta", sensitivity["pillbox_half_angle"]["plus_5_relative_delta"], "1", sensitivity_path, "$.pillbox_half_angle.plus_5_relative_delta")
    add("q1_sun_vector_max_norm_error", validation["max_sun_vector_norm_error"], "1", validation_path, "$.max_sun_vector_norm_error")
    add("q1_reflection_max_residual", validation["max_center_reflection_residual"], "1", validation_path, "$.max_center_reflection_residual")
    add("q1_center_ray_hit_rate", validation["center_ray_receiver_hit_rate"], "1", validation_path, "$.center_ray_receiver_hit_rate")
    add("q1_candidate_pruning_eta_sb_max_delta", validation["candidate_pruning"]["max_eta_sb_abs_delta_vs_all_field"], "1", validation_path, "$.candidate_pruning.max_eta_sb_abs_delta_vs_all_field")

    payload = {
        "schema_version": 1,
        "question": "Q1",
        "frozen_at": frozen_at,
        "frozen_by_skill": "solution-package-builder",
        "decision_id": DECISION_ID,
        "claim_scope_decision_id": "q1_claim_scope",
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "claims": claims,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(claims)} Q1 claims to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
