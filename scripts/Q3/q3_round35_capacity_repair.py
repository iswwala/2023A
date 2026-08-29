#!/usr/bin/env python3
"""Interpolate two feasible-search stages to repair formal Q3 power margin."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE_1 = ROOT / "results/Q3/experiments/round33_full_range_continuous_refine/tables/q3_continuous_stage_1_layout.csv"
STAGE_2 = ROOT / "results/Q3/experiments/round33_full_range_continuous_refine/tables/q3_continuous_stage_2_layout.csv"
ROUND_DIR = ROOT / "results/Q3/experiments/round35_capacity_repair"
OUTPUT = ROUND_DIR / "tables/q3_capacity_repaired_layout.csv"
ALPHA = 0.20


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    first, second = read(STAGE_1), read(STAGE_2)
    if len(first) != len(second):
        raise RuntimeError("stage layouts have different mirror counts")
    rows = []
    for left, right in zip(first, second):
        if (left["heliostat_id"], left["x_m"], left["y_m"]) != (
                right["heliostat_id"], right["x_m"], right["y_m"]):
            raise RuntimeError("stage layouts do not share identical mirror identities and positions")
        row = dict(left)
        for field in ("z_m", "width_m", "height_m"):
            row[field] = (1.0 - ALPHA) * float(left[field]) + ALPHA * float(right[field])
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1, "question": "Q3", "round": ROUND_DIR.name,
        "status": "candidate_generated", "approved_decision_id": "q3_layout_count_joint_revision",
        "method_id": "M3G_formal_capacity_repair", "source_stage_1": str(STAGE_1.relative_to(ROOT)),
        "source_stage_2": str(STAGE_2.relative_to(ROOT)), "interpolation_alpha": ALPHA,
        "mirror_count": len(rows), "output": str(OUTPUT.relative_to(ROOT)),
        "formal_recheck_required": True,
    }
    (ROUND_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
