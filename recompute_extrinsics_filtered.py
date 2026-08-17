from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from extr_calib_d435_cube_cv2_apriltag_grid import (
    MIN_SAMPLES_TO_SOLVE,
    CalibrationSample,
    sample_to_dict,
    serialize_solution,
    solve_with_outlier_rejection,
)


def parse_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def sample_from_dict(data: dict) -> CalibrationSample:
    return CalibrationSample(
        index=int(data["index"]),
        timestamp=float(data["timestamp"]),
        pair_skew_s=float(data["pair_skew_s"]),
        third_view_frame_index=int(data["third_view_frame_index"]),
        thumb_web_frame_index=int(data["thumb_web_frame_index"]),
        T_third_view_cube=np.asarray(
            data["T_third_view_cam_cube"], dtype=np.float64
        ).reshape(4, 4),
        T_thumb_web_charuco=np.asarray(
            data["T_thumb_web_cam_charuco"], dtype=np.float64
        ).reshape(4, 4),
        cube_tags=int(data["cube_tags"]),
        cube_reproj_error_px=float(data["cube_reproj_error_px"]),
        charuco_corners=int(data["charuco_corners"]),
        charuco_reproj_error_px=float(data["charuco_reproj_error_px"]),
        third_view_image_path=str(data["third_view_image_path"]),
        thumb_web_image_path=str(data["thumb_web_image_path"]),
        capture_mode=str(data["capture_mode"]),
    )


def default_output_path(source: Path) -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    return source.with_name(f"{source.stem}_filtered_recompute_{stamp}.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute an extrinsic result after excluding reviewed samples."
    )
    parser.add_argument("source", type=Path, help="Existing extrinsic-result YAML")
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated original sample indices to exclude before solving",
    )
    parser.add_argument(
        "--motion-blur-exclude",
        default="",
        help="Subset of --exclude confirmed to contain motion blur",
    )
    parser.add_argument(
        "--reason",
        default="manual_quality_review",
        help="Short audit note describing the pre-solve exclusion",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        source_data = yaml.safe_load(stream)
    if not isinstance(source_data, dict) or not isinstance(
        source_data.get("samples"), list
    ):
        raise ValueError(f"Missing samples list in {source}")

    samples = [sample_from_dict(item) for item in source_data["samples"]]
    source_indices = {sample.index for sample in samples}
    excluded = parse_indices(args.exclude)
    blur_excluded = parse_indices(args.motion_blur_exclude)
    missing = sorted(set(excluded) - source_indices)
    if missing:
        raise ValueError(f"Requested sample indices not present: {missing}")
    if not set(blur_excluded).issubset(excluded):
        raise ValueError("--motion-blur-exclude must be a subset of --exclude")

    excluded_set = set(excluded)
    retained = [sample for sample in samples if sample.index not in excluded_set]
    if len(retained) < MIN_SAMPLES_TO_SOLVE:
        raise ValueError(
            f"Only {len(retained)} samples remain; need at least {MIN_SAMPLES_TO_SOLVE}"
        )

    solution = solve_with_outlier_rejection(retained)
    serialized_solution = serialize_solution(solution)
    solver_outliers = list(serialized_solution["outlier_indices"])
    effective_outliers = sorted(set(excluded) | set(solver_outliers))
    serialized_solution["outlier_indices"] = effective_outliers

    output_data = deepcopy(source_data)
    output_data["created_at"] = datetime.now().isoformat(timespec="seconds")
    output_data["capture"]["num_source_samples"] = len(samples)
    output_data["capture"]["num_raw_samples"] = len(retained)
    output_data["recomputation"] = {
        "source_result_yaml": str(source),
        "source_sample_count": len(samples),
        "pre_solve_excluded_indices": excluded,
        "motion_blur_excluded_indices": blur_excluded,
        "pre_solve_exclusion_reason": args.reason,
        "solver_outlier_indices_after_prefilter": solver_outliers,
        "effective_outlier_indices": effective_outliers,
        "original_images_deleted": False,
        "note": (
            "Raw images were preserved for auditability; exclusions only affect "
            "this recomputed result."
        ),
    }
    output_data["solution"] = serialized_solution
    output_data["samples"] = [sample_to_dict(sample) for sample in retained]

    output = (args.output or default_output_path(source)).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output_data, stream, sort_keys=False)

    print(f"[INFO] Saved {output}")
    print(f"[INFO] Source/retained: {len(samples)}/{len(retained)}")
    print(f"[INFO] Pre-solve excluded: {excluded}")
    print(f"[INFO] Motion-blur excluded: {blur_excluded}")
    print(f"[INFO] Solver outliers after prefilter: {solver_outliers}")
    print("[RESULT] T_cube_thumb_web_cam:")
    print(solution["T_cube_thumb_web_cam"])
    print(f"[DIAGNOSTICS] rotation residual deg={solution['residual_rot_deg']}")
    print(f"[DIAGNOSTICS] translation residual m={solution['residual_trans_m']}")
    print(
        "[DIAGNOSTICS] jacobian rank="
        f"{solution['jacobian_rank']} condition={solution['jacobian_condition']:.3e}"
    )


if __name__ == "__main__":
    main()
