#!/usr/bin/env python3
"""Move only Wuji finger2/index to a collision-checked open calibration pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from FingerEyeRW.PolicyReview.hand_qpos8_preview import isolated_wuji_operation
from FingerEyeRW.PolicyReview.v2 import load_config


CONFIG = Path(
    "/home/CNF2025915223/桌面/FingerEyeV2/"
    "FingerEyeRW/PolicyReview/configs/v2.yaml"
)
INDEX_CLEAR_QPOS4 = np.zeros(4, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thumb-qpos4", type=float, nargs=4)
    parser.add_argument(
        "--index-qpos4", type=float, nargs=4, default=INDEX_CLEAR_QPOS4.tolist()
    )
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    config = load_config(CONFIG)
    readback = dict(
        isolated_wuji_operation("preview", config, np.zeros(8, dtype=np.float64))
    )
    actual_before = np.asarray(readback["actual"], dtype=np.float64)
    thumb_target = (
        actual_before[:4]
        if args.thumb_qpos4 is None
        else np.asarray(args.thumb_qpos4, dtype=np.float64)
    )
    index_target = np.asarray(args.index_qpos4, dtype=np.float64)
    requested = np.concatenate((thumb_target, index_target))
    preview = dict(isolated_wuji_operation("preview", config, requested))
    if not bool(preview.get("eligible", False)):
        raise RuntimeError(f"Index-clear target rejected: {preview.get('reason', '')}")
    effective = np.asarray(preview["effective_qpos8_rad"], dtype=np.float64)
    if not np.allclose(effective, requested, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("Safety clipping would change the thumb target")
    if args.preview_only:
        print(
            json.dumps(
                {
                    "eligible": True,
                    "actual_before_qpos8_rad": actual_before[:8].tolist(),
                    "requested_qpos8_rad": requested.tolist(),
                    "effective_qpos8_rad": effective.tolist(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    executed = dict(isolated_wuji_operation("execute", config, requested))
    actual_after = np.asarray(executed["actual"], dtype=np.float64)
    report = {
        "actual_before_qpos8_rad": actual_before[:8].tolist(),
        "requested_qpos8_rad": requested.tolist(),
        "effective_qpos8_rad": effective.tolist(),
        "actual_after_qpos8_rad": actual_after[:8].tolist(),
        "thumb_error_rad": (actual_after[:4] - thumb_target).tolist(),
        "index_error_rad": (actual_after[4:8] - index_target).tolist(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
