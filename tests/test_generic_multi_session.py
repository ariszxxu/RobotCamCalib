from __future__ import annotations

from pathlib import Path

import numpy as np

from solve_multi_session_hand_eye import build_groups, dotted_get, validate_documents


def _document(offset: float = 0.0) -> dict:
    transform = np.eye(4)
    transform[0, 3] = offset
    return {
        "schema": "capture.v1",
        "meta": {"serial": "camera-a", "K": np.eye(3).tolist()},
        "records": [
            {
                "id": 4,
                "image": "/tmp/example.png",
                "A": np.eye(4).tolist(),
                "C": transform.tolist(),
                "error": 0.25,
            }
        ],
    }


def _config() -> dict:
    return {
        "input": {
            "allowed_schemas": ["capture.v1"],
            "samples_path": "records",
            "sample_fields": {
                "index": "id",
                "image_path": "image",
                "base_gripper_transform": "A",
                "camera_target_transform": "C",
                "reprojection_error_px": "error",
            },
            "consistency_paths": ["meta.serial", "meta.K"],
        }
    }


def test_dotted_get_supports_nested_mappings_and_lists() -> None:
    assert dotted_get({"a": [{"b": 3}]}, "a.0.b") == 3


def test_generic_adapter_builds_solver_groups() -> None:
    documents = [_document(), _document(0.01)]
    consistency = validate_documents(
        [Path("session-a.yaml"), Path("session-b.yaml")], documents, _config()
    )
    groups = build_groups(documents, _config())
    assert consistency["meta.serial"] == "camera-a"
    assert len(groups) == 2
    assert groups[1][0]["index"] == 4
    assert np.isclose(groups[1][0]["T_camera_charuco"][0, 3], 0.01)
