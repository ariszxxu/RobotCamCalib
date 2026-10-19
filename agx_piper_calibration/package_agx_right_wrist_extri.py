"""Create an auditable user-accepted right-wrist extrinsic archive."""
import hashlib
import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


H = Path(__file__).resolve().parent
SOURCE = H / (
    "sessions/manual-20261019-right-v1/"
    "candidate-right-combined-filtered-1792398605507682129.json"
)
EXPORT_ROOT = H / "exports/agx_right_wrist_extri"
ARCHIVE = H / "exports/agx_right_wrist_extri.tar.gz"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def add_bytes(archive, name, data):
    info = tarfile.TarInfo("agx_right_wrist_extri/" + name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def main():
    if EXPORT_ROOT.exists() or ARCHIVE.exists():
        raise FileExistsError("Preserve the existing named export")
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    accepted_at = datetime.now(timezone(timedelta(hours=8), "Asia/Shanghai")).isoformat()
    deployment = {
        "schema": "agx.robot_camera_extrinsic.accepted.v1",
        "name": "agx_right_wrist_extri",
        "status": "accepted_by_user_with_recorded_validation_limitations",
        "accepted_for_control": True,
        "accepted_by": "user_explicit_instruction_in_current_session",
        "accepted_at": accepted_at,
        "qualified_by_original_checks": bool(source["qualified"]),
        "control_config_changed": False,
        "arm": source["arm"],
        "camera_serial": source["camera_serial"],
        "frame": source["frame"],
        "meaning": source["meaning"],
        "units": source["units"],
        "T_link6_camera_m": source["T_link6_camera_m"],
        "translation_mm": source["translation_mm"],
        "scipy_as_euler_xyz_deg": source["scipy_as_euler_xyz_deg"],
        "validation": {
            "used_training_count": len(source["used_training_indices"]),
            "holdout_count": len(source["holdout_indices"]),
            "holdout_rotation_stats_deg": source["holdout_residuals"][
                "rotation_stats_deg"
            ],
            "holdout_translation_stats_mm": source["holdout_residuals"][
                "translation_stats_mm"
            ],
            "batch_repeatability": source["batch_repeatability"],
            "image_rejected_indices": source["image_rejected_indices"],
            "geometric_training_rejected_indices": source[
                "geometric_training_rejected_indices"
            ],
            "checks": source["checks"],
        },
        "limitations": [
            "The original qualification checks did not all pass.",
            "Held-out translation residual median is about 6.93 mm and maximum is about 13.95 mm.",
            "The two capture batches differ by about 24.19 mm and 1.04 degrees when solved separately.",
            "Acceptance records the user's deployment decision; it does not change measured validation results.",
        ],
        "source": {
            "path": str(SOURCE),
            "sha256": digest(source_bytes),
            "session": "manual-20261019-right-v1",
        },
    }
    deployment_bytes = json_bytes(deployment)
    readme_bytes = (
        "# agx_right_wrist_extri\n\n"
        "User-accepted right-wrist D435 color optical to right link6 extrinsic.\n\n"
        "- `agx_right_wrist_extri.json`: compact accepted deployment record.\n"
        "- `source_candidate.json`: complete solver output and validation evidence.\n"
        "- `SHA256SUMS`: content hashes for both JSON files.\n\n"
        "The source candidate did not pass every original qualification threshold. "
        "The acceptance decision and measured limitations are both retained. "
        "This package does not modify a robot control configuration.\n"
    ).encode()
    sums_bytes = (
        f"{digest(deployment_bytes)}  agx_right_wrist_extri.json\n"
        f"{digest(source_bytes)}  source_candidate.json\n"
        f"{digest(readme_bytes)}  README.md\n"
    ).encode()
    EXPORT_ROOT.mkdir(parents=True)
    (EXPORT_ROOT / "agx_right_wrist_extri.json").write_bytes(deployment_bytes)
    (EXPORT_ROOT / "source_candidate.json").write_bytes(source_bytes)
    (EXPORT_ROOT / "README.md").write_bytes(readme_bytes)
    (EXPORT_ROOT / "SHA256SUMS").write_bytes(sums_bytes)
    with tarfile.open(ARCHIVE, "x:gz", compresslevel=9) as archive:
        add_bytes(archive, "README.md", readme_bytes)
        add_bytes(archive, "SHA256SUMS", sums_bytes)
        add_bytes(archive, "agx_right_wrist_extri.json", deployment_bytes)
        add_bytes(archive, "source_candidate.json", source_bytes)
    print(json.dumps({
        "name": deployment["name"],
        "archive": str(ARCHIVE),
        "archive_sha256": digest(ARCHIVE.read_bytes()),
        "accepted_for_control": deployment["accepted_for_control"],
        "qualified_by_original_checks": deployment["qualified_by_original_checks"],
        "control_config_changed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
