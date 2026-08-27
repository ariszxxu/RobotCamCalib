"""Load and validate the repository's robot, camera, target, and task catalogs."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


CATALOG_FILES = {
    "robots": "robots.yaml",
    "cameras": "cameras.yaml",
    "targets": "targets.yaml",
    "tasks": "tasks.yaml",
}
VALID_CAMERA_BACKENDS = {"opencv", "realsense", "orbbec", "pyav"}
VALID_ROBOT_BACKENDS = {"xarm-python-sdk"}
VALID_TARGET_KINDS = {
    "checkerboard",
    "charuco",
    "apriltag_grid",
    "aprilcube",
}
VALID_TASK_KINDS = {
    "intrinsics",
    "paired_target_extrinsics",
    "mutual_grid_extrinsics",
    "robot_hand_eye",
    "robot_kinematic_extrinsics",
}
_PLACEHOLDER = re.compile(r"\$\{([^{}]+)\}")


class CatalogError(ValueError):
    """Raised when a catalog or task reference is invalid."""


def _load_mapping(path: Path, root_key: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise CatalogError(f"Expected a YAML mapping: {path}")
    if int(payload.get("version", 0)) != 1:
        raise CatalogError(f"Unsupported or missing version in {path}")
    values = payload.get(root_key)
    if not isinstance(values, dict):
        raise CatalogError(f"Missing '{root_key}' mapping in {path}")
    return values


def _lookup(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise CatalogError(f"Unknown configuration reference: {dotted_key}")
            current = current[index]
            continue
        if not isinstance(current, dict) or part not in current:
            raise CatalogError(f"Unknown configuration reference: {dotted_key}")
        current = current[part]
    if isinstance(current, (dict, list)):
        raise CatalogError(
            f"Configuration reference must resolve to a scalar: {dotted_key}"
        )
    return current


@dataclass(frozen=True)
class CalibrationCatalog:
    repo_root: Path
    config_dir: Path
    robots: dict[str, Any]
    cameras: dict[str, Any]
    targets: dict[str, Any]
    tasks: dict[str, Any]

    @classmethod
    def load(cls, config_dir: Path) -> "CalibrationCatalog":
        resolved_dir = config_dir.expanduser().resolve()
        repo_root = resolved_dir.parent
        loaded = {
            key: _load_mapping(resolved_dir / filename, key)
            for key, filename in CATALOG_FILES.items()
        }
        catalog = cls(repo_root=repo_root, config_dir=resolved_dir, **loaded)
        catalog.validate(check_files=False)
        return catalog

    def context(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "robots": self.robots,
            "cameras": self.cameras,
            "targets": self.targets,
            "tasks": self.tasks,
        }

    def render(self, value: str) -> str:
        context = self.context()

        def replace(match: re.Match[str]) -> str:
            return str(_lookup(context, match.group(1)))

        rendered = _PLACEHOLDER.sub(replace, str(value))
        if "${" in rendered:
            raise CatalogError(f"Unresolved configuration value: {value}")
        return rendered

    def build_command(self, task_name: str) -> list[str]:
        task = self.tasks.get(task_name)
        if not isinstance(task, dict):
            raise CatalogError(f"Unknown calibration task: {task_name}")
        runner = task.get("runner")
        arguments = task.get("arguments", [])
        if not isinstance(runner, str) or not isinstance(arguments, list):
            raise CatalogError(f"Invalid runner/arguments for task {task_name}")
        runner_path = Path(self.render(runner))
        if not runner_path.is_absolute():
            runner_path = self.repo_root / runner_path
        return [
            sys.executable,
            str(runner_path),
            *[self.render(v) for v in arguments],
        ]

    def validate(self, *, check_files: bool) -> list[str]:
        warnings: list[str] = []
        for name, robot in self.robots.items():
            if not isinstance(robot, dict):
                raise CatalogError(f"Robot '{name}' must be a mapping")
            if robot.get("backend") not in VALID_ROBOT_BACKENDS:
                raise CatalogError(
                    f"Robot '{name}' has unsupported backend "
                    f"{robot.get('backend')!r}"
                )
            if int(robot.get("axis_count", 0)) <= 0:
                raise CatalogError(f"Robot '{name}' has invalid axis_count")
            connection = robot.get("connection")
            if not isinstance(connection, dict) or not connection.get("ip"):
                raise CatalogError(f"Robot '{name}' must define connection.ip")
            profiles = robot.get("profiles")
            if not isinstance(profiles, dict) or not profiles:
                raise CatalogError(f"Robot '{name}' must define profiles")
        for name, camera in self.cameras.items():
            if not isinstance(camera, dict):
                raise CatalogError(f"Camera '{name}' must be a mapping")
            if camera.get("backend") not in VALID_CAMERA_BACKENDS:
                raise CatalogError(
                    f"Camera '{name}' has unsupported backend {camera.get('backend')!r}"
                )
            connections = camera.get("connections")
            if not isinstance(connections, dict) or not connections:
                raise CatalogError(f"Camera '{name}' must define connections")
            if any(isinstance(value, (dict, list)) for value in connections.values()):
                raise CatalogError(
                    f"Camera '{name}' connection values must be scalar"
                )
            profiles = camera.get("profiles")
            if not isinstance(profiles, dict) or not profiles:
                raise CatalogError(f"Camera '{name}' must define profiles")
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    raise CatalogError(
                        f"Camera profile '{name}.{profile_name}' must be a mapping"
                    )
                resolution = profile.get("resolution")
                if (
                    not isinstance(resolution, list)
                    or len(resolution) != 2
                    or any(int(value) <= 0 for value in resolution)
                ):
                    raise CatalogError(
                        f"Camera profile '{name}.{profile_name}' has invalid resolution"
                    )
                if float(profile.get("fps", 0)) <= 0:
                    raise CatalogError(
                        f"Camera profile '{name}.{profile_name}' has invalid fps"
                    )
                intrinsics = profile.get("intrinsics")
                if check_files and intrinsics not in (None, "runtime"):
                    path = self.resolve_path(str(intrinsics))
                    if not path.is_file():
                        warnings.append(
                            f"missing intrinsics: {name}.{profile_name}: {path}"
                        )

        physical_target_ids: set[str] = set()
        for name, target in self.targets.items():
            if not isinstance(target, dict):
                raise CatalogError(f"Target '{name}' must be a mapping")
            if target.get("kind") not in VALID_TARGET_KINDS:
                raise CatalogError(
                    f"Target '{name}' has unsupported kind {target.get('kind')!r}"
                )
            physical_id = target.get("physical_id")
            if not isinstance(physical_id, str) or not physical_id.strip():
                raise CatalogError(
                    f"Target '{name}' must define a non-empty physical_id"
                )
            if physical_id in physical_target_ids:
                raise CatalogError(
                    f"Duplicate target physical_id: {physical_id}"
                )
            physical_target_ids.add(physical_id)
            config_path = target.get("config")
            if check_files and config_path:
                path = self.resolve_path(str(config_path))
                if not path.is_file():
                    warnings.append(f"missing target config: {name}: {path}")

        for name, task in self.tasks.items():
            if not isinstance(task, dict):
                raise CatalogError(f"Task '{name}' must be a mapping")
            if task.get("kind") not in VALID_TASK_KINDS:
                raise CatalogError(
                    f"Task '{name}' has unsupported kind {task.get('kind')!r}"
                )
            self._validate_references(name, task.get("robots", []), self.robots)
            self._validate_references(name, task.get("cameras", []), self.cameras)
            self._validate_references(name, task.get("targets", []), self.targets)
            command = self.build_command(name)
            if check_files and not Path(command[1]).is_file():
                warnings.append(f"missing task runner: {name}: {command[1]}")
        return warnings

    def resolve_path(self, value: str) -> Path:
        rendered = Path(self.render(value)).expanduser()
        return rendered if rendered.is_absolute() else self.repo_root / rendered

    @staticmethod
    def _validate_references(
        task_name: str,
        references: Iterable[str],
        registry: dict[str, Any],
    ) -> None:
        if not isinstance(references, list):
            raise CatalogError(f"Task '{task_name}' references must be lists")
        missing = [name for name in references if name not in registry]
        if missing:
            raise CatalogError(
                f"Task '{task_name}' references unknown entries: {missing}"
            )
