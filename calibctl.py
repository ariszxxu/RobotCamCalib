#!/usr/bin/env python3
"""Inspect, validate, or run a named calibration task from ``configs/``."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

from robot_cam_calib.config import CalibrationCatalog, CatalogError


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


def print_yaml(value: Any) -> None:
    print(yaml.safe_dump(value, sort_keys=False, allow_unicode=True).rstrip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="list catalog entries")
    list_parser.add_argument(
        "kind",
        choices=("all", "robots", "cameras", "targets", "tasks"),
        default="all",
        nargs="?",
    )

    show_parser = subparsers.add_parser("show", help="show one catalog entry")
    show_parser.add_argument(
        "kind", choices=("robots", "cameras", "targets", "tasks")
    )
    show_parser.add_argument("name")

    validate_parser = subparsers.add_parser(
        "validate", help="validate references and optionally input files"
    )
    validate_parser.add_argument("--check-files", action="store_true")

    command_parser = subparsers.add_parser(
        "command", help="print the resolved legacy command for a task"
    )
    command_parser.add_argument("task")

    run_parser = subparsers.add_parser("run", help="run a configured task")
    run_parser.add_argument("task")
    run_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="arguments appended after '--'",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        catalog = CalibrationCatalog.load(args.config_dir)
        registries = {
            "robots": catalog.robots,
            "cameras": catalog.cameras,
            "targets": catalog.targets,
            "tasks": catalog.tasks,
        }
        if args.action == "list":
            selected = registries if args.kind == "all" else {args.kind: registries[args.kind]}
            for kind, registry in selected.items():
                print(f"{kind}:")
                for name in registry:
                    print(f"  - {name}")
            return
        if args.action == "show":
            registry = registries[args.kind]
            if args.name not in registry:
                raise CatalogError(f"Unknown {args.kind} entry: {args.name}")
            print_yaml(registry[args.name])
            return
        if args.action == "validate":
            warnings = catalog.validate(check_files=args.check_files)
            print("Configuration catalog is valid.")
            for warning in warnings:
                print(f"WARNING: {warning}")
            return

        command = catalog.build_command(args.task)
        if args.action == "command":
            print(shlex.join(command))
            return
        extra_args = list(args.extra_args)
        if extra_args[:1] == ["--"]:
            extra_args = extra_args[1:]
        completed = subprocess.run(
            [*command, *extra_args],
            cwd=catalog.repo_root,
            check=False,
        )
        raise SystemExit(completed.returncode)
    except CatalogError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc


if __name__ == "__main__":
    main()
