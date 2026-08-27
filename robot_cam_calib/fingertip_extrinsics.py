"""Joint camera/moving-target extrinsics for a kinematic fingertip chain.

Every observation follows the explicit ``T_A_B`` convention and satisfies

``T_palm_tip(q_i) @ T_tip_cube = T_palm_camera @ T_camera_cube_i``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import (
    inv_T,
    make_T,
    params_to_transform,
    residual_stats,
    so3_log,
    transform_delta,
    transform_to_params,
)


HAND_JOINT_NAMES = tuple(
    f"left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)


@dataclass(frozen=True)
class FingertipObservation:
    index: int
    T_palm_tip: np.ndarray
    T_camera_cube: np.ndarray


@dataclass(frozen=True)
class MultiChainObservation:
    """One moving-target observation, labelled by its kinematic chain."""

    chain: str
    index: int
    T_palm_tip: np.ndarray
    T_camera_target: np.ndarray


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = np.fromstring(
        "0 0 0" if origin is None else origin.get("xyz", "0 0 0"),
        sep=" ",
        dtype=np.float64,
    )
    rpy = np.fromstring(
        "0 0 0" if origin is None else origin.get("rpy", "0 0 0"),
        sep=" ",
        dtype=np.float64,
    )
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.all(np.isfinite((xyz, rpy))):
        raise ValueError(f"Invalid origin in URDF joint {joint.get('name')}")
    return make_T(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)


class PalmTipKinematics:
    """Minimal, mesh-independent URDF FK from ``left_palm_link`` to one tip."""

    def __init__(
        self,
        urdf_path: Path,
        tip_link: str,
        *,
        palm_link: str = "left_palm_link",
    ) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.tip_link = str(tip_link)
        self.palm_link = str(palm_link)
        root = ET.parse(self.urdf_path).getroot()
        by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            child_name = None if child is None else child.get("link")
            if child_name:
                if child_name in by_child:
                    raise ValueError(f"Multiple parent joints for URDF link {child_name}")
                by_child[child_name] = joint

        reversed_chain: list[ET.Element] = []
        current = self.tip_link
        visited: set[str] = set()
        while current != self.palm_link:
            if current in visited:
                raise ValueError("URDF palm-to-tip chain contains a cycle")
            visited.add(current)
            joint = by_child.get(current)
            parent = None if joint is None else joint.find("parent")
            parent_name = None if parent is None else parent.get("link")
            if joint is None or not parent_name:
                raise ValueError(
                    f"URDF has no chain from {self.palm_link} to {self.tip_link}"
                )
            reversed_chain.append(joint)
            current = parent_name
        self.chain = tuple(reversed(reversed_chain))
        self.chain_joint_names = tuple(
            str(joint.get("name"))
            for joint in self.chain
            if joint.get("type") != "fixed"
        )
        unknown = set(self.chain_joint_names) - set(HAND_JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unsupported non-hand joints in palm-to-tip chain: {unknown}")
        self.chain_joint_indices = tuple(
            HAND_JOINT_NAMES.index(name) for name in self.chain_joint_names
        )

    def forward(self, qpos20_rad: object) -> np.ndarray:
        qpos = np.asarray(qpos20_rad, dtype=np.float64)
        if qpos.shape != (20,) or not np.all(np.isfinite(qpos)):
            raise ValueError("Wuji qpos must contain 20 finite radians")
        values = dict(zip(HAND_JOINT_NAMES, qpos, strict=True))
        result = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            result = result @ _origin_transform(joint)
            joint_type = str(joint.get("type", "fixed"))
            if joint_type == "fixed":
                continue
            name = str(joint.get("name"))
            axis_element = joint.find("axis")
            axis = np.fromstring(
                "1 0 0" if axis_element is None else axis_element.get("xyz", "1 0 0"),
                sep=" ",
                dtype=np.float64,
            )
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise ValueError(f"Invalid axis in URDF joint {name}")
            norm = float(np.linalg.norm(axis))
            if norm <= 0.0:
                raise ValueError(f"Zero axis in URDF joint {name}")
            axis /= norm
            value = values[name]
            if joint_type in {"revolute", "continuous"}:
                result = result @ make_T(
                    Rotation.from_rotvec(axis * value).as_matrix(), np.zeros(3)
                )
            elif joint_type == "prismatic":
                result = result @ make_T(np.eye(3), axis * value)
            else:
                raise ValueError(f"Unsupported URDF joint type {joint_type!r} for {name}")
        return result


def _average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    items = [np.asarray(item, dtype=np.float64).reshape(4, 4) for item in transforms]
    if not items:
        raise ValueError("Cannot average no transforms")
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in items])).mean()
    translation = np.mean([item[:3, 3] for item in items], axis=0)
    return make_T(rotations.as_matrix(), translation)


def _closure_residuals(
    parameters: np.ndarray,
    observations: list[FingertipObservation],
    *,
    rotation_scale_rad: float,
    translation_scale_m: float,
) -> np.ndarray:
    T_palm_camera = params_to_transform(parameters[:6])
    T_tip_cube = params_to_transform(parameters[6:])
    rows: list[np.ndarray] = []
    for observation in observations:
        left = observation.T_palm_tip @ T_tip_cube
        right = T_palm_camera @ observation.T_camera_cube
        delta = inv_T(right) @ left
        rows.append(
            np.hstack(
                (
                    so3_log(delta[:3, :3]) / rotation_scale_rad,
                    delta[:3, 3] / translation_scale_m,
                )
            )
        )
    return np.hstack(rows)


def _sample_residuals(
    observations: list[FingertipObservation],
    T_palm_camera: np.ndarray,
    T_tip_cube: np.ndarray,
) -> tuple[list[float], list[float]]:
    rotations: list[float] = []
    translations: list[float] = []
    for observation in observations:
        left = observation.T_palm_tip @ T_tip_cube
        right = T_palm_camera @ observation.T_camera_cube
        rotation, translation = transform_delta(right, left)
        rotations.append(rotation)
        translations.append(translation)
    return rotations, translations


def _initial_parameters(
    observations: list[FingertipObservation],
    rng: np.random.Generator,
    start_index: int,
) -> np.ndarray:
    if start_index == 0:
        T_tip_cube = np.eye(4, dtype=np.float64)
    else:
        T_tip_cube = make_T(
            Rotation.random(random_state=rng).as_matrix(),
            rng.uniform(-0.05, 0.05, size=3),
        )
    camera_candidates = [
        item.T_palm_tip @ T_tip_cube @ inv_T(item.T_camera_cube)
        for item in observations
    ]
    T_palm_camera = _average_transforms(camera_candidates)
    return np.hstack(
        (transform_to_params(T_palm_camera), transform_to_params(T_tip_cube))
    )


def _fit(
    observations: list[FingertipObservation],
    *,
    starts: int,
    seed: int,
) -> tuple[np.ndarray, object]:
    rotation_scale_rad = np.deg2rad(0.5)
    translation_scale_m = 0.002
    rng = np.random.default_rng(seed)
    best = None
    for start_index in range(max(1, int(starts))):
        initial = _initial_parameters(observations, rng, start_index)
        fitted = least_squares(
            _closure_residuals,
            initial,
            args=(observations,),
            kwargs={
                "rotation_scale_rad": rotation_scale_rad,
                "translation_scale_m": translation_scale_m,
            },
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=3000,
            x_scale="jac",
        )
        if best is None or fitted.cost < best.cost:
            best = fitted
    assert best is not None
    return np.asarray(best.x, dtype=np.float64), best


def solve_fingertip_extrinsics(
    observations: list[FingertipObservation],
    *,
    min_samples: int = 12,
    starts: int = 24,
    seed: int = 20260824,
) -> dict[str, object]:
    """Jointly solve palm-camera and tip-cube constants with robust refinement."""

    if len(observations) < min_samples:
        raise ValueError(f"Need at least {min_samples} observations")
    normalized = [
        FingertipObservation(
            int(item.index),
            np.asarray(item.T_palm_tip, dtype=np.float64).reshape(4, 4),
            np.asarray(item.T_camera_cube, dtype=np.float64).reshape(4, 4),
        )
        for item in observations
    ]
    parameters, fitted = _fit(normalized, starts=starts, seed=seed)
    T_palm_camera = params_to_transform(parameters[:6])
    T_tip_cube = params_to_transform(parameters[6:])
    rotation_residuals, translation_residuals = _sample_residuals(
        normalized, T_palm_camera, T_tip_cube
    )
    rotation_array = np.asarray(rotation_residuals, dtype=np.float64)
    translation_array = np.asarray(translation_residuals, dtype=np.float64)
    rotation_limit = float(
        np.clip(
            np.median(rotation_array)
            + 3.0 * 1.4826 * np.median(
                np.abs(rotation_array - np.median(rotation_array))
            ),
            0.35,
            3.0,
        )
    )
    translation_limit = float(
        np.clip(
            np.median(translation_array)
            + 3.0 * 1.4826 * np.median(
                np.abs(translation_array - np.median(translation_array))
            ),
            0.0015,
            0.010,
        )
    )
    inliers = [
        index
        for index, (rotation, translation) in enumerate(
            zip(rotation_residuals, translation_residuals, strict=True)
        )
        if rotation <= rotation_limit and translation <= translation_limit
    ]
    if len(inliers) >= min_samples and len(inliers) < len(normalized):
        inlier_observations = [normalized[index] for index in inliers]
        parameters, fitted = _fit(
            inlier_observations,
            starts=max(8, starts // 2),
            seed=seed + 1,
        )
        T_palm_camera = params_to_transform(parameters[:6])
        T_tip_cube = params_to_transform(parameters[6:])
        rotation_residuals, translation_residuals = _sample_residuals(
            normalized, T_palm_camera, T_tip_cube
        )
    else:
        inlier_observations = normalized
        inliers = list(range(len(normalized)))

    singular_values = np.linalg.svd(np.asarray(fitted.jac), compute_uv=False)
    tolerance = max(fitted.jac.shape) * np.finfo(float).eps * singular_values[0]
    jacobian_rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > tolerance
        else float("inf")
    )
    return {
        "T_palm_camera": T_palm_camera,
        "T_tip_cube": T_tip_cube,
        "inlier_indices": [normalized[index].index for index in inliers],
        "outlier_indices": [
            item.index for index, item in enumerate(normalized) if index not in inliers
        ],
        "rotation_residual_deg": rotation_residuals,
        "translation_residual_m": translation_residuals,
        "rotation_stats_deg": residual_stats(rotation_residuals),
        "translation_stats_m": residual_stats(translation_residuals),
        "rotation_limit_deg": rotation_limit,
        "translation_limit_m": translation_limit,
        "jacobian_rank": jacobian_rank,
        "jacobian_condition": condition,
        "optimizer_cost": float(fitted.cost),
        "optimizer_optimality": float(fitted.optimality),
        "optimizer_success": bool(fitted.success),
        "num_samples": len(normalized),
        "num_inliers": len(inliers),
    }


def _multichain_residuals(
    parameters: np.ndarray,
    observations: list[MultiChainObservation],
    chain_offsets: dict[str, int],
    *,
    rotation_scale_rad: float,
    translation_scale_m: float,
) -> np.ndarray:
    T_palm_camera = params_to_transform(parameters[:6])
    targets = {
        chain: params_to_transform(parameters[offset : offset + 6])
        for chain, offset in chain_offsets.items()
    }
    rows: list[np.ndarray] = []
    for observation in observations:
        left = observation.T_palm_tip @ targets[observation.chain]
        right = T_palm_camera @ observation.T_camera_target
        delta = inv_T(right) @ left
        rows.append(
            np.hstack(
                (
                    so3_log(delta[:3, :3]) / rotation_scale_rad,
                    delta[:3, 3] / translation_scale_m,
                )
            )
        )
    return np.hstack(rows)


def _fit_multichain(
    observations: list[MultiChainObservation],
    chains: tuple[str, ...],
    *,
    starts: int,
    seed: int,
) -> tuple[np.ndarray, object, dict[str, int]]:
    chain_offsets = {chain: 6 + 6 * index for index, chain in enumerate(chains)}
    grouped = {
        chain: [item for item in observations if item.chain == chain]
        for chain in chains
    }
    rng = np.random.default_rng(seed)

    # Each chain gives a valid independent hand-eye initialization. Averaging
    # their camera estimates keeps the shared-camera solve in the right basin;
    # none of those estimates is held fixed in the optimization below.
    independent: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chain_index, chain in enumerate(chains):
        single = [
            FingertipObservation(item.index, item.T_palm_tip, item.T_camera_target)
            for item in grouped[chain]
        ]
        fitted_parameters, _ = _fit(
            single,
            starts=max(1, min(2, starts)),
            seed=seed + 101 * chain_index,
        )
        independent[chain] = (
            params_to_transform(fitted_parameters[:6]),
            params_to_transform(fitted_parameters[6:]),
        )

    base_camera = _average_transforms(value[0] for value in independent.values())
    base = np.zeros(6 + 6 * len(chains), dtype=np.float64)
    base[:6] = transform_to_params(base_camera)
    for chain in chains:
        offset = chain_offsets[chain]
        base[offset : offset + 6] = transform_to_params(independent[chain][1])

    best = None
    for start_index in range(max(1, int(starts))):
        initial = base.copy()
        if start_index:
            initial[:3] += rng.normal(0.0, np.deg2rad(5.0), 3)
            initial[3:6] += rng.normal(0.0, 0.01, 3)
            for offset in chain_offsets.values():
                initial[offset : offset + 3] += rng.normal(
                    0.0, np.deg2rad(8.0), 3
                )
                initial[offset + 3 : offset + 6] += rng.normal(0.0, 0.01, 3)
        fitted = least_squares(
            _multichain_residuals,
            initial,
            args=(observations, chain_offsets),
            kwargs={
                "rotation_scale_rad": np.deg2rad(0.5),
                "translation_scale_m": 0.002,
            },
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=1500,
            x_scale="jac",
        )
        if best is None or fitted.cost < best.cost:
            best = fitted
    assert best is not None
    return np.asarray(best.x, dtype=np.float64), best, chain_offsets


def solve_multichain_fingertip_extrinsics(
    observations: list[MultiChainObservation],
    *,
    min_samples_per_chain: int = 12,
    starts: int = 24,
    seed: int = 20260824,
) -> dict[str, object]:
    """Jointly solve one palm-camera and one target transform per chain.

    All transforms are free variables. In particular, this does not fix or
    derive a fingertip transform from a previously calibrated camera pose.
    """

    normalized = [
        MultiChainObservation(
            str(item.chain),
            int(item.index),
            np.asarray(item.T_palm_tip, dtype=np.float64).reshape(4, 4),
            np.asarray(item.T_camera_target, dtype=np.float64).reshape(4, 4),
        )
        for item in observations
    ]
    chains = tuple(sorted({item.chain for item in normalized}))
    if len(chains) < 2:
        raise ValueError("Multi-chain solve requires at least two chains")
    counts = {chain: sum(item.chain == chain for item in normalized) for chain in chains}
    deficient = {
        chain: count for chain, count in counts.items() if count < min_samples_per_chain
    }
    if deficient:
        raise ValueError(f"Insufficient observations per chain: {deficient}")

    parameters, fitted, offsets = _fit_multichain(
        normalized, chains, starts=starts, seed=seed
    )
    T_palm_camera = params_to_transform(parameters[:6])
    targets = {
        chain: params_to_transform(parameters[offset : offset + 6])
        for chain, offset in offsets.items()
    }

    per_observation: list[tuple[float, float]] = []
    for item in normalized:
        left = item.T_palm_tip @ targets[item.chain]
        right = T_palm_camera @ item.T_camera_target
        per_observation.append(transform_delta(right, left))

    inlier_mask = np.ones(len(normalized), dtype=bool)
    limits: dict[str, dict[str, float]] = {}
    for chain in chains:
        indices = [index for index, item in enumerate(normalized) if item.chain == chain]
        rotations = np.asarray([per_observation[index][0] for index in indices])
        translations = np.asarray([per_observation[index][1] for index in indices])
        rotation_limit = float(
            np.clip(
                np.median(rotations)
                + 3.0 * 1.4826 * np.median(np.abs(rotations - np.median(rotations))),
                0.35,
                3.0,
            )
        )
        translation_limit = float(
            np.clip(
                np.median(translations)
                + 3.0
                * 1.4826
                * np.median(np.abs(translations - np.median(translations))),
                0.0015,
                0.010,
            )
        )
        limits[chain] = {
            "rotation_deg": rotation_limit,
            "translation_m": translation_limit,
        }
        for index in indices:
            rotation, translation = per_observation[index]
            inlier_mask[index] = (
                rotation <= rotation_limit and translation <= translation_limit
            )

    if all(
        sum(inlier_mask[index] for index, item in enumerate(normalized) if item.chain == chain)
        >= min_samples_per_chain
        for chain in chains
    ) and not bool(np.all(inlier_mask)):
        inlier_observations = [
            item for index, item in enumerate(normalized) if inlier_mask[index]
        ]
        parameters, fitted, offsets = _fit_multichain(
            inlier_observations,
            chains,
            starts=max(8, starts // 2),
            seed=seed + 1,
        )
        T_palm_camera = params_to_transform(parameters[:6])
        targets = {
            chain: params_to_transform(parameters[offset : offset + 6])
            for chain, offset in offsets.items()
        }
        per_observation = []
        for item in normalized:
            left = item.T_palm_tip @ targets[item.chain]
            right = T_palm_camera @ item.T_camera_target
            per_observation.append(transform_delta(right, left))

    singular_values = np.linalg.svd(np.asarray(fitted.jac), compute_uv=False)
    tolerance = max(fitted.jac.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > tolerance
        else float("inf")
    )
    per_chain: dict[str, dict[str, object]] = {}
    for chain in chains:
        indices = [index for index, item in enumerate(normalized) if item.chain == chain]
        rotations = [per_observation[index][0] for index in indices]
        translations = [per_observation[index][1] for index in indices]
        per_chain[chain] = {
            "T_tip_target": targets[chain],
            "sample_count": len(indices),
            "inlier_count": int(sum(inlier_mask[indices])),
            "rotation_residual_deg": rotations,
            "translation_residual_m": translations,
            "rotation_stats_deg": residual_stats(rotations),
            "translation_stats_m": residual_stats(translations),
            "limits": limits[chain],
        }
    return {
        "T_palm_camera": T_palm_camera,
        "chains": per_chain,
        "jacobian_rank": rank,
        "jacobian_condition": condition,
        "optimizer_cost": float(fitted.cost),
        "optimizer_optimality": float(fitted.optimality),
        "optimizer_success": bool(fitted.success),
        "parameter_count": 6 + 6 * len(chains),
        "sample_count": len(normalized),
    }
