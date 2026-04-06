"""Shared G1 policy↔robot action/state helpers.

Used by g1_policy_viz_client (continuous rollout, HDF5 playback) and g1_execution_client
so 16 / 28 / 29-DOF formatting and 16→29 expansion match exactly.
"""

from __future__ import annotations

import numpy as np


def binary_gripper_to_hand_joints(binary_value: float, is_left_hand: bool = False) -> np.ndarray:
    """Binary gripper (0=open, 1=closed) -> 7-DOF Dex3 joint targets; threshold <0.1 => open."""
    if is_left_hand:
        hand_closed = np.array(
            [0.0, 0.7, 0.5, -1.3, -1.1, -1.3, -1.1],
            dtype=np.float32,
        )
    else:
        hand_closed = np.array(
            [0.0, -0.7, -0.5, 1.3, 1.1, 1.3, 1.1],
            dtype=np.float32,
        )
    binary_int = 0 if binary_value < 0.1 else 1
    if binary_int == 0:
        return np.zeros(7, dtype=np.float32)
    return hand_closed.copy()


def convert_16dim_binary_gripper_to_29dim(
    arms: np.ndarray, left_gripper: float, right_gripper: float
) -> np.ndarray:
    """16-D components [arm(14), raw L, raw R] -> 29-D [arm, hand×2, waist=0]."""
    left_gripper = 0.0 if left_gripper < 0.1 else 1.0
    right_gripper = 0.0 if right_gripper < 0.1 else 1.0
    left_hand = binary_gripper_to_hand_joints(left_gripper, is_left_hand=True)
    right_hand = binary_gripper_to_hand_joints(right_gripper, is_left_hand=False)
    return np.concatenate([arms, left_hand, right_hand, [0.0]]).astype(np.float32)


def convert_actions_to_29dim(actions: np.ndarray, action_dim: int) -> np.ndarray:
    """Policy chunk (N, D) with D in {16,28,29} -> (N, 29) robot commands."""
    if actions.shape[1] == 29:
        return actions.copy()
    if actions.shape[1] == 28:
        waist_yaw = np.zeros((actions.shape[0], 1), dtype=np.float32)
        return np.concatenate([actions, waist_yaw], axis=1)
    if actions.shape[1] == 16:
        converted = []
        for action in actions:
            arms = action[0:14]
            action_29 = convert_16dim_binary_gripper_to_29dim(
                arms, float(action[14]), float(action[15])
            )
            converted.append(action_29)
        return np.array(converted, dtype=np.float32)
    raise ValueError(
        f"Unsupported action dimension: {actions.shape[1]}. Expected 16, 28, or 29."
    )


def convert_qpos_to_16dim_state(qpos: np.ndarray) -> np.ndarray:
    """Full qpos (28 or 29) -> [arm(14), mean(left hand), mean(right hand)] for policy state."""
    arms = qpos[:14].astype(np.float32)
    if len(qpos) >= 28:
        left_hand = qpos[14:21]
        right_hand = qpos[21:28]
        left_gripper = np.mean(left_hand).astype(np.float32)
        right_gripper = np.mean(right_hand).astype(np.float32)
    else:
        left_gripper = np.float32(0.0)
        right_gripper = np.float32(0.0)
    binary_gripper = np.array([left_gripper, right_gripper], dtype=np.float32)
    return np.concatenate([arms, binary_gripper]).astype(np.float32)


def round_binary_gripper_raw(left_raw: float, right_raw: float) -> np.ndarray:
    """Same 0.1 threshold as convert_16dim_binary_gripper_to_29dim (for state tracking)."""
    l = 0.0 if left_raw < 0.1 else 1.0
    r = 0.0 if right_raw < 0.1 else 1.0
    return np.array([l, r], dtype=np.float32)
