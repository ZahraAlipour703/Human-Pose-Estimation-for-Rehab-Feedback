"""
Synthetic visual reference motions.

IMPORTANT:
These trajectories are visualization guides only.
They are not clinical normative motion profiles.
"""

from __future__ import annotations

import math

import numpy as np


def _body():
    return {
        "NOSE": (0.50, 0.12, 0.0),

        "LEFT_SHOULDER": (0.43, 0.34, 0.0),
        "RIGHT_SHOULDER": (0.57, 0.34, 0.0),

        "LEFT_ELBOW": (0.43, 0.55, 0.0),
        "RIGHT_ELBOW": (0.57, 0.55, 0.0),

        "LEFT_WRIST": (0.43, 0.76, 0.0),
        "RIGHT_WRIST": (0.57, 0.76, 0.0),

        "LEFT_HIP": (0.46, 0.64, 0.0),
        "RIGHT_HIP": (0.54, 0.64, 0.0),

        "LEFT_KNEE": (0.46, 0.82, 0.0),
        "RIGHT_KNEE": (0.54, 0.82, 0.0),

        "LEFT_ANKLE": (0.46, 0.98, 0.0),
        "RIGHT_ANKLE": (0.54, 0.98, 0.0),

        "LEFT_HEEL": (0.46, 1.00, 0.0),
        "RIGHT_HEEL": (0.54, 1.00, 0.0),

        "LEFT_FOOT_INDEX": (0.48, 1.00, 0.0),
        "RIGHT_FOOT_INDEX": (0.56, 1.00, 0.0),
    }


def _phase(num_frames):
    return np.linspace(
        0.0,
        1.0,
        max(2, int(num_frames)),
    )


def _arm_motion(
    pose,
    side,
    amount,
    lateral=False,
):
    shoulder = pose[
        f"{side}_SHOULDER"
    ]

    sx, sy, sz = shoulder

    angle = math.radians(
        (20.0 if lateral else 35.0)
        + 125.0 * amount
    )

    elbow_x = (
        sx
        + 0.22 * math.sin(angle)
    )

    elbow_y = (
        sy
        - 0.22 * math.cos(angle)
    )

    wrist_x = (
        elbow_x
        + 0.20 * math.sin(angle)
    )

    wrist_y = (
        elbow_y
        - 0.20 * math.cos(angle)
    )

    pose[f"{side}_ELBOW"] = (
        elbow_x,
        elbow_y,
        sz,
    )

    pose[f"{side}_WRIST"] = (
        wrist_x,
        wrist_y,
        sz,
    )


def shoulder_flexion_reference(
    num_frames=140,
    side="LEFT",
):
    poses = []

    for t in _phase(num_frames):
        amount = (
            0.5
            - 0.5
            * math.cos(
                2.0
                * math.pi
                * t
            )
        )

        pose = _body()

        _arm_motion(
            pose,
            side,
            amount,
            lateral=False,
        )

        poses.append(pose)

    return poses


def shoulder_abduction_reference(
    num_frames=140,
    side="LEFT",
):
    poses = []

    for t in _phase(num_frames):
        amount = (
            0.5
            - 0.5
            * math.cos(
                2.0
                * math.pi
                * t
            )
        )

        pose = _body()

        _arm_motion(
            pose,
            side,
            amount,
            lateral=True,
        )

        poses.append(pose)

    return poses


def mini_squat_reference(
    num_frames=140,
):
    poses = []

    for t in _phase(num_frames):

        depth = (
            0.5
            - 0.5
            * math.cos(
                2.0
                * math.pi
                * t
            )
        )

        pose = _body()

        pose["LEFT_HIP"] = (
            0.46,
            0.64 + 0.04 * depth,
            0.0,
        )

        pose["RIGHT_HIP"] = (
            0.54,
            0.64 + 0.04 * depth,
            0.0,
        )

        pose["LEFT_KNEE"] = (
            0.435,
            0.82 - 0.01 * depth,
            0.0,
        )

        pose["RIGHT_KNEE"] = (
            0.565,
            0.82 - 0.01 * depth,
            0.0,
        )

        poses.append(pose)

    return poses


def biceps_curl_reference(
    num_frames=140,
    side="LEFT",
):
    poses = []

    for t in _phase(num_frames):

        flex = (
            0.5
            - 0.5
            * math.cos(
                2.0
                * math.pi
                * t
            )
        )

        pose = _body()

        shoulder = pose[
            f"{side}_SHOULDER"
        ]

        sx, sy, sz = shoulder

        pose[f"{side}_ELBOW"] = (
            sx,
            sy + 0.22,
            sz,
        )

        pose[f"{side}_WRIST"] = (
            sx
            + 0.10 * flex,
            sy
            + 0.10
            - 0.28 * flex,
            sz,
        )

        poses.append(pose)

    return poses


def calf_raise_reference(
    num_frames=140,
):
    poses = []

    for t in _phase(num_frames):

        lift = (
            0.025
            * (
                0.5
                - 0.5
                * math.cos(
                    2.0
                    * math.pi
                    * t
                )
            )
        )

        pose = _body()

        for side in (
            "LEFT",
            "RIGHT",
        ):
            pose[
                f"{side}_HEEL"
            ] = (
                pose[
                    f"{side}_HEEL"
                ][0],
                1.00 - lift,
                0.0,
            )

            pose[
                f"{side}_FOOT_INDEX"
            ] = (
                pose[
                    f"{side}_FOOT_INDEX"
                ][0],
                1.00 - lift,
                0.0,
            )

        poses.append(pose)

    return poses


def single_leg_stance_reference(
    num_frames=140,
):
    poses = []

    for t in _phase(num_frames):

        sway = (
            0.006
            * math.sin(
                2.0
                * math.pi
                * t
            )
        )

        pose = _body()

        pose["LEFT_SHOULDER"] = (
            0.43 + sway,
            0.34,
            0.0,
        )

        pose["RIGHT_SHOULDER"] = (
            0.57 + sway,
            0.34,
            0.0,
        )

        # Right-leg support reference.
        pose["RIGHT_KNEE"] = (
            0.54,
            0.82,
            0.0,
        )

        pose["RIGHT_ANKLE"] = (
            0.54,
            0.98,
            0.0,
        )

        # Lifted left leg.
        pose["LEFT_KNEE"] = (
            0.40,
            0.80,
            0.0,
        )

        pose["LEFT_ANKLE"] = (
            0.37,
            0.70,
            0.0,
        )

        pose["LEFT_FOOT_INDEX"] = (
            0.35,
            0.68,
            0.0,
        )

        poses.append(pose)

    return poses


REFERENCE_FUNCTIONS = {
    "mini_squat": mini_squat_reference,
    "shoulder_flexion": shoulder_flexion_reference,
    "shoulder_abduction": shoulder_abduction_reference,
    "biceps_curl": biceps_curl_reference,
    "calf_raise": calf_raise_reference,
    "single_leg_stance": single_leg_stance_reference,
}