"""
Landmark conversion and fusion utilities.
"""

from __future__ import annotations

from collections.abc import Mapping


def landmarks_to_dict(
    landmarks,
):
    """
    Convert landmark representations into:

        {
            "LEFT_SHOULDER": (x, y, z),
            ...
        }
    """

    if landmarks is None:
        return {}

    if isinstance(
        landmarks,
        Mapping,
    ):

        result = {}

        for name, value in (
            landmarks.items()
        ):

            if value is None:
                continue

            if isinstance(
                value,
                (tuple, list),
            ):

                if len(value) < 2:
                    continue

                xyz = list(
                    value[:3]
                )

                while len(xyz) < 3:
                    xyz.append(0.0)

                try:
                    result[
                        str(name)
                    ] = tuple(
                        float(v)
                        for v in xyz
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            elif (
                hasattr(value, "x")
                and hasattr(value, "y")
            ):

                result[
                    str(name)
                ] = (
                    float(value.x),
                    float(value.y),
                    float(
                        getattr(
                            value,
                            "z",
                            0.0,
                        )
                    ),
                )

        return result

    # MediaPipe Pose landmarks.
    try:
        import mediapipe as mp

        pose_landmark_enum = (
            mp.solutions.pose.PoseLandmark
        )

    except Exception as exc:
        raise RuntimeError(
            "Raw MediaPipe landmarks require "
            "MediaPipe 0.10.x legacy Solutions API."
        ) from exc

    result = {}

    try:
        iterator = enumerate(
            landmarks
        )

        for index, landmark in iterator:

            try:
                name = (
                    pose_landmark_enum(
                        index
                    ).name
                )
            except ValueError:
                continue

            if not (
                hasattr(
                    landmark,
                    "x",
                )
                and hasattr(
                    landmark,
                    "y",
                )
            ):
                continue

            result[name] = (
                float(landmark.x),
                float(landmark.y),
                float(
                    getattr(
                        landmark,
                        "z",
                        0.0,
                    )
                ),
            )

    except TypeError:
        return {}

    return result


def fuse_landmarks(
    yolo_landmarks,
    mp_landmarks,
):
    """
    MediaPipe provides the fallback.
    YOLO/Visolus values override matching keys.
    """

    fused = {}

    fused.update(
        landmarks_to_dict(
            mp_landmarks
        )
    )

    fused.update(
        landmarks_to_dict(
            yolo_landmarks
        )
    )

    return fused