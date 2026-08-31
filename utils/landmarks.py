"""
Landmark conversion and fusion utilities.
"""

from __future__ import annotations

from collections.abc import Mapping


def landmarks_to_dict(landmarks):
    """
    Convert:

        MediaPipe landmarks
        OR
        existing mapping

    into:

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

        for name, value in landmarks.items():

            if value is None:
                continue

            if isinstance(
                value,
                (tuple, list),
            ):

                if len(value) < 2:
                    continue

                xyz = list(value[:3])

                if len(xyz) == 2:
                    xyz.append(0.0)

                result[str(name)] = tuple(
                    float(v)
                    for v in xyz
                )

            elif hasattr(
                value,
                "x",
            ) and hasattr(
                value,
                "y",
            ):

                result[str(name)] = (
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

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required to convert raw pose landmarks."
        ) from exc

    result = {}

    for index, lm in enumerate(
        landmarks
    ):

        try:
            name = (
                mp.solutions.pose.PoseLandmark(
                    index
                ).name
            )
        except ValueError:
            continue

        result[name] = (
            float(lm.x),
            float(lm.y),
            float(
                getattr(
                    lm,
                    "z",
                    0.0,
                )
            ),
        )

    return result


def fuse_landmarks(
    yolo_landmarks,
    mp_landmarks,
):
    """
    Merge two named landmark dictionaries.

    MediaPipe acts as the fallback source.
    YOLO values take priority when present.
    """

    yolo = landmarks_to_dict(
        yolo_landmarks
    )

    mp_data = landmarks_to_dict(
        mp_landmarks
    )

    fused = dict(mp_data)
    fused.update(yolo)

    return fused