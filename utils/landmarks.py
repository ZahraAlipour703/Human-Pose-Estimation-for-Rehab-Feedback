"""
Landmark normalization and fusion.
"""

from __future__ import annotations

from collections.abc import Mapping


def landmarks_to_dict(
    landmarks,
):
    if landmarks is None:
        return {}

    if isinstance(
        landmarks,
        Mapping,
    ):

        output = {}

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
                    output[
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

                continue

            if (
                hasattr(value, "x")
                and hasattr(value, "y")
            ):

                output[
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

        return output

    try:

        import mediapipe as mp

        enum = (
            mp.solutions.pose.PoseLandmark
        )

    except Exception as exc:

        raise RuntimeError(
            "MediaPipe 0.10.x legacy Solutions API "
            "is required for raw MediaPipe landmarks."
        ) from exc

    output = {}

    for index, landmark in enumerate(
        landmarks
    ):

        try:

            name = enum(
                index
            ).name

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

        output[name] = (
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

    return output


def fuse_landmarks(
    yolo_landmarks,
    mp_landmarks,
):
    """
    Merge YOLO and MediaPipe landmarks.

    MediaPipe is the fallback/base source.
    YOLO overwrites only the landmarks it provides.
    """

    mp_data = landmarks_to_dict(
        mp_landmarks
    )

    yolo_data = landmarks_to_dict(
        yolo_landmarks
    )

    fused = dict(mp_data)

    for name, point in yolo_data.items():
        fused[name] = point

    return fused