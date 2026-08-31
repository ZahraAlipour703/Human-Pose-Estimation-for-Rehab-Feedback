"""
Shared infrastructure for rehabilitation exercise checkers.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from utils.angles import angle_between_3d
from utils.smoothing import SimpleSmoother


Point3D = Tuple[float, float, float]


class BaseExerciseChecker:
    """
    Base class for all exercise checkers.

    Input:
        Either:
            - dict[str, (x, y, z)]
            - MediaPipe PoseLandmark list

    Output:
        Standard dictionary containing:
            exercise
            status
            stage
            reps
            feedback
            metrics
            per_side
    """

    SIDES = ("LEFT", "RIGHT")

    def __init__(
        self,
        name: str,
        config: Optional[dict] = None,
        logger=None,
    ) -> None:
        self.name = name
        self.config = config or {}
        self.logger = logger

        self.state = "idle"
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    @staticmethod
    def landmark_dict(landmarks) -> Dict[str, Point3D]:
        """
        Normalize landmarks into a named dictionary.

        Supported:
            1. Existing mapping:
               {"LEFT_SHOULDER": (x, y, z), ...}

            2. MediaPipe landmark iterable.
        """

        if landmarks is None:
            return {}

        # Already normalized.
        if isinstance(landmarks, dict):
            output = {}

            for name, value in landmarks.items():
                if value is None:
                    continue

                # Tuple/list/array.
                if isinstance(value, (tuple, list, np.ndarray)):
                    if len(value) < 2:
                        continue

                    xyz = list(value[:3])

                    if len(xyz) == 2:
                        xyz.append(0.0)

                    output[str(name)] = tuple(
                        float(v) for v in xyz
                    )

                    continue

                # MediaPipe-like object.
                if hasattr(value, "x") and hasattr(value, "y"):
                    output[str(name)] = (
                        float(value.x),
                        float(value.y),
                        float(getattr(value, "z", 0.0)),
                    )

            return output

        # MediaPipe landmark list.
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required when passing raw landmark lists. "
                "Install the project requirements first."
            ) from exc

        output = {}

        try:
            for index, lm in enumerate(landmarks):
                name = mp.solutions.pose.PoseLandmark(index).name

                output[name] = (
                    float(lm.x),
                    float(lm.y),
                    float(getattr(lm, "z", 0.0)),
                )
        except (TypeError, AttributeError, ValueError, IndexError):
            return {}

        return output

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def has_points(
        landmarks: Dict[str, Point3D],
        names: Iterable[str],
    ) -> bool:
        return all(name in landmarks for name in names)

    @staticmethod
    def angle(
        a: Point3D,
        b: Point3D,
        c: Point3D,
    ) -> float:
        return float(angle_between_3d(a, b, c))

    @staticmethod
    def midpoint(
        a: Point3D,
        b: Point3D,
    ) -> Point3D:
        return (
            (a[0] + b[0]) / 2.0,
            (a[1] + b[1]) / 2.0,
            (a[2] + b[2]) / 2.0,
        )

    @staticmethod
    def distance(
        a: Point3D,
        b: Point3D,
    ) -> float:
        return float(
            np.linalg.norm(
                np.asarray(a, dtype=float)
                - np.asarray(b, dtype=float)
            )
        )

    @staticmethod
    def vertical_tilt_deg(
        top: Point3D,
        bottom: Point3D,
    ) -> float:
        """
        2-D deviation from vertical.

        0° = vertical.
        """
        dx = top[0] - bottom[0]
        dy = top[1] - bottom[1]

        denominator = abs(dy)

        if abs(dx) < 1e-9 and denominator < 1e-9:
            return 0.0

        return float(
            math.degrees(
                math.atan2(abs(dx), denominator + 1e-9)
            )
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def selected_sides(self) -> list[str]:
        side = str(
            self.config.get("side", "both")
        ).strip().lower()

        if side == "left":
            return ["LEFT"]

        if side == "right":
            return ["RIGHT"]

        return ["LEFT", "RIGHT"]

    def create_smoothers(
        self,
        window: int = 5,
    ) -> Dict[str, SimpleSmoother]:
        window = max(1, int(window))

        return {
            "LEFT": SimpleSmoother(window),
            "RIGHT": SimpleSmoother(window),
        }

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    @staticmethod
    def at_or_above(
        value: float,
        threshold: float,
    ) -> bool:
        return value >= threshold

    @staticmethod
    def at_or_below(
        value: float,
        threshold: float,
    ) -> bool:
        return value <= threshold

    # ------------------------------------------------------------------
    # Standard outputs
    # ------------------------------------------------------------------

    def no_pose_result(self) -> dict[str, Any]:
        self.state = "no_pose"

        return {
            "exercise": self.name,
            "status": "no_pose",
            "stage": "no_pose",
            "reps": 0,
            "feedback": ["Pose not detected"],
            "metrics": {},
            "per_side": {},
        }

    def standard_result(
        self,
        *,
        status: str,
        reps: int = 0,
        stage: str = "idle",
        feedback: Optional[list[str]] = None,
        metrics: Optional[dict] = None,
        per_side: Optional[dict] = None,
    ) -> dict[str, Any]:

        self.state = status

        return {
            "exercise": self.name,
            "status": status,
            "stage": stage,
            "reps": int(reps),
            "feedback": list(
                dict.fromkeys(feedback or [])
            ),
            "metrics": metrics or {},
            "per_side": per_side or {},
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        metric: str,
        value: Any,
        note: str = "",
        timestamp: Optional[float] = None,
    ) -> None:

        row = {
            "timestamp": (
                timestamp
                if timestamp is not None
                else time.time()
            ),
            "exercise": self.name,
            "metric": metric,
            "value": value,
            "note": note,
        }

        self.events.append(row)

        if self.logger is not None:
            try:
                self.logger.writerow(
                    [
                        row["timestamp"],
                        row["exercise"],
                        row["metric"],
                        row["value"],
                        row["note"],
                    ]
                )
            except Exception:
                # Logging must never crash inference.
                pass

    def save_events(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                self.events,
                indent=2,
            ),
            encoding="utf-8",
        )