"""
Single-leg stance assessment.

Measures:
    - hold duration
    - trunk sway
    - hip sway

For reliable use, set:
    side = "left"
or:
    side = "right"

Automatic bilateral selection is supported but is less reliable.
"""

from __future__ import annotations

import time

from exercises.base import BaseExerciseChecker


class SingleLegStanceChecker(BaseExerciseChecker):

    def __init__(
        self,
        config=None,
        logger=None,
    ):
        super().__init__(
            "single_leg_stance",
            config,
            logger,
        )

        c = self.config

        self.hold_time = float(
            c.get("hold_time_sec", 10.0)
        )

        self.max_trunk_sway = float(
            c.get("max_trunk_sway_deg", 12.0)
        )

        self.max_hip_sway = float(
            c.get("max_hip_sway", 0.05)
        )

        self.active_side = None
        self.start_time = None
        self.best_hold = 0.0

        self.baseline = {
            "LEFT": None,
            "RIGHT": None,
        }

        self.completed = {
            "LEFT": False,
            "RIGHT": False,
        }

        self._last_completed_side = None

    def _automatic_support_side(self, lm):
        if not self.has_points(
            lm,
            [
                "LEFT_ANKLE",
                "RIGHT_ANKLE",
            ],
        ):
            return None

        left_y = lm["LEFT_ANKLE"][1]
        right_y = lm["RIGHT_ANKLE"][1]

        # Smaller image y = higher/lifted foot.
        if abs(left_y - right_y) < 0.05:
            return self.active_side

        # If left foot is higher, right leg supports.
        if left_y < right_y:
            return "RIGHT"

        return "LEFT"

    def _support_side(self, lm):
        sides = self.selected_sides()

        if sides == ["LEFT"]:
            return "LEFT"

        if sides == ["RIGHT"]:
            return "RIGHT"

        return self._automatic_support_side(lm)

    def update(
        self,
        landmarks,
        t=None,
    ):

        now = (
            float(t)
            if t is not None
            else time.time()
        )

        lm = self.landmark_dict(
            landmarks
        )

        if not lm:
            return self.no_pose_result()

        support = self._support_side(lm)

        if support is None:
            return self.no_pose_result()

        required = [
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
            f"{support}_HIP",
            f"{support}_KNEE",
            f"{support}_ANKLE",
        ]

        if not self.has_points(
            lm,
            required,
        ):
            return self.no_pose_result()

        shoulder_mid = self.midpoint(
            lm["LEFT_SHOULDER"],
            lm["RIGHT_SHOULDER"],
        )

        hip_mid = self.midpoint(
            lm["LEFT_HIP"],
            lm["RIGHT_HIP"],
        )

        trunk_tilt = self.vertical_tilt_deg(
            shoulder_mid,
            hip_mid,
        )

        hip = lm[f"{support}_HIP"]

        # Start/reset timer when selected support leg changes.
        if self.active_side != support:
            self.active_side = support
            self.start_time = now
            self.baseline[support] = (
                trunk_tilt,
                hip,
            )

        if self.baseline[support] is None:
            self.baseline[support] = (
                trunk_tilt,
                hip,
            )

        baseline_trunk, baseline_hip = (
            self.baseline[support]
        )

        trunk_sway = abs(
            trunk_tilt - baseline_trunk
        )

        hip_sway = self.distance(
            hip,
            baseline_hip,
        )

        elapsed = max(
            0.0,
            now
            - (
                self.start_time
                if self.start_time is not None
                else now
            ),
        )

        stable = (
            trunk_sway
            <= self.max_trunk_sway
            and
            hip_sway
            <= self.max_hip_sway
        )

        feedback = []

        if trunk_sway > self.max_trunk_sway:
            feedback.append(
                "Reduce trunk sway"
            )

        if hip_sway > self.max_hip_sway:
            feedback.append(
                "Reduce hip movement"
            )

        if stable:
            self.best_hold = max(
                self.best_hold,
                elapsed,
            )

        completed = (
            elapsed >= self.hold_time
            and stable
        )

        if (
            completed
            and not self.completed[support]
        ):
            self.completed[support] = True
            self._last_completed_side = support

            self.log(
                "completed_hold_sec",
                elapsed,
                support,
                now,
            )

        status = (
            "completed"
            if completed
            else (
                "stable"
                if stable
                else "balance_warning"
            )
        )

        return self.standard_result(
            status=status,
            reps=0,
            stage=(
                "holding"
                if stable
                else "unstable"
            ),
            feedback=feedback,
            metrics={
                "active_side": support,
                "hold_time_sec": float(elapsed),
                "best_hold_sec": float(
                    self.best_hold
                ),
                "trunk_sway_deg": float(
                    trunk_sway
                ),
                "hip_sway": float(hip_sway),
            },
            per_side={
                support: {
                    "status": status,
                    "hold_time_sec": float(elapsed),
                    "trunk_sway_deg": float(
                        trunk_sway
                    ),
                    "hip_sway": float(
                        hip_sway
                    ),
                    "reps": 0,
                    "stage": (
                        "holding"
                        if stable
                        else "unstable"
                    ),
                    "reasons": feedback,
                }
            },
        )