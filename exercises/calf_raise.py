"""
Calf raise exercise checker.

Primary signal:
    normalized heel elevation relative to the ankle.

The implementation is intended for computer-vision
exercise analysis. Thresholds are engineering parameters,
not clinically validated reference values.
"""

from __future__ import annotations

import time

from exercises.base import BaseExerciseChecker


class CalfRaiseChecker(BaseExerciseChecker):
    """Detect repeated calf raises."""

    def __init__(
        self,
        config=None,
        logger=None,
    ):
        super().__init__(
            "calf_raise",
            config or {},
            logger,
        )

        config = config or {}

        self.rise_threshold = float(
            config.get(
                "rise_threshold",
                0.025,
            )
        )

        self.return_threshold = float(
            config.get(
                "return_threshold",
                0.010,
            )
        )

        self.smoothing_window = int(
            config.get(
                "smoothing_window",
                5,
            )
        )

        self.smoothers = self.create_smoothers(
            self.smoothing_window
        )

        self.baseline_ratio = {
            "LEFT": None,
            "RIGHT": None,
        }

        self.stage = {
            "LEFT": "down",
            "RIGHT": "down",
        }

        self.reps = {
            "LEFT": 0,
            "RIGHT": 0,
        }

    def _heel_to_ankle_ratio(
        self,
        landmarks,
        side,
    ):
        """
        Estimate heel-to-ankle vertical separation
        normalized by lower-leg length.
        """

        knee = landmarks[
            f"{side}_KNEE"
        ]

        ankle = landmarks[
            f"{side}_ANKLE"
        ]

        heel = landmarks[
            f"{side}_HEEL"
        ]

        lower_leg_length = max(
            self.distance(
                knee,
                ankle,
            ),
            1e-6,
        )

        vertical_separation = (
            heel[1] - ankle[1]
        )

        return (
            vertical_separation
            / lower_leg_length
        )

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

        feedback = []
        per_side = {}

        for side in self.selected_sides():

            required = [
                f"{side}_KNEE",
                f"{side}_ANKLE",
                f"{side}_HEEL",
            ]

            if not self.has_points(
                lm,
                required,
            ):

                per_side[side] = {
                    "status": "no_pose",
                    "rise": None,
                    "reps": self.reps[side],
                    "stage": self.stage[side],
                    "reasons": [],
                }

                continue

            ratio = (
                self._heel_to_ankle_ratio(
                    lm,
                    side,
                )
            )

            baseline = (
                self.baseline_ratio[side]
            )

            if baseline is None:

                self.baseline_ratio[
                    side
                ] = ratio

                baseline = ratio

            # Heel moves upward => y becomes smaller.
            rise = max(
                0.0,
                baseline - ratio,
            )

            rise = self.smoothers[
                side
            ].update(rise)

            previous_stage = (
                self.stage[side]
            )

            if (
                rise
                >= self.rise_threshold
            ):

                self.stage[side] = "up"

            elif (
                rise
                <= self.return_threshold
            ):

                self.stage[side] = "down"

            else:

                self.stage[side] = "moving"

            # One repetition:
            # UP -> DOWN.
            if (
                previous_stage == "up"
                and self.stage[side] == "down"
            ):

                self.reps[side] += 1

                self.log(
                    "rep",
                    self.reps[side],
                    side,
                    now,
                )

            per_side[side] = {
                "status": "good",
                "rise": float(rise),
                "reps": self.reps[side],
                "stage": self.stage[side],
                "reasons": [],
            }

        return self.standard_result(
            status="active",
            reps=max(
                self.reps.values()
            ),
            stage="active",
            feedback=feedback,
            metrics={},
            per_side=per_side,
        )