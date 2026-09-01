"""
Biceps curl exercise checker.

Measures elbow flexion/extension using:

    SHOULDER -> ELBOW -> WRIST

The checker uses hysteresis-style thresholds so small
frame-to-frame fluctuations do not create repeated counts.
"""

from __future__ import annotations

import time

from exercises.base import BaseExerciseChecker


class BicepsCurlChecker(BaseExerciseChecker):
    """Evaluate repeated elbow flexion and extension."""

    def __init__(
        self,
        config=None,
        logger=None,
    ):
        super().__init__(
            "biceps_curl",
            config,
            logger,
        )

        config = config or {}

        self.flexion_angle = float(
            config.get("flexion_angle", 55.0)
        )

        self.extension_angle = float(
            config.get("extension_angle", 155.0)
        )

        self.tolerance = float(
            config.get("tolerance_deg", 12.0)
        )

        self.max_shoulder_motion = float(
            config.get(
                "max_shoulder_motion",
                0.06,
            )
        )

        self.smoothers = self.create_smoothers(
            config.get(
                "smoothing_window",
                5,
            )
        )

        self.stage = {
            "LEFT": "extended",
            "RIGHT": "extended",
        }

        self.reps = {
            "LEFT": 0,
            "RIGHT": 0,
        }

        self.baseline_shoulder = {
            "LEFT": None,
            "RIGHT": None,
        }

        self.baseline_upper_arm_length = {
            "LEFT": None,
            "RIGHT": None,
        }

    def _upper_arm_length(
        self,
        landmarks,
        side,
    ):
        return max(
            self.distance(
                landmarks[
                    f"{side}_SHOULDER"
                ],
                landmarks[
                    f"{side}_ELBOW"
                ],
            ),
            1e-6,
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
                f"{side}_SHOULDER",
                f"{side}_ELBOW",
                f"{side}_WRIST",
            ]

            if not self.has_points(
                lm,
                required,
            ):
                per_side[side] = {
                    "status": "no_pose",
                    "angle": None,
                    "reps": self.reps[side],
                    "stage": self.stage[side],
                    "reasons": [],
                }
                continue

            shoulder = lm[
                f"{side}_SHOULDER"
            ]

            elbow = lm[
                f"{side}_ELBOW"
            ]

            wrist = lm[
                f"{side}_WRIST"
            ]

            elbow_angle = self.smoothers[
                side
            ].update(
                self.angle(
                    shoulder,
                    elbow,
                    wrist,
                )
            )

            if (
                self.baseline_shoulder[
                    side
                ]
                is None
            ):
                self.baseline_shoulder[
                    side
                ] = shoulder

            if (
                self.baseline_upper_arm_length[
                    side
                ]
                is None
            ):
                self.baseline_upper_arm_length[
                    side
                ] = self._upper_arm_length(
                    lm,
                    side,
                )

            shoulder_motion = (
                self.distance(
                    shoulder,
                    self.baseline_shoulder[
                        side
                    ],
                )
                / self.baseline_upper_arm_length[
                    side
                ]
            )

            reasons = []

            if (
                shoulder_motion
                > self.max_shoulder_motion
            ):
                reasons.append(
                    "Keep upper arm stable"
                )

            previous_stage = (
                self.stage[side]
            )

            flexed_limit = (
                self.flexion_angle
                + self.tolerance
            )

            extended_limit = (
                self.extension_angle
                - self.tolerance
            )

            if elbow_angle <= flexed_limit:
                self.stage[side] = "flexed"

            elif (
                elbow_angle
                >= extended_limit
            ):
                self.stage[side] = "extended"

            else:
                self.stage[side] = "moving"

            # One repetition = flexed -> extended.
            if (
                previous_stage == "flexed"
                and self.stage[side] == "extended"
            ):
                self.reps[side] += 1

                self.log(
                    "rep",
                    self.reps[side],
                    side,
                    now,
                )

            per_side[side] = {
                "status": (
                    "good"
                    if not reasons
                    else "form_warning"
                ),
                "angle": float(
                    elbow_angle
                ),
                "shoulder_motion": float(
                    shoulder_motion
                ),
                "reps": self.reps[side],
                "stage": self.stage[side],
                "reasons": reasons,
            }

            feedback.extend(reasons)

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