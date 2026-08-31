"""
Biceps curl assessment.

Primary signal:
    SHOULDER -> ELBOW -> WRIST

Secondary signal:
    normalized shoulder displacement.
"""

from __future__ import annotations

import time

from exercises.base import BaseExerciseChecker


class BicepsCurlChecker(BaseExerciseChecker):

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

        c = self.config

        self.flexion_angle = float(
            c.get("flexion_angle", 55)
        )

        self.extension_angle = float(
            c.get("extension_angle", 155)
        )

        self.tolerance = float(
            c.get("tolerance_deg", 12)
        )

        self.max_shoulder_motion = float(
            c.get("max_shoulder_motion", 0.06)
        )

        self.smoothers = self.create_smoothers(
            c.get("smoothing_window", 5)
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

        self.baseline_scale = {
            "LEFT": None,
            "RIGHT": None,
        }

    def _scale(self, lm, side):
        elbow = lm[f"{side}_ELBOW"]
        shoulder = lm[f"{side}_SHOULDER"]

        return max(
            self.distance(
                shoulder,
                elbow,
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

        per_side = {}
        feedback = []

        for side in self.selected_sides():

            names = [
                f"{side}_SHOULDER",
                f"{side}_ELBOW",
                f"{side}_WRIST",
            ]

            if not self.has_points(
                lm,
                names,
            ):
                per_side[side] = {
                    "status": "no_pose",
                    "angle": None,
                    "reps": self.reps[side],
                    "stage": self.stage[side],
                    "reasons": [],
                }
                continue

            shoulder = lm[f"{side}_SHOULDER"]
            elbow = lm[f"{side}_ELBOW"]
            wrist = lm[f"{side}_WRIST"]

            angle = self.smoothers[
                side
            ].update(
                self.angle(
                    shoulder,
                    elbow,
                    wrist,
                )
            )

            if (
                self.baseline_shoulder[side]
                is None
            ):
                self.baseline_shoulder[
                    side
                ] = shoulder

            if (
                self.baseline_scale[side]
                is None
            ):
                self.baseline_scale[
                    side
                ] = self._scale(lm, side)

            shoulder_displacement = (
                self.distance(
                    shoulder,
                    self.baseline_shoulder[side],
                )
                / self.baseline_scale[side]
            )

            reasons = []

            if (
                shoulder_displacement
                > self.max_shoulder_motion
            ):
                reasons.append(
                    "Keep upper arm stable"
                )

            previous = self.stage[side]

            at_flexion = (
                angle
                <= (
                    self.flexion_angle
                    + self.tolerance
                )
            )

            at_extension = (
                angle
                >= (
                    self.extension_angle
                    - self.tolerance
                )
            )

            if at_flexion:
                self.stage[side] = "flexed"

            elif at_extension:
                self.stage[side] = "extended"

            else:
                self.stage[side] = "moving"

            if (
                previous == "flexed"
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
                "angle": float(angle),
                "shoulder_displacement": float(
                    shoulder_displacement
                ),
                "reps": self.reps[side],
                "stage": self.stage[side],
                "reasons": reasons,
            }

            feedback.extend(reasons)

        return self.standard_result(
            status="active",
            reps=max(self.reps.values()),
            stage="active",
            feedback=feedback,
            per_side=per_side,
        )