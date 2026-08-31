"""
Mini squat assessment.

Primary signal:
    HIP -> KNEE -> ANKLE

Secondary signal:
    trunk inclination.

Camera recommendation:
    frontal or 45-degree view with the full body visible.
"""

from __future__ import annotations

import time

from exercises.base import BaseExerciseChecker


class MiniSquatChecker(BaseExerciseChecker):

    def __init__(
        self,
        config=None,
        logger=None,
    ):
        super().__init__(
            "mini_squat",
            config,
            logger,
        )

        c = self.config

        self.down_knee_angle = float(
            c.get("down_knee_angle", 85)
        )

        self.up_knee_angle = float(
            c.get("up_knee_angle", 155)
        )

        self.tolerance = float(
            c.get("tolerance_deg", 12)
        )

        self.max_torso_tilt = float(
            c.get("max_torso_tilt_deg", 30)
        )

        self.smoothers = self.create_smoothers(
            c.get("smoothing_window", 5)
        )

        self.stage = {
            "LEFT": "up",
            "RIGHT": "up",
        }

        self.reps = {
            "LEFT": 0,
            "RIGHT": 0,
        }

    def _torso_tilt(self, lm):
        names = [
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
        ]

        if not self.has_points(
            lm,
            names,
        ):
            return None

        shoulder_mid = self.midpoint(
            lm["LEFT_SHOULDER"],
            lm["RIGHT_SHOULDER"],
        )

        hip_mid = self.midpoint(
            lm["LEFT_HIP"],
            lm["RIGHT_HIP"],
        )

        return self.vertical_tilt_deg(
            shoulder_mid,
            hip_mid,
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

        torso_tilt = self._torso_tilt(lm)

        per_side = {}
        feedback = []

        for side in self.selected_sides():

            names = [
                f"{side}_HIP",
                f"{side}_KNEE",
                f"{side}_ANKLE",
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

            knee_angle = self.smoothers[
                side
            ].update(
                self.angle(
                    lm[f"{side}_HIP"],
                    lm[f"{side}_KNEE"],
                    lm[f"{side}_ANKLE"],
                )
            )

            reasons = []

            if (
                torso_tilt is not None
                and torso_tilt > self.max_torso_tilt
            ):
                reasons.append(
                    "Reduce forward trunk lean"
                )

            previous = self.stage[side]

            at_bottom = (
                knee_angle
                <= (
                    self.down_knee_angle
                    + self.tolerance
                )
            )

            at_top = (
                knee_angle
                >= (
                    self.up_knee_angle
                    - self.tolerance
                )
            )

            if at_bottom:
                self.stage[side] = "down"

            elif at_top:
                self.stage[side] = "up"

            else:
                self.stage[side] = "moving"

            if (
                previous == "down"
                and self.stage[side] == "up"
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
                "angle": float(knee_angle),
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
            metrics={
                "torso_tilt_deg": torso_tilt
            },
            per_side=per_side,
        )