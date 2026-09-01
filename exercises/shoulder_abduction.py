"""
Shoulder abduction assessment.

Uses the same anatomical shoulder-elevation geometry as
the existing system, but is kept as a separate exercise
because the intended movement plane and reference trajectory
are different.

Use a frontal camera view for abduction.
"""

from __future__ import annotations

import time

from exercises.base import BaseExerciseChecker


class ShoulderAbductionChecker(BaseExerciseChecker):

    def __init__(
        self,
        config=None,
        logger=None,
    ):
        super().__init__(
            "shoulder_abduction",
            config,
            logger,
        )

        c = self.config

        self.target_up = float(
            c.get("target_angle_up", 160)
        )

        self.target_down = float(
            c.get("target_angle_down", 25)
        )

        self.tolerance = float(
            c.get("tolerance_deg", 15)
        )

        self.hold_time = float(
            c.get("hold_time_sec", 1.0)
        )

        self.max_elbow_flexion = float(
            c.get("max_elbow_flexion_deg", 20)
        )

        self.max_torso_tilt = float(
            c.get("max_torso_tilt_deg", 12)
        )

        self.smoothers = self.create_smoothers(
            c.get("smoothing_window", 5)
        )

        self.stage = {
            "LEFT": "down",
            "RIGHT": "down",
        }

        self.reps = {
            "LEFT": 0,
            "RIGHT": 0,
        }

        self.hold_start = {
            "LEFT": None,
            "RIGHT": None,
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

        feedback = []
        per_side = {}

        for side in self.selected_sides():

            names = [
                f"{side}_HIP",
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

            hip = lm[f"{side}_HIP"]
            shoulder = lm[f"{side}_SHOULDER"]
            elbow = lm[f"{side}_ELBOW"]
            wrist = lm[f"{side}_WRIST"]

            angle = self.smoothers[
                side
            ].update(
                self.angle(
                    hip,
                    shoulder,
                    elbow,
                )
            )

            elbow_angle = self.angle(
                shoulder,
                elbow,
                wrist,
            )

            elbow_flexion = max(
                0.0,
                180.0 - elbow_angle,
            )

            reasons = []

            if (
                torso_tilt is not None
                and torso_tilt > self.max_torso_tilt
            ):
                reasons.append(
                    "Keep torso upright"
                )

            if (
                elbow_flexion
                > self.max_elbow_flexion
            ):
                reasons.append(
                    "Keep elbow straighter"
                )

            previous = self.stage[side]

            at_down = (
                angle
                <= self.target_down
                + self.tolerance
            )

            at_up = (
                angle
                >= self.target_up
                - self.tolerance
            )

            if at_down:
                self.stage[side] = "down"
                self.hold_start[side] = None

            elif at_up:

                if self.hold_start[side] is None:
                    self.hold_start[side] = now

                held = (
                    now
                    - self.hold_start[side]
                )

                self.stage[side] = (
                    "up"
                    if held >= self.hold_time
                    else "holding"
                )

            else:
                self.stage[side] = "moving"

            if (
                previous == "up"
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
                "status": (
                    "good"
                    if not reasons
                    else "form_warning"
                ),
                "angle": float(angle),
                "elbow_flexion": float(
                    elbow_flexion
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
            metrics={
                "torso_tilt_deg": torso_tilt
            },
            per_side=per_side,
        )