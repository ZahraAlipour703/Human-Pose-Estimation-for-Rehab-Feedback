"""
YOLOv8 Pose adapter.

Uses Ultralytics YOLO Pose directly and converts
the detected pose into the same named-landmark
dictionary used by the rehabilitation exercise layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from ultralytics import YOLO


Point3D = Tuple[float, float, float]


# COCO-17 keypoint order used by YOLO Pose.
COCO_KEYPOINT_NAMES = [
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]


class YOLOv8PoseWrapper:
    """
    Small wrapper around Ultralytics YOLO Pose.

    Parameters
    ----------
    model_path:
        YOLO pose checkpoint. By default, Ultralytics
        downloads yolo11n-pose.pt / yolo8n-pose.pt
        depending on the supplied filename.

    confidence:
        Minimum person confidence.

    device:
        "cpu", "cuda", 0, etc.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence: float = 0.35,
        device=None,
    ) -> None:

        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.device = device

        self.model = YOLO(
            str(self.model_path)
        )

    def _predict(
        self,
        frame: np.ndarray,
    ):
        kwargs = {
            "source": frame,
            "conf": self.confidence,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        return self.model.predict(
            **kwargs
        )

    def predict_landmarks(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Point3D]:
        """
        Return the strongest detected person as
        normalized named landmarks.

        Coordinates are:
            x, y in [0, 1]
            z in YOLO pose coordinate units
        """

        results = self._predict(frame)

        if not results:
            return {}

        result = results[0]

        if result.keypoints is None:
            return {}

        if result.keypoints.xy is None:
            return {}

        xy = (
            result.keypoints.xy
            .detach()
            .cpu()
            .numpy()
        )

        if len(xy) == 0:
            return {}

        # Choose the person with the largest
        # visible keypoint confidence / box confidence.
        person_index = 0

        if result.boxes is not None:
            boxes = result.boxes

            if boxes.conf is not None:
                confidences = (
                    boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                if len(confidences):
                    person_index = int(
                        np.argmax(
                            confidences
                        )
                    )

        person_xy = xy[person_index]

        # Optional keypoint confidence.
        confidence_array = None

        if result.keypoints.conf is not None:

            confidence_array = (
                result.keypoints.conf[
                    person_index
                ]
                .detach()
                .cpu()
                .numpy()
            )

        frame_h, frame_w = frame.shape[:2]

        if frame_w <= 0 or frame_h <= 0:
            return {}

        landmarks: Dict[
            str, Point3D
        ] = {}

        for index, name in enumerate(
            COCO_KEYPOINT_NAMES
        ):

            if index >= len(person_xy):
                break

            x_px = float(
                person_xy[index][0]
            )

            y_px = float(
                person_xy[index][1]
            )

            if confidence_array is not None:

                if index >= len(
                    confidence_array
                ):
                    continue

                keypoint_conf = float(
                    confidence_array[index]
                )

                if keypoint_conf < 0.20:
                    continue

            x = x_px / frame_w
            y = y_px / frame_h

            landmarks[name] = (
                float(x),
                float(y),
                0.0,
            )

        return landmarks

    def findPose(
        self,
        frame: np.ndarray,
        draw: bool = False,
    ):
        """
        Backward-compatible interface for the project.

        Returns:
            landmark dictionary

        If draw=True, the YOLO result is rendered and
        returned as:

            (annotated_frame, landmarks)
        """

        results = self._predict(frame)

        landmarks = (
            {}
            if not results
            else self._result_to_landmarks(
                results[0],
                frame,
            )
        )

        if not draw:
            return landmarks

        annotated = frame.copy()

        if results:
            try:
                annotated = results[0].plot()
            except Exception:
                # Drawing failure should not destroy
                # the inference result.
                annotated = frame.copy()

        return (
            annotated,
            landmarks,
        )

    def _result_to_landmarks(
        self,
        result,
        frame: np.ndarray,
    ) -> Dict[str, Point3D]:

        if result.keypoints is None:
            return {}

        xy_tensor = result.keypoints.xy

        if xy_tensor is None:
            return {}

        xy = (
            xy_tensor
            .detach()
            .cpu()
            .numpy()
        )

        if len(xy) == 0:
            return {}

        person_index = 0

        if result.boxes is not None:
            if result.boxes.conf is not None:

                confidence = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                if len(confidence):
                    person_index = int(
                        np.argmax(
                            confidence
                        )
                    )

        points = xy[person_index]

        keypoint_conf = None

        if result.keypoints.conf is not None:

            keypoint_conf = (
                result.keypoints.conf[
                    person_index
                ]
                .detach()
                .cpu()
                .numpy()
            )

        h, w = frame.shape[:2]

        if h <= 0 or w <= 0:
            return {}

        output = {}

        for index, name in enumerate(
            COCO_KEYPOINT_NAMES
        ):

            if index >= len(points):
                break

            if keypoint_conf is not None:

                if (
                    index >= len(
                        keypoint_conf
                    )
                ):
                    continue

                if float(
                    keypoint_conf[index]
                ) < 0.20:
                    continue

            x = float(
                points[index][0]
            ) / w

            y = float(
                points[index][1]
            ) / h

            output[name] = (
                x,
                y,
                0.0,
            )

        return output