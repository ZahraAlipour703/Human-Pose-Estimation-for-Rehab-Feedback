"""
Local rehabilitation runner.

Pipeline:

    Camera / Video
          |
          +------> YOLOv8 Pose
          |
          +------> MediaPipe Pose
                     |
                     v
               Landmark Fusion
                     |
                     v
                Exercise Checker
                     |
                     v
             Feedback + Logging
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Generator

import cv2
import mediapipe as mp

from pose.yolov8_adapter import (
    YOLOv8PoseWrapper,
)

from exercises import (
    BicepsCurlChecker,
    CalfRaiseChecker,
    MiniSquatChecker,
    ShoulderAbductionChecker,
    ShoulderFlexionChecker,
    SingleLegStanceChecker,
)

from utils.landmarks import (
    fuse_landmarks,
)

from utils.reference_motion import (
    REFERENCE_FUNCTIONS,
)


ROOT = Path(
    __file__
).resolve().parent

CONFIG_PATH = (
    ROOT / "config.json"
)

LOG_DIR = (
    ROOT / "logs"
)


EXERCISE_CHECKERS = {
    "mini_squat":
        MiniSquatChecker,

    "shoulder_flexion":
        ShoulderFlexionChecker,

    "shoulder_abduction":
        ShoulderAbductionChecker,

    "biceps_curl":
        BicepsCurlChecker,

    "calf_raise":
        CalfRaiseChecker,

    "single_leg_stance":
        SingleLegStanceChecker,
}


EXERCISE_TITLES = {
    "mini_squat":
        "Mini Squat",

    "shoulder_flexion":
        "Shoulder Flexion",

    "shoulder_abduction":
        "Shoulder Abduction",

    "biceps_curl":
        "Biceps Curl",

    "calf_raise":
        "Calf Raise",

    "single_leg_stance":
        "Single-Leg Stance",
}


def load_config():

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Missing config file: "
            f"{CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def open_source(source):

    cap = cv2.VideoCapture(
        source
        if isinstance(source, int)
        else str(source)
    )

    if not cap.isOpened():

        cap.release()

        raise RuntimeError(
            f"Cannot open video source: "
            f"{source}"
        )

    return cap


def media_pipe_to_dict(results):

    if not results.pose_landmarks:
        return {}

    return {
        mp.solutions.pose.PoseLandmark(
            i
        ).name: (
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
        for i, lm
        in enumerate(
            results.pose_landmarks.landmark
        )
    }


def draw_reference(
    frame,
    pose_dict,
):
    """
    Draw a small visual reference pose.
    """

    if not pose_dict:
        return frame

    out = frame.copy()

    height, width = (
        out.shape[:2]
    )

    size = min(
        180,
        max(
            120,
            min(
                width,
                height,
            )
            // 4,
        ),
    )

    x0 = (
        width
        - size
        - 15
    )

    y0 = 15

    connections = [
        (
            "LEFT_SHOULDER",
            "LEFT_ELBOW",
        ),
        (
            "LEFT_ELBOW",
            "LEFT_WRIST",
        ),
        (
            "RIGHT_SHOULDER",
            "RIGHT_ELBOW",
        ),
        (
            "RIGHT_ELBOW",
            "RIGHT_WRIST",
        ),
        (
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
        ),
        (
            "LEFT_SHOULDER",
            "LEFT_HIP",
        ),
        (
            "RIGHT_SHOULDER",
            "RIGHT_HIP",
        ),
        (
            "LEFT_HIP",
            "RIGHT_HIP",
        ),
        (
            "LEFT_HIP",
            "LEFT_KNEE",
        ),
        (
            "LEFT_KNEE",
            "LEFT_ANKLE",
        ),
        (
            "RIGHT_HIP",
            "RIGHT_KNEE",
        ),
        (
            "RIGHT_KNEE",
            "RIGHT_ANKLE",
        ),
    ]

    def px(point):

        return (
            x0
            + int(point[0] * size),

            y0
            + int(point[1] * size),
        )

    cv2.rectangle(
        out,
        (
            x0 - 5,
            y0 - 5,
        ),
        (
            x0 + size + 5,
            y0 + size + 28,
        ),
        (20, 20, 20),
        -1,
    )

    for a, b in connections:

        if (
            a in pose_dict
            and b in pose_dict
        ):

            cv2.line(
                out,
                px(
                    pose_dict[a]
                ),
                px(
                    pose_dict[b]
                ),
                (190, 190, 190),
                2,
                cv2.LINE_AA,
            )

    for point in pose_dict.values():

        cv2.circle(
            out,
            px(point),
            3,
            (80, 220, 255),
            -1,
            cv2.LINE_AA,
        )

    return out


def annotate(
    frame,
    exercise_name,
    result,
    pose_results,
    reference,
    fps,
):

    out = frame.copy()

    if pose_results.pose_landmarks:

        mp.solutions.drawing_utils.draw_landmarks(
            out,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
        )

    title = EXERCISE_TITLES[
        exercise_name
    ]

    cv2.rectangle(
        out,
        (0, 0),
        (520, 190),
        (15, 15, 15),
        -1,
    )

    cv2.putText(
        out,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"Reps: {result.get('reps', 0)}",
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (100, 230, 130),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"Stage: {result.get('stage', '-')}",
        (15, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 220, 100),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"Status: {result.get('status', '-')}",
        (15, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"FPS: {fps:.1f}",
        (15, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    y = 177

    for message in (
        result.get("feedback", [])
        [:2]
    ):

        cv2.putText(
            out,
            message[:60],
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 130, 255),
            2,
            cv2.LINE_AA,
        )

        y += 19

    if reference:

        out = draw_reference(
            out,
            reference,
        )

    return out


def create_checker(
    exercise_name,
    writer,
):

    config = load_config()

    exercise_config = config.get(
        exercise_name,
        {},
    )

    return EXERCISE_CHECKERS[
        exercise_name
    ](
        exercise_config,
        logger=writer,
    )


def run_exercise_live(
    exercise_name: str,
    source=0,
) -> Generator:

    if exercise_name not in (
        EXERCISE_CHECKERS
    ):
        raise ValueError(
            f"Unsupported exercise: "
            f"{exercise_name}"
        )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        LOG_DIR
        / f"{exercise_name}_session.csv"
    )

    new_file = (
        not log_path.exists()
        or log_path.stat().st_size == 0
    )

    log_file = log_path.open(
        "a",
        newline="",
        encoding="utf-8",
    )

    writer = csv.writer(
        log_file
    )

    if new_file:

        writer.writerow(
            [
                "timestamp",
                "exercise",
                "metric",
                "value",
                "note",
            ]
        )

    cap = None
    pose = None

    try:

        checker = create_checker(
            exercise_name,
            writer,
        )

        cap = open_source(
            source
        )

        # Keep your YOLO component in the pipeline.
        yolo_wrapper = (
            YOLOv8PoseWrapper()
        )

        pose = (
            mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )

        ref_fn = (
            REFERENCE_FUNCTIONS.get(
                exercise_name
            )
        )

        reference_poses = (
            ref_fn(num_frames=140)
            if ref_fn
            else []
        )

        reference_index = 0
        previous_time = time.perf_counter()

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            now = time.perf_counter()

            fps = (
                1.0
                / max(
                    now
                    - previous_time,
                    1e-6,
                )
            )

            previous_time = now

            # ----------------------------
            # YOLO pose
            # ----------------------------

            try:

                yolo_output = (
                    yolo_wrapper.findPose(
                        frame,
                        draw=False,
                    )
                )

                yolo_landmarks = (
                    yolo_output
                    if isinstance(
                        yolo_output,
                        dict,
                    )
                    else {}
                )

            except Exception:

                yolo_landmarks = {}

            # ----------------------------
            # MediaPipe pose
            # ----------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_results = pose.process(
                rgb
            )

            mp_landmarks = (
                media_pipe_to_dict(
                    mp_results
                )
            )

            # ----------------------------
            # Fusion
            # ----------------------------

            fused = fuse_landmarks(
                yolo_landmarks,
                mp_landmarks,
            )

            # ----------------------------
            # Exercise checker
            # ----------------------------

            result = checker.update(
                fused,
                t=time.time(),
            )

            # ----------------------------
            # Reference
            # ----------------------------

            reference = None

            if reference_poses:

                reference = (
                    reference_poses[
                        reference_index
                        % len(
                            reference_poses
                        )
                    ]
                )

                reference_index += 1

            # ----------------------------
            # Draw
            # ----------------------------

            output = annotate(
                frame,
                exercise_name,
                result,
                mp_results,
                reference,
                fps,
            )

            yield output, result

    finally:

        if pose is not None:
            pose.close()

        if cap is not None:
            cap.release()

        log_file.close()


def run_local(
    exercise_name,
    source=0,
):

    for frame, _ in run_exercise_live(
        exercise_name,
        source,
    ):

        cv2.imshow(
            (
                "Rehab - "
                + EXERCISE_TITLES[
                    exercise_name
                ]
            ),
            frame,
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):
            break

    cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "YOLO + MediaPipe "
            "rehabilitation assessment"
        )
    )

    parser.add_argument(
        "--exercise",
        required=True,
        choices=sorted(
            EXERCISE_CHECKERS
        ),
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Camera index or "
            "video path."
        ),
    )

    args = parser.parse_args()

    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    run_local(
        args.exercise,
        source,
    )


if __name__ == "__main__":
    main()