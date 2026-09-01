from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp


import cv2
import mediapipe as mp

from pose.yolov8_adapter import (
    YOLOv8PoseWrapper,
)

from exercises import (
    MiniSquatChecker,
    ShoulderFlexionChecker,
    ShoulderAbductionChecker,
    BicepsCurlChecker,
    CalfRaiseChecker,
    SingleLegStanceChecker,
)

from utils.landmarks import (
    fuse_landmarks,
    landmarks_to_dict,
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
    "mini_squat": MiniSquatChecker,
    "shoulder_flexion": ShoulderFlexionChecker,
    "shoulder_abduction": ShoulderAbductionChecker,
    "biceps_curl": BicepsCurlChecker,
    "calf_raise": CalfRaiseChecker,
    "single_leg_stance": SingleLegStanceChecker,
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


def open_source(
    source,
):

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


def media_pipe_to_dict(
    results,
):

    if (
        results is None
        or not results.pose_landmarks
    ):
        return {}

    result = {}

    enum = (
        mp.solutions.pose.PoseLandmark
    )

    for index, landmark in enumerate(
        results.pose_landmarks.landmark
    ):

        result[
            enum(index).name
        ] = (
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

    return result


def create_checker(
    exercise_name,
    writer,
):

    config = load_config()

    exercise_config = (
        config.get(
            exercise_name,
            {},
        )
    )

    checker_cls = (
        EXERCISE_CHECKERS[
            exercise_name
        ]
    )

    return checker_cls(
        exercise_config,
        logger=writer,
    )


def run_exercise_live(
    exercise_name,
    source=0,
):

    if (
        exercise_name
        not in EXERCISE_CHECKERS
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

        # ----------------------------
        # Visolus pose backend
        # ----------------------------

        pose_wrapper = YOLOv8PoseWrapper(
            model_path="yolov8n-pose.pt",
            confidence=0.35,
        )

        if not hasattr(
            pose_wrapper,
            "findPose",
        ):
            raise RuntimeError(
                "Loaded pose backend does not "
                "provide findPose(frame, draw=...)."
            )

        # ----------------------------
        # MediaPipe
        # ----------------------------

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
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

        previous_time = (
            time.perf_counter()
        )

        while True:

            ok, frame = (
                cap.read()
            )

            if not ok:
                break

            current_time = (
                time.perf_counter()
            )

            fps = 1.0 / max(
                current_time
                - previous_time,
                1e-6,
            )

            previous_time = (
                current_time
            )

            # ==========================================================
            # ==========================================================
            # YOLO POSE
            # ==========================================================

            try:
                yolo_landmarks = (
                    pose_wrapper.predict_landmarks(
                        frame
                    )
                )

            except Exception:
                yolo_landmarks = {}


            # ==========================================================
            # MEDIAPIPE POSE
            # ==========================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_results = pose.process(
                rgb
            )

            if mp_results.pose_landmarks:

                mp_landmarks = (
                    landmarks_to_dict(
                        mp_results.pose_landmarks.landmark
                    )
                )

            else:

                mp_landmarks = {}


            # ==========================================================
            # FUSION
            # ==========================================================

            fused_landmarks = fuse_landmarks(
                yolo_landmarks,
                mp_landmarks,
            )


            # ==========================================================
            # EXERCISE ANALYSIS
            # ==========================================================

            result = checker.update(
                fused_landmarks,
                t=time.time(),
            )
            # ------------------------
            # Visualization
            # ------------------------

            if mp_results.pose_landmarks:

                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    mp_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                )

            cv2.putText(
                frame,
                EXERCISE_TITLES[
                    exercise_name
                ],
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Reps: {result.get('reps', 0)}",
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 220, 120),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Stage: {result.get('stage', '-')}",
                (15, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (220, 220, 80),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            feedback = result.get(
                "feedback",
                [],
            )

            y = 150

            for message in feedback[:2]:

                cv2.putText(
                    frame,
                    str(message)[:55],
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (80, 120, 255),
                    2,
                    cv2.LINE_AA,
                )

                y += 20

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

            # Simple reference panel.
            if reference:

                h, w = frame.shape[:2]

                size = min(
                    180,
                    h // 4,
                )

                x0 = (
                    w
                    - size
                    - 15
                )

                y0 = 15

                cv2.rectangle(
                    frame,
                    (
                        x0 - 5,
                        y0 - 5,
                    ),
                    (
                        x0
                        + size
                        + 5,
                        y0
                        + size
                        + 5,
                    ),
                    (25, 25, 25),
                    -1,
                )

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

                def point_to_px(
                    point,
                ):
                    return (
                        x0
                        + int(
                            point[0]
                            * size
                        ),
                        y0
                        + int(
                            point[1]
                            * size
                        ),
                    )

                for a, b in connections:

                    if (
                        a in reference
                        and b in reference
                    ):

                        cv2.line(
                            frame,
                            point_to_px(
                                reference[a]
                            ),
                            point_to_px(
                                reference[b]
                            ),
                            (180, 180, 180),
                            2,
                            cv2.LINE_AA,
                        )

                for point in (
                    reference.values()
                ):

                    cv2.circle(
                        frame,
                        point_to_px(
                            point
                        ),
                        3,
                        (80, 220, 255),
                        -1,
                        cv2.LINE_AA,
                    )

            yield frame, result

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

    for frame, _ in (
        run_exercise_live(
            exercise_name,
            source,
        )
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

    parser = argparse.ArgumentParser()

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