"""
Streamlit interface for the rehabilitation system.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from exercises import (
    BicepsCurlChecker,
    CalfRaiseChecker,
    MiniSquatChecker,
    ShoulderAbductionChecker,
    ShoulderFlexionChecker,
    SingleLegStanceChecker,
)

from run_local import (
    EXERCISE_CHECKERS,
    EXERCISE_TITLES,
    run_exercise_live,
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


st.set_page_config(
    page_title="AI Rehabilitation Monitor",
    page_icon="🏥",
    layout="wide",
)


def load_config():

    if not CONFIG_PATH.exists():

        st.error(
            f"Missing config: "
            f"{CONFIG_PATH}"
        )

        st.stop()

    return json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


cfg = load_config()


st.title(
    "🏥 AI Rehabilitation Monitor"
)

st.caption(
    "YOLOv8 Pose + MediaPipe + "
    "biomechanical exercise analysis"
)


# ==========================================================
# SIDEBAR
# ==========================================================

exercise = st.sidebar.selectbox(
    "Exercise",
    options=list(
        EXERCISE_CHECKERS
    ),
    format_func=lambda key:
        EXERCISE_TITLES[key],
)


source_text = st.sidebar.text_input(
    "Source",
    value="0",
    help=(
        "0 = webcam. "
        "You can also provide a "
        "video file path."
    ),
)


source = (
    int(source_text)
    if source_text.isdigit()
    else source_text
)


exercise_cfg = cfg.get(
    exercise,
    {},
)


st.sidebar.markdown(
    "---"
)

st.sidebar.subheader(
    "Configuration"
)

st.sidebar.json(
    exercise_cfg
)


# ==========================================================
# MAIN INFORMATION
# ==========================================================

st.header(
    EXERCISE_TITLES[exercise]
)


c1, c2, c3 = st.columns(3)


with c1:
    st.metric(
        "Exercise",
        EXERCISE_TITLES[exercise],
    )


with c2:
    st.metric(
        "Side",
        str(
            exercise_cfg.get(
                "side",
                "both",
            )
        ).title(),
    )


with c3:
    st.metric(
        "Smoothing",
        exercise_cfg.get(
            "smoothing_window",
            "—",
        ),
    )


st.markdown(
    "---"
)


# ==========================================================
# LIVE MODE
# ==========================================================

start = st.button(
    "▶ Start Live Session",
    type="primary",
)


if start:

    frame_placeholder = (
        st.empty()
    )

    metrics_placeholder = (
        st.empty()
    )

    feedback_placeholder = (
        st.empty()
    )

    try:

        for frame, result in (
            run_exercise_live(
                exercise,
                source,
            )
        ):

            frame_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True,
            )

            with metrics_placeholder.container():

                a, b, c = (
                    st.columns(3)
                )

                a.metric(
                    "Repetitions",
                    result.get(
                        "reps",
                        0,
                    ),
                )

                b.metric(
                    "Stage",
                    result.get(
                        "stage",
                        "—",
                    ),
                )

                c.metric(
                    "Status",
                    result.get(
                        "status",
                        "—",
                    ),
                )

            feedback = result.get(
                "feedback",
                [],
            )

            if feedback:

                feedback_placeholder.warning(
                    " • ".join(
                        feedback
                    )
                )

            else:

                feedback_placeholder.success(
                    "No active form warning."
                )

    except Exception as exc:

        st.error(
            "Live session failed: "
            f"{exc}"
        )


# ==========================================================
# SESSION LOGS
# ==========================================================

st.markdown(
    "---"
)

st.header(
    "📊 Session Logs"
)


if not LOG_DIR.exists():

    st.info(
        "No session logs yet."
    )

else:

    csv_files = sorted(
        LOG_DIR.glob(
            "*.csv"
        ),
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:

        st.info(
            "No session logs yet."
        )

    else:

        for csv_path in csv_files[:10]:

            with st.expander(
                csv_path.name
            ):

                try:

                    df = pd.read_csv(
                        csv_path,
                        on_bad_lines="skip",
                    )

                    st.dataframe(
                        df.tail(100),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download CSV",
                        df.to_csv(
                            index=False
                        ),
                        file_name=(
                            csv_path.name
                        ),
                        mime="text/csv",
                        key=(
                            "download_"
                            + csv_path.name
                        ),
                    )

                except Exception as exc:

                    st.error(
                        f"Could not read "
                        f"log: {exc}"
                    )