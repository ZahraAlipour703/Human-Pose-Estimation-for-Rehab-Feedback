# utils/smoothing.py
import numpy as np
from collections import deque

def moving_average(values, window=5):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode="valid")

def ema_smoothing(values, alpha=0.7):
    """Exponential moving average smoothing (fast, adaptive)."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return smoothed


def angle_between_3d(a, b, c):

    a = np.asarray(
        a[:3],
        dtype=float,
    )

    b = np.asarray(
        b[:3],
        dtype=float,
    )

    c = np.asarray(
        c[:3],
        dtype=float,
    )

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-12:
        return 0.0

    cosine = np.clip(
        np.dot(ba, bc)
        / denominator,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def angle_3pts(a, b, c):
    return angle_between_3d(
        a,
        b,
        c,
    )

class SimpleSmoother:
    """Incremental moving average smoother (for streaming values)."""
    def __init__(self, window=5):
        self.q = deque(maxlen=window)

    def update(self, x):
        if x is None:
            return None
        self.q.append(float(x))
        return float(sum(self.q) / len(self.q))
