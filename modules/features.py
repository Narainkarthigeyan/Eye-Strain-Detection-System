# ============================================================
# features.py — Stage 3: Physiological Feature Extraction
# ============================================================
"""
Extracts four key signals every frame:
  1. Eye Aspect Ratio (EAR)          — eye openness
  2. Blink detection & rate          — blinks per minute
  3. Screen distance (cm)            — via inter-pupillary distance
  4. Session duration (minutes)      — continuous screen time
  5. Gaze deviation (optional)       — head-pose horizontal angle
"""

import time
import math
import numpy as np
from collections import deque
from typing import Optional, Tuple

from utils.constants import (
    LEFT_EYE_EAR, RIGHT_EYE_EAR,
    LEFT_PUPIL_FALLBACK, RIGHT_PUPIL_FALLBACK,
    EAR_THRESHOLD, BLINK_CONSEC_FRAMES,
    AVG_IPD_MM, REFERENCE_IPD_PIXELS, REFERENCE_DISTANCE_CM,
)


# ── helpers ──────────────────────────────────────────────────

def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """Return Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(p1 - p2))


def compute_ear(eye_pts: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio for one eye.

    Formula (Soukupová & Čech, 2016):
        EAR = (||p2−p6|| + ||p3−p5||) / (2 * ||p1−p4||)

    Parameters
    ----------
    eye_pts : np.ndarray, shape (6, 2)
        Pixel coordinates for the 6 standard EAR landmarks in order:
        [p1(left corner), p2(upper-left), p3(upper-right),
         p4(right corner), p5(lower-right), p6(lower-left)]

    Returns
    -------
    ear : float
    """
    A = _euclidean(eye_pts[1], eye_pts[5])   # vertical 1
    B = _euclidean(eye_pts[2], eye_pts[4])   # vertical 2
    C = _euclidean(eye_pts[0], eye_pts[3])   # horizontal
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


# ── FeatureExtractor ─────────────────────────────────────────

class FeatureExtractor:
    """
    Stateful extractor that accumulates blink history and computes
    all physiological features for the scoring engine.

    Call `reset_session()` at the start of each new recording session.
    """

    def __init__(self) -> None:
        # blink state
        self._frames_below_threshold: int = 0
        self._total_blinks: int = 0
        self._blink_timestamps: deque = deque(maxlen=200)  # rolling 60-s window

        # session timer
        self._session_start: float = time.time()

        # gaze / head-pose state (rolling EAR for gaze proxy)
        self._ear_history: deque = deque(maxlen=30)

    # ── public API ───────────────────────────────────────────

    def extract(
        self,
        landmarks,
        detector,
        frame_shape: Tuple[int, int, int],
    ) -> dict:
        """
        Run all feature computations for one frame.

        Parameters
        ----------
        landmarks : MediaPipe landmark list (468 or 478 points)
        detector  : FaceDetector instance (for coordinate helpers)
        frame_shape : (H, W, C)

        Returns
        -------
        features : dict with keys:
            ear, blink_rate, distance_cm, session_minutes,
            gaze_deviation, blink_count, eye_closed
        """
        # ── 1. EAR ───────────────────────────────────────────
        left_pts  = detector.get_pixel_coords(landmarks, LEFT_EYE_EAR,  frame_shape)
        right_pts = detector.get_pixel_coords(landmarks, RIGHT_EYE_EAR, frame_shape)
        ear_left  = compute_ear(left_pts)
        ear_right = compute_ear(right_pts)
        ear_avg   = (ear_left + ear_right) / 2.0
        self._ear_history.append(ear_avg)

        # ── 2. Blink detection ───────────────────────────────
        eye_closed = ear_avg < EAR_THRESHOLD
        if eye_closed:
            self._frames_below_threshold += 1
        else:
            if self._frames_below_threshold >= BLINK_CONSEC_FRAMES:
                self._total_blinks += 1
                self._blink_timestamps.append(time.time())
            self._frames_below_threshold = 0

        blink_rate = self._compute_blink_rate()

        # ── 3. Screen distance ───────────────────────────────
        distance_cm = self._compute_distance(landmarks, detector, frame_shape)

        # ── 4. Session duration ──────────────────────────────
        session_minutes = (time.time() - self._session_start) / 60.0

        # ── 5. Gaze deviation (head-pose proxy) ──────────────
        gaze_deviation = self._compute_gaze_deviation(landmarks)

        return {
            "ear":             round(ear_avg, 4),
            "ear_left":        round(ear_left, 4),
            "ear_right":       round(ear_right, 4),
            "eye_closed":      eye_closed,
            "blink_count":     self._total_blinks,
            "blink_rate":      round(blink_rate, 2),
            "distance_cm":     round(distance_cm, 1),
            "session_minutes": round(session_minutes, 2),
            "gaze_deviation":  round(gaze_deviation, 4),
        }

    def reset_session(self) -> None:
        """Reset all session-level counters (call at start of new session)."""
        self._frames_below_threshold = 0
        self._total_blinks = 0
        self._blink_timestamps.clear()
        self._session_start = time.time()
        self._ear_history.clear()

    # ── private helpers ──────────────────────────────────────

    def _compute_blink_rate(self) -> float:
        """
        Count blinks that occurred within the last 60 seconds.
        Returns blinks-per-minute.
        """
        now = time.time()
        cutoff = now - 60.0
        recent = [t for t in self._blink_timestamps if t >= cutoff]
        # Scale to per-minute based on elapsed time
        elapsed = min(now - self._session_start, 60.0)
        if elapsed < 5:
            return 0.0
        return len(recent) * (60.0 / elapsed)

    def _compute_distance(
        self,
        landmarks,
        detector,
        frame_shape: Tuple[int, int, int],
    ) -> float:
        """
        Estimate screen distance in cm using inter-pupillary distance (IPD).

        Uses iris landmarks (468/473) when available, falls back to
        inner-eye-corner landmarks.

        Physics:
            distance ∝ (reference_ipd_pixels / measured_ipd_pixels)
                        * reference_distance_cm
        """
        h, w = frame_shape[:2]
        n_landmarks = len(landmarks)

        if n_landmarks > 468:
            # Iris landmarks available
            lp = landmarks[468]
            rp = landmarks[473]
            left_px  = np.array([lp.x * w, lp.y * h])
            right_px = np.array([rp.x * w, rp.y * h])
        else:
            # Fallback: use inner eye corners
            lc = landmarks[LEFT_PUPIL_FALLBACK]
            rc = landmarks[RIGHT_PUPIL_FALLBACK]
            left_px  = np.array([lc.x * w, lc.y * h])
            right_px = np.array([rc.x * w, rc.y * h])

        ipd_pixels = _euclidean(left_px, right_px)
        if ipd_pixels < 1.0:
            return REFERENCE_DISTANCE_CM   # guard against degenerate case

        distance_cm = (REFERENCE_IPD_PIXELS / ipd_pixels) * REFERENCE_DISTANCE_CM
        # Clamp to plausible range
        return float(np.clip(distance_cm, 20.0, 120.0))

    def _compute_gaze_deviation(self, landmarks) -> float:
        """
        Lightweight head-pose/gaze proxy:
        Returns the variance of the EAR history — high variance signals
        rapid eye movement or squinting (surrogate for gaze strain).
        Normalised to [0, 1].
        """
        if len(self._ear_history) < 5:
            return 0.0
        var = float(np.var(list(self._ear_history)))
        # Typical EAR variance sits below 0.005; scale to [0,1]
        return float(np.clip(var / 0.005, 0.0, 1.0))
