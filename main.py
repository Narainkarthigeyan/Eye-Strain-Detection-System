# ============================================================
# main.py — Core Pipeline Runner (CLI / headless mode)
# ============================================================
"""
Runs the full EyeGuard AI pipeline without the Streamlit UI.
Useful for:
  - Debugging the CV pipeline
  - Running on a server / in a container
  - Integration testing

Press  Q  to quit.
"""

import cv2
import time
import numpy as np
from collections import deque

from modules.detector  import FaceDetector
from modules.features  import FeatureExtractor
from modules.scoring   import ScoringEngine
from modules.decision  import DecisionEngine
from modules.adaptive  import AdaptiveBaseline
from utils.config      import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, DEFAULT_USER
from utils.constants   import COLOR_LOW, COLOR_MEDIUM, COLOR_HIGH


def _ess_to_bgr(color_hex: str) -> tuple:
    """Convert '#rrggbb' hex to OpenCV BGR tuple."""
    color_hex = color_hex.lstrip("#")
    r, g, b   = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
    return (b, g, r)


def run_pipeline(user_id: str = DEFAULT_USER) -> None:
    """
    Launch the real-time eye strain detection pipeline.

    Opens the default webcam, processes frames, and prints
    live metrics + alerts to the terminal.
    Draws a minimal HUD overlay on the webcam window.
    """
    # ── initialise components ────────────────────────────────
    detector  = FaceDetector()
    extractor = FeatureExtractor()
    scorer    = ScoringEngine()
    decider   = DecisionEngine()
    baseline  = AdaptiveBaseline(user_id=user_id)

    # ── open camera ──────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {CAMERA_INDEX}. "
            "Check that your webcam is connected and not in use."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("EyeGuard AI — real-time pipeline started.  Press [Q] to quit.\n")

    frame_delay = 1.0 / TARGET_FPS
    prev_time   = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to capture frame — retrying …")
                time.sleep(0.1)
                continue

            # ── Stage 2: Detect ──────────────────────────────
            landmarks, frame_rgb = detector.process(frame)

            if landmarks is None:
                # No face detected — show warning on frame
                cv2.putText(
                    frame, "No face detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
                cv2.imshow("EyeGuard AI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── Stage 3: Extract features ────────────────────
            features = extractor.extract(landmarks, detector, frame.shape)

            # ── Stage 6: Update adaptive baseline ────────────
            baseline.update(features)

            # ── Stage 4: Compute ESS ─────────────────────────
            bl      = baseline.get_baseline()
            result  = scorer.compute(features, bl)

            # ── Stage 5: Decision / alerts ───────────────────
            alerts  = decider.evaluate(result, features)
            for alert in alerts:
                print(f"\n ALERT [{alert.severity.upper()}]  {alert.title}")
                print(f"    {alert.message}\n")

            # ── HUD overlay ──────────────────────────────────
            _draw_hud(frame, features, result)

            # ── FPS throttle ─────────────────────────────────
            now   = time.time()
            elapsed = now - prev_time
            wait  = max(1, int((frame_delay - elapsed) * 1000))
            prev_time = now

            cv2.imshow("EyeGuard AI", frame)
            if cv2.waitKey(wait) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # ── graceful shutdown ─────────────────────────────────
        session_min = features.get("session_minutes", 0) if "features" in dir() else 0
        baseline.end_session(session_min)
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Session ended.  Baseline saved.")


def _draw_hud(frame: np.ndarray, features: dict, result: dict) -> None:
    """Draw a minimal HUD (metrics overlay) on the frame in-place."""
    h, w = frame.shape[:2]
    color = _ess_to_bgr(result["color"])

    # Semi-transparent background rectangle for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 160), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"ESS     : {result['ess']:.1f}  [{result['strain_level']}]",
        f"Blink/m : {features['blink_rate']:.1f}",
        f"Distance: {features['distance_cm']:.0f} cm",
        f"Session : {features['session_minutes']:.1f} min",
        f"EAR     : {features['ear']:.3f}",
    ]
    for idx, line in enumerate(lines):
        y_pos = 26 + idx * 26
        cv2.putText(
            frame, line,
            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA,
        )


if __name__ == "__main__":
    run_pipeline()
