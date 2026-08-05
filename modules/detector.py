# ============================================================
# detector.py — Stage 2: Face & Eye Detection via MediaPipe
# ============================================================
"""
Wraps MediaPipe FaceMesh to provide a clean interface:
  - initialise once, reuse across frames
  - returns normalised + pixel-space landmarks
  - gracefully handles no-face / multi-face scenarios
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

from utils.config import (
    FACE_MESH_MAX_FACES,
    FACE_MESH_REFINE_LANDMARKS,
    FACE_MESH_MIN_DETECTION_CONF,
    FACE_MESH_MIN_TRACKING_CONF,
)


class FaceDetector:
    """
    Thin wrapper around MediaPipe FaceMesh.

    Usage
    -----
    detector = FaceDetector()
    landmarks, frame_rgb = detector.process(frame_bgr)
    """

    def __init__(self) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing   = mp.solutions.drawing_utils
        self._mp_styles    = mp.solutions.drawing_styles

        self.face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=FACE_MESH_MAX_FACES,
            refine_landmarks=FACE_MESH_REFINE_LANDMARKS,
            min_detection_confidence=FACE_MESH_MIN_DETECTION_CONF,
            min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONF,
        )

    # ── public API ───────────────────────────────────────────

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[List], np.ndarray]:
        """
        Run FaceMesh inference on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw webcam frame in BGR colour space.

        Returns
        -------
        landmarks : list of NormalizedLandmark | None
            468 (+10 iris) landmarks for the first detected face,
            or None if no face is found.
        frame_rgb : np.ndarray
            The frame converted to RGB (side-effect free copy).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False          # perf: skip copy inside MP
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark, frame_rgb
        return None, frame_rgb

    def get_pixel_coords(
        self,
        landmarks,
        indices: List[int],
        frame_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Convert normalised landmarks to integer pixel coordinates.

        Parameters
        ----------
        landmarks : list[NormalizedLandmark]
        indices   : list[int]   — landmark indices to extract
        frame_shape : (H, W, C)

        Returns
        -------
        coords : np.ndarray, shape (N, 2), dtype int
        """
        h, w = frame_shape[:2]
        coords = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
            dtype=np.int32,
        )
        return coords

    def get_normalised_coords(
        self,
        landmarks,
        indices: List[int],
    ) -> np.ndarray:
        """
        Return normalised (x, y, z) for selected landmarks.

        Returns
        -------
        coords : np.ndarray, shape (N, 3)
        """
        return np.array(
            [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in indices],
            dtype=np.float32,
        )

    def draw_landmarks(self, frame_bgr: np.ndarray, landmarks) -> np.ndarray:
        """
        Draw eye-contour landmarks on the frame (debug / display).
        Returns annotated frame.
        """
        annotated = frame_bgr.copy()
        # draw tessellation lightly so the face mesh is visible
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=mp.solutions.face_mesh.FaceMesh,
            connections=self._mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
        )
        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_mesh.close()
