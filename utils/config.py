# ============================================================
# config.py — Runtime configuration for EyeGuard AI
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR            = os.path.join(BASE_DIR, "data")
USER_PROFILES_PATH  = os.path.join(DATA_DIR, "user_profiles.json")

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX        = 0      # default webcam
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
TARGET_FPS          = 30

# ── MediaPipe FaceMesh ───────────────────────────────────────
FACE_MESH_MAX_FACES         = 1
FACE_MESH_REFINE_LANDMARKS  = True   # enables iris landmarks (468/473)
FACE_MESH_MIN_DETECTION_CONF = 0.5
FACE_MESH_MIN_TRACKING_CONF  = 0.5

# ── Analytics history window ─────────────────────────────────
HISTORY_MAX_POINTS  = 200    # number of data-points kept in memory for plots

# ── Streamlit page settings ──────────────────────────────────
PAGE_TITLE          = "EyeGuard AI — Eye Strain Detection"
PAGE_ICON           = "👁️"
LAYOUT              = "wide"

# ── Default user name (overrideable via UI) ──────────────────
DEFAULT_USER        = "default_user"
