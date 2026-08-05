# ============================================================
# constants.py — System-wide constants for EyeGuard AI
# ============================================================

# ── MediaPipe landmark indices ───────────────────────────────
# Left eye: upper/lower lid and inner/outer corners
LEFT_EYE_TOP    = [386, 374, 373, 390]
LEFT_EYE_BOTTOM = [263, 362, 382, 381, 380, 385, 384, 398]
LEFT_EYE_H      = [362, 263]          # horizontal endpoints for EAR

# Right eye: upper/lower lid and inner/outer corners
RIGHT_EYE_TOP    = [159, 145, 144, 153]
RIGHT_EYE_BOTTOM = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_H      = [33, 133]

# Standard 6-point EAR landmarks (left / right)
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# Pupil centres (used for IPD-based distance estimation)
LEFT_PUPIL  = 468   # MediaPipe iris centre (index 468, extended mesh)
RIGHT_PUPIL = 473

# Fallback pupil landmarks when iris landmarks unavailable
LEFT_PUPIL_FALLBACK  = 386
RIGHT_PUPIL_FALLBACK = 159

# ── EAR / Blink thresholds ───────────────────────────────────
EAR_THRESHOLD          = 0.21   # below this → eye considered closed
BLINK_CONSEC_FRAMES    = 2      # min consecutive frames below threshold to count 1 blink
HEALTHY_BLINK_RATE_MIN = 12     # blinks / min — clinical lower bound
HEALTHY_BLINK_RATE_MAX = 20     # blinks / min — clinical upper bound

# ── Screen-distance estimation ────────────────────────────────
# Average adult inter-pupillary distance in mm
AVG_IPD_MM              = 63.0
# Reference pixel-width of IPD when user sits at ~60 cm
REFERENCE_IPD_PIXELS    = 62.0
REFERENCE_DISTANCE_CM   = 60.0
SAFE_DISTANCE_CM        = 50.0   # anything below → high-risk flag

# ── Session fatigue ──────────────────────────────────────────
FATIGUE_LOG_BASE        = 30     # every 30-min block increases fatigue score
MAX_SESSION_MINUTES     = 120    # above this → maximum fatigue contribution

# ── Eye Strain Score (ESS) weights — must sum to 1.0 ─────────
W_BLINK    = 0.30
W_DISTANCE = 0.25
W_FATIGUE  = 0.25
W_GAZE     = 0.20

# ── ESS classification bands ─────────────────────────────────
ESS_LOW_MAX    = 33   # 0–33   → Low strain
ESS_MEDIUM_MAX = 66   # 34–66  → Medium strain
# 67–100 → High strain

# ── Alert cooldown (seconds between repeated alerts) ─────────
ALERT_COOLDOWN_SEC = 60

# ── Adaptive baseline ────────────────────────────────────────
BASELINE_WARMUP_SECONDS  = 30    # seconds before baseline is "trusted"
BASELINE_ALPHA           = 0.05  # EMA smoothing factor for baseline updates

# ── Colours used across the UI ───────────────────────────────
COLOR_LOW    = "#2ecc71"
COLOR_MEDIUM = "#f39c12"
COLOR_HIGH   = "#e74c3c"
COLOR_IDLE   = "#95a5a6"
