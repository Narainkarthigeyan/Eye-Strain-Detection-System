# ============================================================
# adaptive.py — Stage 6: Adaptive Per-User Baseline Learning
# ============================================================
"""
Maintains a per-user profile that captures each individual's
"normal" physiological behaviour:

  • normal_blink_rate   (blinks / min)
  • normal_distance_cm  (habitual seating distance)

These baselines are used by the ScoringEngine to personalise
alert thresholds — a user who naturally blinks 12 times / min
should not be penalised as heavily as one who normally blinks 18.

Persistence
-----------
Profiles are stored in data/user_profiles.json so baselines
survive across sessions.

Learning algorithm
------------------
Exponential Moving Average (EMA) with a small alpha so the
baseline changes gradually and isn't corrupted by brief anomalies.

    baseline_new = (1 - alpha) * baseline_old + alpha * observed
"""

import json
import os
import time
from typing import Dict, Optional

from utils.constants import (
    BASELINE_ALPHA,
    BASELINE_WARMUP_SECONDS,
    HEALTHY_BLINK_RATE_MIN,
    REFERENCE_DISTANCE_CM,
)
from utils.config import USER_PROFILES_PATH, DATA_DIR


# ── default profile values ───────────────────────────────────

DEFAULT_PROFILE = {
    "blink_rate":       float(HEALTHY_BLINK_RATE_MIN),
    "distance_cm":      float(REFERENCE_DISTANCE_CM),
    "sessions":         0,
    "total_minutes":    0.0,
    "last_updated":     None,
}


class AdaptiveBaseline:
    """
    Loads, updates, and persists per-user physiological baselines.

    Parameters
    ----------
    user_id : str   — unique identifier (e.g., "alice", "default_user")
    """

    def __init__(self, user_id: str = "default_user") -> None:
        self.user_id     = user_id
        self._profile    = self._load_profile()
        self._session_start = time.time()

        # Warm-up accumulator: don't update baseline until we have
        # at least BASELINE_WARMUP_SECONDS of stable data
        self._warmup_done   = False
        self._obs_blink:    list = []
        self._obs_distance: list = []

    # ── public API ───────────────────────────────────────────

    def get_baseline(self) -> Dict:
        """Return current personalised baseline thresholds."""
        return {
            "blink_rate":   self._profile["blink_rate"],
            "distance_cm":  self._profile["distance_cm"],
        }

    def update(self, features: Dict) -> None:
        """
        Feed one frame's features to the adaptive learner.

        During warm-up: accumulate observations.
        After warm-up:  apply EMA update every frame.
        """
        elapsed = time.time() - self._session_start

        # Ignore frames where blink_rate is 0 (insufficient data)
        if features["blink_rate"] > 0:
            self._obs_blink.append(features["blink_rate"])
        if features["distance_cm"] > 0:
            self._obs_distance.append(features["distance_cm"])

        # ── warm-up phase ────────────────────────────────────
        if not self._warmup_done:
            if elapsed < BASELINE_WARMUP_SECONDS:
                return  # still collecting warm-up data
            # Warm-up complete — seed baseline from observed average
            if self._obs_blink:
                self._profile["blink_rate"] = float(
                    sum(self._obs_blink) / len(self._obs_blink)
                )
            if self._obs_distance:
                self._profile["distance_cm"] = float(
                    sum(self._obs_distance) / len(self._obs_distance)
                )
            self._warmup_done = True
            self._obs_blink.clear()
            self._obs_distance.clear()
            return

        # ── live EMA updates ─────────────────────────────────
        if features["blink_rate"] > 0:
            self._profile["blink_rate"] = self._ema(
                self._profile["blink_rate"], features["blink_rate"]
            )
        if features["distance_cm"] > 0:
            self._profile["distance_cm"] = self._ema(
                self._profile["distance_cm"], features["distance_cm"]
            )

    def end_session(self, session_minutes: float) -> None:
        """
        Call at the end of each session to persist the profile.
        Updates cumulative statistics.
        """
        self._profile["sessions"]      = self._profile.get("sessions", 0) + 1
        self._profile["total_minutes"] = (
            self._profile.get("total_minutes", 0.0) + session_minutes
        )
        self._profile["last_updated"]  = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_profile()

    def get_profile_summary(self) -> Dict:
        """Return the full profile dict for display in the UI."""
        return dict(self._profile)

    # ── private helpers ──────────────────────────────────────

    @staticmethod
    def _ema(old: float, new: float) -> float:
        return (1.0 - BASELINE_ALPHA) * old + BASELINE_ALPHA * new

    def _load_profile(self) -> Dict:
        """Load the user's profile from disk, or create a fresh one."""
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(USER_PROFILES_PATH):
            return dict(DEFAULT_PROFILE)

        try:
            with open(USER_PROFILES_PATH, "r") as f:
                all_profiles = json.load(f)
            return all_profiles.get(self.user_id, dict(DEFAULT_PROFILE))
        except (json.JSONDecodeError, OSError):
            return dict(DEFAULT_PROFILE)

    def _save_profile(self) -> None:
        """Persist the updated profile to disk."""
        os.makedirs(DATA_DIR, exist_ok=True)
        all_profiles: Dict = {}
        if os.path.exists(USER_PROFILES_PATH):
            try:
                with open(USER_PROFILES_PATH, "r") as f:
                    all_profiles = json.load(f)
            except (json.JSONDecodeError, OSError):
                all_profiles = {}

        all_profiles[self.user_id] = self._profile
        with open(USER_PROFILES_PATH, "w") as f:
            json.dump(all_profiles, f, indent=2)
