# ============================================================
# scoring.py — Stage 4: Eye Strain Score (ESS) Engine
# ============================================================
"""
Computes a normalised Eye Strain Score (0–100) from four
physiological / behavioural signals:

  Component 1 — Blink Deficit
      Healthy adults blink 15–20 times/min.  Fewer blinks → dry eyes.
      Score rises as blink_rate falls below the healthy lower bound.

  Component 2 — Distance Risk
      Sitting closer than 50 cm strains accommodation muscles.
      Score rises as distance_cm falls below SAFE_DISTANCE_CM.

  Component 3 — Session Fatigue
      Continuous screen time causes cumulative eye-muscle fatigue.
      Score increases logarithmically over time (diminishing marginal
      returns model — each extra 30 min hurts less than the last).

  Component 4 — Gaze / Head Strain
      Rapid EAR variance → user squinting or gaze-darting.
      Already normalised [0,1] from the feature extractor.

Final ESS = w1*C1 + w2*C2 + w3*C3 + w4*C4,  scaled to 0–100.
"""

import math
import numpy as np
from typing import Dict

from utils.constants import (
    HEALTHY_BLINK_RATE_MIN, HEALTHY_BLINK_RATE_MAX,
    SAFE_DISTANCE_CM,
    FATIGUE_LOG_BASE, MAX_SESSION_MINUTES,
    W_BLINK, W_DISTANCE, W_FATIGUE, W_GAZE,
    ESS_LOW_MAX, ESS_MEDIUM_MAX,
    COLOR_LOW, COLOR_MEDIUM, COLOR_HIGH, COLOR_IDLE,
)


class ScoringEngine:
    """
    Converts raw physiological features → Eye Strain Score.

    The engine is stateless (all state lives in the feature dict),
    making it easy to unit-test and replay.
    """

    # ── public API ───────────────────────────────────────────

    def compute(
        self,
        features: Dict,
        baseline: Dict,
    ) -> Dict:
        """
        Compute the Eye Strain Score and its components.

        Parameters
        ----------
        features : dict (output of FeatureExtractor.extract)
        baseline : dict (output of AdaptiveBaseline.get_baseline)
            Contains personalised thresholds: blink_rate, distance_cm

        Returns
        -------
        result : dict
            ess, strain_level, color,
            c_blink, c_distance, c_fatigue, c_gaze  (each 0–100)
        """
        c_blink    = self._blink_deficit(features, baseline)
        c_distance = self._distance_risk(features, baseline)
        c_fatigue  = self._session_fatigue(features)
        c_gaze     = self._gaze_strain(features)

        ess_raw = (
            W_BLINK    * c_blink
            + W_DISTANCE * c_distance
            + W_FATIGUE  * c_fatigue
            + W_GAZE     * c_gaze
        )
        ess = float(np.clip(ess_raw, 0.0, 100.0))

        strain_level, color = self._classify(ess)

        return {
            "ess":          round(ess, 1),
            "strain_level": strain_level,
            "color":        color,
            "c_blink":      round(c_blink,    1),
            "c_distance":   round(c_distance, 1),
            "c_fatigue":    round(c_fatigue,  1),
            "c_gaze":       round(c_gaze,     1),
        }

    # ── component scorers ────────────────────────────────────

    @staticmethod
    def _blink_deficit(features: Dict, baseline: Dict) -> float:
        """
        Blink Deficit Score — 0 (healthy rate) to 100 (no blinking).

        Uses the user's personalised normal blink rate as the target.
        If the user naturally blinks less, the threshold is adjusted down.
        """
        blink_rate = features["blink_rate"]
        # Personalised lower bound: midpoint between system min and user normal
        user_normal   = baseline.get("blink_rate", HEALTHY_BLINK_RATE_MIN)
        personal_floor = max(
            HEALTHY_BLINK_RATE_MIN * 0.7,          # never go below 70 % of clinical min
            (user_normal + HEALTHY_BLINK_RATE_MIN) / 2.0,
        )

        if blink_rate >= personal_floor:
            return 0.0   # healthy → no penalty

        # Linear ramp: score = 100 when blink_rate == 0
        score = (1.0 - blink_rate / personal_floor) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    @staticmethod
    def _distance_risk(features: Dict, baseline: Dict) -> float:
        """
        Distance Risk Score — 0 (safe distance) to 100 (very close).

        Personalised safe distance is the max of the system constant and
        the user's habitual distance minus a small tolerance.
        """
        dist = features["distance_cm"]
        user_normal_dist = baseline.get("distance_cm", SAFE_DISTANCE_CM)
        personal_safe    = max(SAFE_DISTANCE_CM, user_normal_dist * 0.85)

        if dist >= personal_safe:
            return 0.0

        # Linear ramp: score = 100 at dist == 20 cm (very close)
        score = (1.0 - (dist - 20.0) / max(personal_safe - 20.0, 1.0)) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    @staticmethod
    def _session_fatigue(features: Dict) -> float:
        """
        Session Fatigue Score — logarithmic growth over continuous usage.

        f(t) = 100 * log(1 + t / BASE) / log(1 + MAX / BASE)

        At t=0   → 0
        At t=30  → ~32  (first half-hour)
        At t=60  → ~52  (one hour)
        At t=120 → ~100 (two hours, ceiling)
        """
        t_min = features["session_minutes"]
        if t_min <= 0:
            return 0.0
        t_capped = min(t_min, MAX_SESSION_MINUTES)
        numerator   = math.log(1.0 + t_capped / FATIGUE_LOG_BASE)
        denominator = math.log(1.0 + MAX_SESSION_MINUTES / FATIGUE_LOG_BASE)
        return float(np.clip(100.0 * numerator / denominator, 0.0, 100.0))

    @staticmethod
    def _gaze_strain(features: Dict) -> float:
        """
        Gaze / Head Strain Score — already normalised [0,1] from feature extractor.
        Scale to [0, 100].
        """
        return float(np.clip(features["gaze_deviation"] * 100.0, 0.0, 100.0))

    # ── classification ───────────────────────────────────────

    @staticmethod
    def _classify(ess: float):
        """Map ESS → (label, hex-colour)."""
        if ess <= ESS_LOW_MAX:
            return "Low", COLOR_LOW
        elif ess <= ESS_MEDIUM_MAX:
            return "Medium", COLOR_MEDIUM
        else:
            return "High", COLOR_HIGH
