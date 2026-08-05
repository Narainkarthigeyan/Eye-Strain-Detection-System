# ============================================================
# decision.py — Stage 5: Adaptive Alert / Decision Engine
# ============================================================
"""
Translates the Eye Strain Score into actionable, context-aware
alerts.  Alerts are suppressed during cooldown periods to avoid
notification fatigue (a key UX failing of static reminder apps).

Alert types
-----------
BLINK_LOW       Triggered when blink deficit component is high
DISTANCE_CLOSE  Triggered when user is too close to screen
BREAK_NEEDED    Triggered when session fatigue component is high
HIGH_STRAIN     Triggered when overall ESS is in the High band
"""

import time
from typing import Dict, List, Optional, Tuple

from utils.constants import (
    ALERT_COOLDOWN_SEC,
    ESS_LOW_MAX, ESS_MEDIUM_MAX,
    HEALTHY_BLINK_RATE_MIN,
    SAFE_DISTANCE_CM,
)


# ── Alert definitions ────────────────────────────────────────

ALERT_DEFINITIONS = {
    "BLINK_LOW": {
        "title":   "👁️  Low Blink Rate Detected",
        "message": "You are blinking less than normal. "
                   "Consciously blink 10 times now to re-lubricate your eyes.",
        "severity": "medium",
    },
    "DISTANCE_CLOSE": {
        "title":   "📏  Too Close to Screen",
        "message": "You are sitting too close to your screen. "
                   "Move back to at least 50–60 cm for comfortable viewing.",
        "severity": "high",
    },
    "BREAK_NEEDED": {
        "title":   "⏱️  Take a Screen Break",
        "message": "You have been looking at the screen for a long time. "
                   "Follow the 20-20-20 rule: look 20 ft away for 20 seconds.",
        "severity": "medium",
    },
    "HIGH_STRAIN": {
        "title":   "🚨  High Eye Strain Detected",
        "message": "Your Eye Strain Score is critically high. "
                   "Close your eyes for 30 seconds and rest before continuing.",
        "severity": "high",
    },
}


class AlertRecord:
    """Simple container for a triggered alert with timestamp."""
    __slots__ = ("alert_id", "title", "message", "severity", "timestamp")

    def __init__(self, alert_id: str) -> None:
        defn = ALERT_DEFINITIONS[alert_id]
        self.alert_id  = alert_id
        self.title     = defn["title"]
        self.message   = defn["message"]
        self.severity  = defn["severity"]
        self.timestamp = time.time()

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class DecisionEngine:
    """
    Evaluates current ESS / component scores and decides whether
    to fire alerts.  Maintains per-alert cooldown timers so the
    user is never spammed.
    """

    def __init__(self) -> None:
        # last_triggered[alert_id] → timestamp of most recent fire
        self._last_triggered: Dict[str, float] = {}
        # full history of all alerts (for analytics)
        self._alert_history: List[AlertRecord] = []

    # ── public API ───────────────────────────────────────────

    def evaluate(
        self,
        score_result: Dict,
        features: Dict,
    ) -> List[AlertRecord]:
        """
        Decide which (if any) alerts to fire this frame.

        Parameters
        ----------
        score_result : output of ScoringEngine.compute
        features     : output of FeatureExtractor.extract

        Returns
        -------
        new_alerts : list[AlertRecord]   — empty if nothing to fire
        """
        new_alerts: List[AlertRecord] = []

        ess          = score_result["ess"]
        c_blink      = score_result["c_blink"]
        c_distance   = score_result["c_distance"]
        c_fatigue    = score_result["c_fatigue"]
        strain_level = score_result["strain_level"]

        # Rule 1 — Low blink rate (component threshold > 60 out of 100)
        if c_blink > 60:
            alert = self._try_fire("BLINK_LOW")
            if alert:
                new_alerts.append(alert)

        # Rule 2 — Too close to screen (component > 50)
        if c_distance > 50:
            alert = self._try_fire("DISTANCE_CLOSE")
            if alert:
                new_alerts.append(alert)

        # Rule 3 — Session fatigue (component > 65 → ~45 min continuous use)
        if c_fatigue > 65:
            alert = self._try_fire("BREAK_NEEDED")
            if alert:
                new_alerts.append(alert)

        # Rule 4 — Overall high strain
        if strain_level == "High":
            alert = self._try_fire("HIGH_STRAIN")
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def get_alert_history(self) -> List[AlertRecord]:
        """Return full alert history (most recent last)."""
        return list(self._alert_history)

    def clear_history(self) -> None:
        """Clear history (call at start of new session)."""
        self._alert_history.clear()
        self._last_triggered.clear()

    def get_alert_counts(self) -> Dict[str, int]:
        """Count alerts per type for the analytics dashboard."""
        counts: Dict[str, int] = {k: 0 for k in ALERT_DEFINITIONS}
        for record in self._alert_history:
            counts[record.alert_id] = counts.get(record.alert_id, 0) + 1
        return counts

    # ── private helpers ──────────────────────────────────────

    def _try_fire(self, alert_id: str) -> Optional[AlertRecord]:
        """
        Fire an alert only if its cooldown has elapsed.
        Records the alert in history if fired.
        """
        now  = time.time()
        last = self._last_triggered.get(alert_id, 0.0)
        if (now - last) < ALERT_COOLDOWN_SEC:
            return None  # still in cooldown

        record = AlertRecord(alert_id)
        self._last_triggered[alert_id] = now
        self._alert_history.append(record)
        return record
