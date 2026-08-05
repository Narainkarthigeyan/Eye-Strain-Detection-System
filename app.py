# ============================================================
# app.py — EyeGuard AI  |  Streamlit Dashboard
# ============================================================
"""
Professional real-time eye strain monitoring dashboard.

Pages
-----
  1. Live Monitor   — webcam feed + live metrics + strain gauge
  2. Analytics      — ESS trend, blink history, alert log
  3. User Profile   — baseline settings, session statistics
"""

import time
import threading
import queue
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from typing import Optional

from modules.detector  import FaceDetector
from modules.features  import FeatureExtractor
from modules.scoring   import ScoringEngine
from modules.decision  import DecisionEngine
from modules.adaptive  import AdaptiveBaseline
from utils.config      import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    HISTORY_MAX_POINTS, PAGE_TITLE, PAGE_ICON, LAYOUT, DEFAULT_USER,
)
from utils.constants   import (
    COLOR_LOW, COLOR_MEDIUM, COLOR_HIGH, COLOR_IDLE,
    ESS_LOW_MAX, ESS_MEDIUM_MAX,
)


# ═══════════════════════════════════════════════════════════════
#  Page configuration (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
#  Global CSS / styling
# ═══════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #0e1117; }

    /* Metric card */
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

    /* Strain badge */
    .strain-badge {
        border-radius: 20px;
        padding: 6px 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    /* Alert box */
    .alert-box {
        border-left: 4px solid;
        padding: 10px 16px;
        margin-bottom: 8px;
        border-radius: 4px;
        background: #1e2130;
    }

    /* Hide default streamlit menu / footer */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
#  Session state initialisation
# ═══════════════════════════════════════════════════════════════

def _init_state() -> None:
    """Initialise all session-state variables on first run."""
    defaults = {
        "running":         False,
        "user_id":         DEFAULT_USER,
        "current_frame":   None,
        "latest_features": {},
        "latest_result":   {},
        "latest_alerts":   [],
        "ess_history":     deque(maxlen=HISTORY_MAX_POINTS),
        "blink_history":   deque(maxlen=HISTORY_MAX_POINTS),
        "distance_history":deque(maxlen=HISTORY_MAX_POINTS),
        "time_history":    deque(maxlen=HISTORY_MAX_POINTS),
        "alert_log":       [],
        "no_face_count":   0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ═══════════════════════════════════════════════════════════════
#  Background camera thread
# ═══════════════════════════════════════════════════════════════

class CameraThread(threading.Thread):
    """
    Runs the full detection pipeline in a background thread so the
    Streamlit main thread remains responsive.

    Writes results into st.session_state which is thread-safe for
    simple read/write operations.
    """

    def __init__(self, user_id: str) -> None:
        super().__init__(daemon=True)
        self.user_id   = user_id
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        # ── pipeline components ──────────────────────────────
        detector  = FaceDetector()
        extractor = FeatureExtractor()
        scorer    = ScoringEngine()
        decider   = DecisionEngine()
        baseline  = AdaptiveBaseline(user_id=self.user_id)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            st.session_state["running"] = False
            st.session_state["camera_error"] = (
                f"Cannot open camera {CAMERA_INDEX}. "
                "Ensure your webcam is connected and accessible."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        features: dict = {}
        result:   dict = {}

        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # Flip horizontally (mirror view)
                frame = cv2.flip(frame, 1)

                # ── Detection ────────────────────────────────
                landmarks, _ = detector.process(frame)

                if landmarks is None:
                    st.session_state["no_face_count"] += 1
                    st.session_state["current_frame"] = frame
                    time.sleep(0.03)
                    continue

                st.session_state["no_face_count"] = 0

                # ── Feature extraction ───────────────────────
                features = extractor.extract(landmarks, detector, frame.shape)

                # ── Adaptive baseline update ─────────────────
                baseline.update(features)

                # ── Scoring ──────────────────────────────────
                bl     = baseline.get_baseline()
                result = scorer.compute(features, bl)

                # ── Alerts ───────────────────────────────────
                alerts = decider.evaluate(result, features)
                if alerts:
                    st.session_state["latest_alerts"] = alerts
                    for a in alerts:
                        st.session_state["alert_log"].append({
                            "time":     time.strftime("%H:%M:%S"),
                            "title":    a.title,
                            "message":  a.message,
                            "severity": a.severity,
                        })

                # ── Draw eye landmarks on frame ───────────────
                _annotate_frame(frame, features, result)

                # ── Store in session state ────────────────────
                st.session_state["current_frame"]    = frame
                st.session_state["latest_features"]  = features
                st.session_state["latest_result"]    = result

                # ── Append to history ─────────────────────────
                t = features.get("session_minutes", 0)
                st.session_state["ess_history"].append(result.get("ess", 0))
                st.session_state["blink_history"].append(features.get("blink_rate", 0))
                st.session_state["distance_history"].append(features.get("distance_cm", 60))
                st.session_state["time_history"].append(t)

                time.sleep(0.04)   # ~25 fps

        finally:
            session_min = features.get("session_minutes", 0)
            baseline.end_session(session_min)
            cap.release()
            detector.close()
            st.session_state["running"] = False


def _annotate_frame(frame: np.ndarray, features: dict, result: dict) -> None:
    """Draw a minimal metrics overlay on the live frame."""
    color_hex = result.get("color", COLOR_IDLE)
    color_hex = color_hex.lstrip("#")
    r, g, b   = (int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    bgr       = (b, g, r)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (240, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"ESS     : {result.get('ess', 0):.1f}  [{result.get('strain_level','–')}]",
        f"Blink/m : {features.get('blink_rate', 0):.1f}",
        f"Distance: {features.get('distance_cm', 0):.0f} cm",
        f"Session : {features.get('session_minutes', 0):.1f} min",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 24 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, bgr, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/eye.png", width=72)
        st.title("EyeGuard AI")
        st.caption("Real-Time Eye Strain Detection")
        st.divider()

        user_id = st.text_input("👤  User ID", value=st.session_state["user_id"])
        st.session_state["user_id"] = user_id

        st.divider()
        page = st.radio(
            "Navigation",
            ["📷  Live Monitor", "📊  Analytics", "⚙️  User Profile"],
        )
        st.divider()

        # Start / Stop button
        if not st.session_state["running"]:
            if st.button("▶  Start Monitoring", use_container_width=True, type="primary"):
                _start_monitoring(user_id)
        else:
            if st.button("⏹  Stop Monitoring", use_container_width=True):
                _stop_monitoring()

        # Camera error
        if "camera_error" in st.session_state:
            st.error(st.session_state["camera_error"])

        st.divider()
        st.caption("v1.0.0 — CSE322 Patent Project")

    return page


def _start_monitoring(user_id: str) -> None:
    if not st.session_state["running"]:
        # Reset histories
        st.session_state["ess_history"].clear()
        st.session_state["blink_history"].clear()
        st.session_state["distance_history"].clear()
        st.session_state["time_history"].clear()
        st.session_state["alert_log"] = []
        st.session_state["latest_alerts"] = []

        thread = CameraThread(user_id=user_id)
        thread.start()
        st.session_state["running"]       = True
        st.session_state["_cam_thread"]   = thread


def _stop_monitoring() -> None:
    thread: Optional[CameraThread] = st.session_state.get("_cam_thread")
    if thread:
        thread.stop()
    st.session_state["running"] = False


# ═══════════════════════════════════════════════════════════════
#  Page 1 — Live Monitor
# ═══════════════════════════════════════════════════════════════

def page_live_monitor() -> None:
    st.header("📷  Live Monitor")

    if not st.session_state["running"]:
        st.info("Press **▶ Start Monitoring** in the sidebar to begin.", icon="ℹ️")
        return

    # ── layout: video | metrics ──────────────────────────────
    col_video, col_metrics = st.columns([3, 2], gap="large")

    # ── video feed ───────────────────────────────────────────
    video_placeholder = col_video.empty()

    # ── no-face warning ──────────────────────────────────────
    face_warning = col_video.empty()

    # ── alert banner ─────────────────────────────────────────
    alert_placeholder = col_video.empty()

    # ── metric boxes ─────────────────────────────────────────
    with col_metrics:
        st.subheader("Live Metrics")
        ph_ess        = st.empty()
        ph_strain     = st.empty()
        st.divider()
        ph_blink      = st.empty()
        ph_distance   = st.empty()
        ph_session    = st.empty()
        ph_ear        = st.empty()
        st.divider()
        st.caption("Component Scores")
        ph_components = st.empty()

    # ── real-time loop ────────────────────────────────────────
    while st.session_state.get("running", False):
        frame    = st.session_state.get("current_frame")
        features = st.session_state.get("latest_features", {})
        result   = st.session_state.get("latest_result",   {})
        alerts   = st.session_state.get("latest_alerts",   [])
        no_face  = st.session_state.get("no_face_count", 0)

        # Video
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # No-face warning
        if no_face > 10:
            face_warning.warning("⚠️  No face detected — please position yourself in front of the camera.")
        else:
            face_warning.empty()

        # Alerts
        if alerts:
            a = alerts[0]
            border = "#e74c3c" if a.severity == "high" else "#f39c12"
            alert_placeholder.markdown(
                f'<div class="alert-box" style="border-color:{border};">'
                f'<strong>{a.title}</strong><br>{a.message}</div>',
                unsafe_allow_html=True,
            )
        else:
            alert_placeholder.empty()

        if features and result:
            ess          = result.get("ess", 0)
            strain_level = result.get("strain_level", "–")
            color        = result.get("color", COLOR_IDLE)

            # ESS gauge
            ph_ess.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color};">{ess:.1f}</div>'
                f'<div class="metric-label">Eye Strain Score (0–100)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Strain badge
            ph_strain.markdown(
                f'<div style="text-align:center; margin:4px 0 12px 0;">'
                f'<span class="strain-badge" style="background:{color}20; color:{color}; '
                f'border: 1px solid {color};">{strain_level} Strain</span></div>',
                unsafe_allow_html=True,
            )

            # Individual metrics
            _render_metric(ph_blink,    "👁️  Blink Rate",
                           f"{features.get('blink_rate', 0):.1f}", "blinks/min")
            _render_metric(ph_distance, "📏  Distance",
                           f"{features.get('distance_cm', 0):.0f}", "cm from screen")
            _render_metric(ph_session,  "⏱️  Session Time",
                           f"{features.get('session_minutes', 0):.1f}", "minutes")
            _render_metric(ph_ear,      "📐  EAR",
                           f"{features.get('ear', 0):.3f}", "eye openness ratio")

            # Component scores
            ph_components.markdown(
                _component_bars(result),
                unsafe_allow_html=True,
            )

        time.sleep(0.1)


def _render_metric(placeholder, label: str, value: str, unit: str) -> None:
    placeholder.markdown(
        f'<div class="metric-card" style="text-align:left; padding:10px 16px;">'
        f'<span style="color:#8b949e; font-size:0.8rem;">{label}</span><br>'
        f'<span style="font-size:1.5rem; font-weight:700;">{value}</span> '
        f'<span style="color:#8b949e; font-size:0.8rem;">{unit}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _component_bars(result: dict) -> str:
    components = [
        ("Blink Deficit",   result.get("c_blink",    0), COLOR_MEDIUM),
        ("Distance Risk",   result.get("c_distance",  0), "#3498db"),
        ("Session Fatigue", result.get("c_fatigue",   0), "#9b59b6"),
        ("Gaze Strain",     result.get("c_gaze",      0), "#1abc9c"),
    ]
    html = ""
    for name, val, col in components:
        pct = min(val, 100)
        html += (
            f'<div style="margin-bottom:6px;">'
            f'<span style="font-size:0.75rem; color:#8b949e;">{name}: {pct:.0f}</span>'
            f'<div style="background:#2d3748; border-radius:4px; height:6px; margin-top:3px;">'
            f'<div style="width:{pct}%; background:{col}; height:6px; border-radius:4px;"></div>'
            f'</div></div>'
        )
    return html


# ═══════════════════════════════════════════════════════════════
#  Page 2 — Analytics
# ═══════════════════════════════════════════════════════════════

def page_analytics() -> None:
    st.header("📊  Session Analytics")

    ess_hist      = list(st.session_state["ess_history"])
    blink_hist    = list(st.session_state["blink_history"])
    distance_hist = list(st.session_state["distance_history"])
    alert_log     = st.session_state["alert_log"]

    if len(ess_hist) < 5:
        st.info("Analytics populate once monitoring has been running for a few seconds.", icon="ℹ️")
        return

    x = list(range(len(ess_hist)))

    # ── ESS over time ─────────────────────────────────────────
    st.subheader("Eye Strain Score Over Time")
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e2130")
    ax.plot(x, ess_hist, color="#e74c3c", linewidth=1.5, label="ESS")
    ax.axhline(ESS_LOW_MAX,    color=COLOR_LOW,    linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(ESS_MEDIUM_MAX, color=COLOR_MEDIUM, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.fill_between(x, ess_hist, alpha=0.15, color="#e74c3c")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Frame", color="#8b949e")
    ax.set_ylabel("ESS", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    patches = [
        mpatches.Patch(color=COLOR_LOW,    label="Low (0–33)"),
        mpatches.Patch(color=COLOR_MEDIUM, label="Medium (34–66)"),
        mpatches.Patch(color=COLOR_HIGH,   label="High (67–100)"),
    ]
    ax.legend(handles=patches, facecolor="#1e2130", labelcolor="#cdd9e5", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    # ── Blink rate trend ──────────────────────────────────────
    with col1:
        st.subheader("Blink Rate Trend")
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#1e2130")
        ax2.plot(x, blink_hist, color="#3498db", linewidth=1.5)
        ax2.axhline(15, color="#2ecc71", linestyle="--", linewidth=0.8, alpha=0.7, label="Healthy (15/min)")
        ax2.set_ylim(0, max(max(blink_hist) + 5, 30))
        ax2.set_xlabel("Frame", color="#8b949e")
        ax2.set_ylabel("Blinks/min", color="#8b949e")
        ax2.tick_params(colors="#8b949e")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2d3748")
        ax2.legend(facecolor="#1e2130", labelcolor="#cdd9e5", fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Distance trend ────────────────────────────────────────
    with col2:
        st.subheader("Screen Distance Trend")
        fig3, ax3 = plt.subplots(figsize=(5, 2.8))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#1e2130")
        ax3.plot(x, distance_hist, color="#9b59b6", linewidth=1.5)
        ax3.axhline(50, color=COLOR_MEDIUM, linestyle="--", linewidth=0.8, alpha=0.7, label="Safe (≥50 cm)")
        ax3.set_ylim(20, 120)
        ax3.set_xlabel("Frame", color="#8b949e")
        ax3.set_ylabel("Distance (cm)", color="#8b949e")
        ax3.tick_params(colors="#8b949e")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#2d3748")
        ax3.legend(facecolor="#1e2130", labelcolor="#cdd9e5", fontsize=8)
        st.pyplot(fig3)
        plt.close(fig3)

    # ── Alert log ─────────────────────────────────────────────
    st.subheader("Alert Log")
    if not alert_log:
        st.success("✅  No alerts triggered this session.")
    else:
        for entry in reversed(alert_log[-20:]):    # show last 20
            border = "#e74c3c" if entry["severity"] == "high" else "#f39c12"
            st.markdown(
                f'<div class="alert-box" style="border-color:{border};">'
                f'<span style="color:#8b949e; font-size:0.75rem;">{entry["time"]}</span><br>'
                f'<strong>{entry["title"]}</strong><br>'
                f'<span style="font-size:0.85rem;">{entry["message"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
#  Page 3 — User Profile
# ═══════════════════════════════════════════════════════════════

def page_user_profile() -> None:
    st.header("⚙️  User Profile & Baseline")

    user_id  = st.session_state["user_id"]
    baseline = AdaptiveBaseline(user_id=user_id)
    profile  = baseline.get_profile_summary()
    bl       = baseline.get_baseline()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Profile — {user_id}")
        st.metric("Sessions completed",  profile.get("sessions", 0))
        total_min = profile.get("total_minutes", 0)
        st.metric("Total screen time",   f"{total_min:.0f} min  ({total_min/60:.1f} h)")
        st.metric("Last updated",        profile.get("last_updated") or "Never")

    with col2:
        st.subheader("Personalised Thresholds")
        st.metric("Personal blink baseline",   f"{bl['blink_rate']:.1f} blinks/min")
        st.metric("Personal distance baseline", f"{bl['distance_cm']:.0f} cm")
        st.info(
            "These thresholds are learned from your behaviour over multiple sessions. "
            "The system will not over-alert if your natural blink rate is lower than average.",
            icon="🧠",
        )

    st.divider()
    st.subheader("How Adaptive Learning Works")
    st.markdown(
        """
        1. **Warm-up phase** (first 30 seconds of each session):
           The system observes your natural behaviour without triggering any alerts.

        2. **Baseline seeding**: At the end of warm-up, your average blink rate and
           habitual screen distance are recorded as your personal baseline.

        3. **Continuous EMA update**: Each subsequent frame updates the baseline
           slowly using Exponential Moving Average (α = 0.05), so a bad day doesn't
           permanently distort your profile.

        4. **Personalised scoring**: The Eye Strain Score components are computed
           relative to *your* baseline rather than a fixed clinical average.
        """
    )

    if st.button("🗑️  Reset My Profile", type="secondary"):
        import json, os
        from utils.config import USER_PROFILES_PATH
        if os.path.exists(USER_PROFILES_PATH):
            try:
                with open(USER_PROFILES_PATH, "r") as f:
                    all_p = json.load(f)
                all_p.pop(user_id, None)
                with open(USER_PROFILES_PATH, "w") as f:
                    json.dump(all_p, f, indent=2)
                st.success("Profile reset successfully.")
            except Exception as e:
                st.error(f"Could not reset profile: {e}")
        else:
            st.info("No profile file found — nothing to reset.")


# ═══════════════════════════════════════════════════════════════
#  Main router
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    page = render_sidebar()

    if "Live"    in page: page_live_monitor()
    elif "Analy" in page: page_analytics()
    elif "Profile" in page: page_user_profile()


if __name__ == "__main__":
    main()
