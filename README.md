# 👁️ EyeGuard AI — Real-Time Eye Strain Detection & Adaptive Prevention System

> **CSE322 — Industry Ethics and Legal Issues | Patent Project**
> An AI-powered, webcam-based system that monitors physiological eye signals in real time, computes a dynamic Eye Strain Score, and triggers personalised adaptive alerts — no specialised hardware required.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Installation](#installation)
6. [Running the Application](#running-the-application)
7. [Dashboard Guide](#dashboard-guide)
8. [Configuration](#configuration)
9. [Patent Innovations](#patent-innovations)
10. [Future Improvements](#future-improvements)
11. [Team](#team)

---

## Overview

**Digital Eye Strain (Computer Vision Syndrome)** affects ~75 % of computer users worldwide, costing an estimated $2 billion annually in lost productivity. Existing solutions (20-20-20 reminder apps, blue-light filters, screen-time trackers) are:

- **Static** — fire alerts on fixed intervals regardless of actual strain
- **Non-physiological** — ignore real biological signals
- **Generic** — apply the same thresholds to every user

**EyeGuard AI** solves all three gaps with a five-stage computer vision pipeline that:

| Feature | Traditional Apps | EyeGuard AI |
|---------|-----------------|-------------|
| Real-time detection | ❌ | ✅ Continuous, every frame |
| Physiological monitoring | ❌ | ✅ EAR, blink rate, IPD, gaze |
| Personalised thresholds | ❌ | ✅ Per-user ML baseline |
| Adaptive alerts | ❌ Fixed timers | ✅ Triggered by actual strain |
| Multi-factor scoring | ❌ Single metric | ✅ Four-factor weighted ESS |
| User behaviour learning | ❌ | ✅ Improves each session |

---

## System Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  Webcam Input│───▶│ Stage 1: Capture  │───▶│ Stage 2: Detect  │
│  (30 fps)    │    │  OpenCV VideoCapt │    │  MediaPipe Mesh  │
└──────────────┘    └───────────────────┘    └──────────────────┘
                                                       │
                                             ┌─────────▼────────┐
                                             │ Stage 3: Extract │
                                             │  EAR, blinks,    │
                                             │  distance, time  │
                                             └─────────┬────────┘
                                                       │
                             ┌─────────────────────────▼──────────┐
                             │         Stage 4: Score              │
                             │  ESS = w1·Blink + w2·Dist +         │
                             │        w3·Fatigue + w4·Gaze         │
                             └─────────────────────────┬──────────┘
                                                       │
                   ┌───────────────────────────────────▼──────────┐
                   │             Stage 5: Decide                   │
                   │  Low / Medium / High  →  Adaptive Alert       │
                   └───────────────────────────────────┬──────────┘
                                                       │
                   ┌───────────────────────────────────▼──────────┐
                   │       Stage 6: Adaptive Learning              │
                   │  EMA baseline update per user per session     │
                   └──────────────────────────────────────────────┘
```

---

## Project Structure

```
eye_strain_ai/
│
├── app.py                  # Streamlit dashboard (main UI entry point)
├── main.py                 # Headless CLI pipeline (no UI, terminal output)
│
├── modules/
│   ├── detector.py         # MediaPipe FaceMesh wrapper (Stage 2)
│   ├── features.py         # EAR, blink detection, distance, session (Stage 3)
│   ├── scoring.py          # Eye Strain Score engine (Stage 4)
│   ├── decision.py         # Alert / decision system (Stage 5)
│   └── adaptive.py         # Per-user baseline learning (Stage 6)
│
├── utils/
│   ├── config.py           # Runtime configuration (camera, paths, FPS)
│   └── constants.py        # All tunable numeric constants
│
├── data/
│   └── user_profiles.json  # Persistent per-user baselines
│
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Eye Aspect Ratio (EAR)

EAR measures how open the eye is each frame using 6 MediaPipe landmarks:

```
       p2  p3
p1  ──────────  p4
       p6  p5

EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

- Open eye  → EAR ≈ 0.25–0.35
- Closed eye → EAR < 0.21 (blink threshold)

### 2. Blink Rate

A blink is counted when EAR stays below the threshold for ≥ 2 consecutive frames.
Blinks per minute are computed over a rolling 60-second window.

Healthy range: **12–20 blinks / minute**

### 3. Screen Distance

Inter-pupillary distance (IPD) in pixels is measured using MediaPipe iris landmarks (468/473).

```
distance_cm = (reference_ipd_pixels / measured_ipd_pixels) × reference_distance_cm
```

Safe minimum distance: **50 cm**

### 4. Eye Strain Score (ESS)

```
ESS = 0.30 × Blink_Deficit
    + 0.25 × Distance_Risk
    + 0.25 × Session_Fatigue
    + 0.20 × Gaze_Strain
```

Each component is normalised to [0, 100]:

| Component | Formula |
|-----------|---------|
| Blink Deficit | Linear ramp as blink rate drops below personal floor |
| Distance Risk | Linear ramp as user moves closer than personal safe distance |
| Session Fatigue | Logarithmic growth: `100 × log(1+t/30) / log(1+120/30)` |
| Gaze Strain | EAR variance over last 30 frames (proxy for squinting/darting) |

| ESS Band | Strain Level |
|----------|-------------|
| 0–33     | 🟢 Low      |
| 34–66    | 🟡 Medium   |
| 67–100   | 🔴 High     |

### 5. Adaptive Learning

On first launch a 30-second warm-up period is observed silently.
At the end of warm-up, the system seeds your personal baseline:
- `personal_blink_rate` = your average blinks/min during warm-up
- `personal_distance`   = your average seating distance during warm-up

Each subsequent frame updates the baseline using **Exponential Moving Average**:

```
baseline_new = 0.95 × baseline_old + 0.05 × observed_value
```

Baselines are saved to `data/user_profiles.json` and reloaded each session.

---

## Installation

### Prerequisites

- Python **3.9 – 3.11** (MediaPipe has limited Python 3.12 support)
- A working webcam

### Steps

```bash
# 1. Clone / download the project
cd eye_strain_ai

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Troubleshooting install:**
> - On macOS with Apple Silicon: `pip install mediapipe` may require `pip install mediapipe-silicon`
> - On Ubuntu/Debian: `sudo apt install libgl1` may be needed for OpenCV

---

## Running the Application

### Option A — Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

This opens the full interactive dashboard in your browser at `http://localhost:8501`.

### Option B — Headless CLI Mode

```bash
python main.py
```

Opens the webcam in an OpenCV window with a minimal HUD overlay.
Press **Q** to quit.

---

## Dashboard Guide

### 📷 Live Monitor

1. Enter your **User ID** in the sidebar (first-time users: `default_user`)
2. Click **▶ Start Monitoring**
3. The webcam feed appears with a live overlay
4. Real-time metrics update every ~100 ms:
   - **ESS gauge** — colour-coded 0–100
   - **Strain badge** — Low / Medium / High
   - **Blink rate** — blinks per minute
   - **Screen distance** — estimated cm
   - **Session time** — continuous usage in minutes
   - **EAR** — raw eye openness ratio
   - **Component bars** — breakdown of the four ESS factors
5. Alert banners appear at the bottom of the video when thresholds are crossed

### 📊 Analytics

- **ESS over time** — full session trend with colour-coded bands
- **Blink rate trend** — with healthy-range reference line
- **Distance trend** — with safe-distance reference line
- **Alert log** — all alerts triggered this session (reverse chronological)

### ⚙️ User Profile

- Displays your personalised baselines and session statistics
- Explains how adaptive learning works
- **Reset Profile** button to start fresh

---

## Configuration

All tunable parameters live in `utils/constants.py` and `utils/config.py`.

Key constants you may want to adjust:

| Constant | Default | Description |
|----------|---------|-------------|
| `EAR_THRESHOLD` | `0.21` | EAR below which eye is "closed" |
| `HEALTHY_BLINK_RATE_MIN` | `12` | Clinical lower bound for blink rate |
| `SAFE_DISTANCE_CM` | `50` | Minimum recommended screen distance |
| `ALERT_COOLDOWN_SEC` | `60` | Seconds between repeated alerts |
| `W_BLINK` | `0.30` | Weight of blink deficit in ESS |
| `W_DISTANCE` | `0.25` | Weight of distance risk in ESS |
| `W_FATIGUE` | `0.25` | Weight of session fatigue in ESS |
| `W_GAZE` | `0.20` | Weight of gaze strain in ESS |
| `BASELINE_WARMUP_SECONDS` | `30` | Warm-up period before baseline seeded |
| `CAMERA_INDEX` | `0` | OpenCV camera device index |

---

## Patent Innovations

This project introduces four patent-worthy contributions absent in all existing solutions:

1. **Real-Time Physiological Monitoring** — reads live biological signals (EAR, blink rate, IPD) every frame; sub-second strain detection without any wearable hardware.

2. **Adaptive, Personalised Thresholds** — the system builds a unique biological baseline per user via EMA, eliminating over-triggering for people who naturally blink less.

3. **Multi-Factor Composite Scoring** — ESS integrates four independent signals (blink deficit, distance risk, session fatigue, gaze variance) into a single clinically-informed score, unlike any existing product.

4. **Strain-Triggered Alerts with Cooldown** — alerts fire only when physiological data confirms strain, and are suppressed during cooldown, removing the "notification blindness" that makes static reminder apps ineffective.

---

## Future Improvements

| Feature | Description |
|---------|-------------|
| Gaze tracking | Full 3-D gaze vector estimation for precise gaze-deviation scoring |
| Pupil dilation | Camera-based pupil-diameter monitoring as a stress indicator |
| ML classifier | Replace rule-based scoring with a trained Random Forest / SVM |
| Mobile app | React Native or Flutter port for smartphone-based monitoring |
| Cloud sync | Multi-device baseline synchronisation via REST API |
| Calibration wizard | On-screen guided calibration to improve IPD → distance accuracy |
| Dark / dry-eye mode | Ambient-light-aware recommendations |
| Export report | PDF health report per session |

---

## Team

| Name | Registration | Roll No |
|------|-------------|---------|
| Narain Karthigeyan E | 12302489 | 65 |
| Yugandheran G | 12310206 | 51 |
| K Karthik Kumar | 12324726 | 67 |

**Course:** CSE322 — Industry Ethics and Legal Issues
