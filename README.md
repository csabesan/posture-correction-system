# Real-Time Posture Correction System

**CPS843 - Introduction to Computer Vision**

A real-time posture correction and ergonomic feedback system using MediaPipe Pose and OpenCV.

---

## Project Overview

This system analyzes a user's posture in real-time using webcam input. It detects body keypoints, computes postural angles, and provides immediate visual feedback to help users maintain proper sitting/standing posture.

### Key Features

- **Real-time pose detection** using MediaPipe Pose
- **Rule-based posture classification** (no ML training required)
- **Visual feedback overlay** with posture status and angle measurements
- **FPS counter** for performance monitoring

---

## System Pipeline

```
┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌─────────┐
│ Webcam  │ →  │ MediaPipe    │ →  │ Keypoint      │ →  │ Angle        │ →  │ Visual  │
│ Input   │    │ Pose         │    │ Extraction    │    │ Computation  │    │ Feedback│
└─────────┘    └──────────────┘    └───────────────┘    └──────────────┘    └─────────┘
                                          │                    │
                                          ▼                    ▼
                                   5 Keypoints:          Threshold-Based
                                   - nose                Classification
                                   - left_shoulder
                                   - right_shoulder
                                   - left_hip
                                   - right_hip
```

---

## How Posture is Classified

### Angles Measured

1. **Neck Angle** (Forward Head Posture)
   - Vector: Shoulder midpoint → Nose
   - Reference: Vertical axis
   - Detects: Head tilting forward

2. **Back Angle** (Slouching)
   - Vector: Hip midpoint → Shoulder midpoint
   - Reference: Vertical axis
   - Detects: Upper body leaning forward

### Classification Rules

| Condition | Result |
|-----------|--------|
| Neck angle > 20° | ❌ Bad Posture |
| Back angle > 15° | ❌ Bad Posture |
| Both angles within thresholds | ✅ Good Posture |

### Geometric Computation

The angle between a body vector and the vertical axis is computed using:

```
angle = arccos( (v · vertical) / |v| )
```

Where:
- `v` is the body segment vector
- `vertical` is the upward unit vector (0, -1) in image coordinates
- Result is converted from radians to degrees

---

## Project Structure

```
posture_project/
├── pose/
│   └── pose_detector.py    # MediaPipe Pose integration
├── logic/
│   └── posture_logic.py    # Angle computation & classification
├── ui/
│   └── visualizer.py       # Visual feedback rendering
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `pose_detector.py` | Extract 5 keypoints from video frame |
| `posture_logic.py` | Compute angles, classify posture |
| `visualizer.py` | Draw feedback overlay on frame |
| `main.py` | Webcam capture, integration, display |

---

## How to Run

### Prerequisites

- Python 3.10 or higher
- Webcam

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd posture_project
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv .venv
   ```

3. **Activate virtual environment**
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Application

```bash
python main.py
```

- Stand or sit in front of your webcam
- The system will display your posture status in real-time
- Press `q` to quit

---

## Visual Output

```
┌────────────────────────────────────────┐
│ GOOD POSTURE          (or BAD POSTURE) │
│ Neck Angle: 12.5 deg                   │
│ Back Angle: 8.3 deg                    │
│ FPS: 30.0                              │
│                                        │
│         ┌───o───┐  ← keypoints         │
│         │   │   │                      │
│         │   │   │  ← skeleton          │
│         └───┴───┘                      │
│                                        │
└────────────────────────────────────────┘
```

---

## Limitations

1. **Single person only** - MediaPipe Pose is configured for single-person detection
2. **Frontal view required** - Best results when facing the camera directly
3. **Lighting sensitive** - Poor lighting may affect pose detection accuracy
4. **Fixed thresholds** - Angle thresholds are not personalized to individual body types
5. **2D analysis only** - Depth information is not used (no 3D posture analysis)
6. **No temporal smoothing** - Posture status may flicker between frames

---

## Future Improvements

1. **Temporal smoothing** - Average posture over multiple frames to reduce flickering
2. **Personalized thresholds** - Calibration phase to adapt to user's body proportions
3. **Audio feedback** - Voice alerts for posture correction
4. **Session logging** - Track posture statistics over time
5. **Multi-angle analysis** - Support for side-view posture assessment
6. **Exercise suggestions** - Recommend stretches based on posture patterns

---

## Technical Notes

### Why Rule-Based Classification?

This project uses simple geometric thresholds instead of machine learning because:
- **Explainability**: Easy to understand and explain in academic reports
- **No training data required**: Works immediately without dataset collection
- **Deterministic**: Same input always produces same output
- **Computationally efficient**: Real-time performance on any hardware

### Coordinate System

- OpenCV/MediaPipe use image coordinates where:
  - Origin (0, 0) is at top-left
  - X increases to the right
  - Y increases downward
- The "vertical" reference is (0, -1) pointing upward

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Webcam capture, image display, drawing |
| mediapipe | ≥0.10.0 | Pose detection (33 landmarks) |
| numpy | ≥1.24.0 | Vector math, angle computation |

---

## Authors

CPS843 Computer Vision Project Team

---

## License

This project is for academic purposes only.
