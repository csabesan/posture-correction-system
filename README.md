# Real-Time Posture Correction System

**CPS843 – Introduction to Computer Vision**

A real-time posture correction and ergonomic feedback system built using MediaPipe Pose and OpenCV.

---

## Project Overview

This project implements a real-time posture monitoring system using webcam input. The system detects upper-body keypoints, computes posture-related angles, and displays visual feedback to indicate whether the user’s posture is good or needs correction. The goal is to demonstrate how pose estimation and basic geometric analysis can be combined for a practical computer vision application.

### Key Features

* Real-time pose detection using MediaPipe Pose
* Rule-based posture classification (no machine learning training required)
* On-screen visual feedback with posture status and angle measurements
* FPS counter for basic performance monitoring

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

## How Posture Is Classified

### Angles Measured

1. **Neck Angle (Forward Head Posture)**

   * Vector: Shoulder midpoint → Nose
   * Reference: Vertical axis
   * Indicates forward head positioning

2. **Back Angle (Slouching)**

   * Vector: Hip midpoint → Shoulder midpoint
   * Reference: Vertical axis
   * Indicates upper-body leaning

### Classification Rules

| Condition                     | Result       |
| ----------------------------- | ------------ |
| Neck angle > 20°              | Bad posture  |
| Back angle > 15°              | Bad posture  |
| Both angles within thresholds | Good posture |

### Geometric Computation

The angle between a body vector and the vertical axis is computed as:

```
angle = arccos( (v · vertical) / |v| )
```

Where:

* `v` is the body segment vector
* `vertical` is the upward unit vector (0, -1) in image coordinates
* The result is converted from radians to degrees

---

## Project Structure

```
posture_project/
├── pose/
│   └── pose_detector.py    # MediaPipe Pose integration
├── logic/
│   └── posture_logic.py    # Angle computation and posture classification
├── ui/
│   └── visualizer.py       # Visual feedback rendering
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Module Responsibilities

| Module             | Description                                     |
| ------------------ | ----------------------------------------------- |
| `pose_detector.py` | Extracts keypoints from each video frame        |
| `posture_logic.py` | Computes angles and classifies posture          |
| `visualizer.py`    | Draws skeleton, angles, and posture feedback    |
| `main.py`          | Handles webcam input and integrates all modules |

---

## How to Run

### Prerequisites

* **Python 3.10 or 3.11**
* A working webcam

> Note: MediaPipe currently requires Python 3.10+ for compatibility.

---

### Setup Instructions

1. **Download or clone the project**

   ```bash
   git clone <repository-url>
   cd posture_project
   ```

   If using a Google Drive submission, download the folder and navigate into it:

   ```bash
   cd posture_project
   ```

2. **Create a virtual environment**

   ```bash
   python3.11 -m venv .venv
   ```

3. **Activate the virtual environment**

   macOS / Linux:

   ```bash
   source .venv/bin/activate
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Application

```bash
python main.py
```

* Sit or stand facing the webcam
* The application window will display posture status in real time
* Angle values and FPS are shown on screen
* Press **`q`** to exit the application

---

## Visual Output (Example)

```
┌────────────────────────────────────────┐
│ GOOD POSTURE        (or BAD POSTURE)   │
│ Neck Angle: 12.5°                      │
│ Back Angle: 8.3°                       │
│ FPS: 30.0                              │
│                                        │
│         ┌───o───┐                      │
│         │   │   │   Pose landmarks     │
│         │   │   │                      │
│         └───┴───┘                      │
│                                        │
└────────────────────────────────────────┘
```

---

## Limitations

* Single-person posture analysis only
* Frontal camera view required for reliable results
* Performance may degrade under poor lighting conditions
* Fixed posture thresholds are not personalized
* 2D pose estimation only (no depth information)
* No temporal smoothing, which may cause brief label flickering

---

## Future Improvements

* Temporal smoothing across frames
* User-specific posture calibration
* Audio or notification-based feedback
* Posture session logging and statistics
* Support for side-view posture analysis
* Stretch or exercise recommendations

---

## Technical Notes

### Why a Rule-Based Approach?

A rule-based geometric method was chosen because it:

* Produces interpretable and explainable results
* Does not require training data
* Runs efficiently in real time
* Aligns well with classical computer vision concepts taught in CPS843

### Coordinate System

* Image origin at the top-left corner
* X-axis increases to the right
* Y-axis increases downward
* Vertical reference vector defined as (0, -1)

---

## Dependencies

| Package       | Purpose                          |
| ------------- | -------------------------------- |
| opencv-python | Webcam capture and visualization |
| mediapipe     | Pose estimation                  |
| numpy         | Vector and angle computations    |

---

## Authors

CPS843 Computer Vision Project Team

---

## License

This project was developed for academic purposes only.

---
