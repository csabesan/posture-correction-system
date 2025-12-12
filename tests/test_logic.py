import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.posture_logic import classify_posture

fake_keypoints = {
    "left_shoulder": (400, 300),
    "right_shoulder": (500, 300),
    "left_hip": (420, 450),
    "right_hip": (480, 450),
    "nose": (450, 250)
}

result = classify_posture(fake_keypoints)
print(result)