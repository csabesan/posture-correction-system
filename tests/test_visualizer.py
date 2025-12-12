import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from ui.visualizer import draw_feedback

frame = np.zeros((480, 640, 3), dtype=np.uint8)

posture_data = {
    "neck_angle": 25.0,
    "back_angle": 18.0,
    "is_good": False
}

draw_feedback(frame, posture_data)
cv2.imshow("Visualizer Test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
