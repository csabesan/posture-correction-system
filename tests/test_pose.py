import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from pose.pose_detector import get_keypoints

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = get_keypoints(frame)
    if keypoints:
        for name, (x, y) in keypoints.items():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, name, (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Pose Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
