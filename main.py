"""
main.py - Real-Time Posture Correction System

Main entry point for the CPS843 posture correction application.
This module integrates pose detection, posture classification, and
visual feedback into a real-time video processing pipeline.

Pipeline:
    Webcam → MediaPipe Pose → Keypoint Extraction → 
    Angle Computation → Threshold Classification → 
    Visual Feedback Overlay

CPS843 - Computer Vision Project
"""

import cv2
import time

# Import project modules
from pose.pose_detector import get_keypoints, release as release_pose
from logic.posture_logic import classify_posture
from ui.visualizer import draw_feedback


# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_NAME = "Posture Correction System - CPS843"
CAMERA_INDEX = 0  # Default webcam (change if using external camera)

# FPS display settings
FPS_POSITION = (20, 150)
FPS_COLOR = (255, 255, 0)  # Cyan
FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_SCALE = 0.6


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application loop for real-time posture correction.
    
    This function:
    1. Opens the webcam
    2. Processes each frame through the posture analysis pipeline
    3. Displays results with visual feedback
    4. Exits cleanly when 'q' is pressed
    """
    print("=" * 50)
    print("  CPS843 Posture Correction System")
    print("=" * 50)
    print("Starting webcam...")
    print("Press 'q' to quit")
    print()
    
    # Initialize webcam capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Please check camera permissions and connection.")
        return
    
    # Variables for FPS calculation
    prev_time = time.time()
    fps = 0.0
    
    # -----------------------------------------------------------------
    # Main processing loop
    # -----------------------------------------------------------------
    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        
        # Check if frame was captured successfully
        if not ret:
            print("ERROR: Failed to capture frame.")
            break
        
        # Flip frame horizontally for mirror effect (more intuitive for user)
        frame = cv2.flip(frame, 1)
        
        # -------------------------------------------------------------
        # Step 1: Get pose keypoints from frame
        # -------------------------------------------------------------
        keypoints = get_keypoints(frame)
        
        # -------------------------------------------------------------
        # Step 2 & 3: If pose detected, classify and visualize
        # -------------------------------------------------------------
        if keypoints is not None:
            # Step 2: Classify posture using geometric analysis
            posture_data = classify_posture(keypoints)
            
            # Step 3: Draw visual feedback on frame
            draw_feedback(frame, posture_data)
            
            # Optional: Draw keypoint markers for debugging
            _draw_keypoints(frame, keypoints)
        else:
            # No pose detected - show message
            cv2.putText(
                frame,
                "No pose detected - please stand in view",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),  # Orange
                2
            )
        
        # -------------------------------------------------------------
        # Step 4: Calculate and display FPS
        # -------------------------------------------------------------
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            FPS_POSITION,
            FPS_FONT,
            FPS_SCALE,
            FPS_COLOR,
            1
        )
        
        # -------------------------------------------------------------
        # Step 5: Display the frame
        # -------------------------------------------------------------
        cv2.imshow(WINDOW_NAME, frame)
        
        # -------------------------------------------------------------
        # Step 6: Check for quit key
        # -------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nExiting...")
            break
    
    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    release_pose()  # Release MediaPipe resources
    print("Done.")


def _draw_keypoints(frame, keypoints: dict):
    """
    Draw detected keypoints on the frame for visual debugging.
    
    Args:
        frame: OpenCV BGR image
        keypoints: Dictionary of keypoint coordinates
    """
    # Define colors for different body parts
    colors = {
        "nose": (255, 0, 255),        # Magenta
        "left_shoulder": (255, 0, 0),  # Blue
        "right_shoulder": (255, 0, 0), # Blue
        "left_hip": (0, 255, 255),     # Yellow
        "right_hip": (0, 255, 255)     # Yellow
    }
    
    # Draw each keypoint as a circle
    for name, (x, y) in keypoints.items():
        color = colors.get(name, (0, 255, 0))
        
        # Draw filled circle at keypoint location
        cv2.circle(frame, (x, y), 6, color, -1)
        
        # Draw keypoint label
        cv2.putText(
            frame,
            name,
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
    
    # Draw skeleton lines connecting keypoints
    # Shoulder line
    cv2.line(
        frame,
        keypoints["left_shoulder"],
        keypoints["right_shoulder"],
        (255, 255, 255),
        2
    )
    
    # Hip line
    cv2.line(
        frame,
        keypoints["left_hip"],
        keypoints["right_hip"],
        (255, 255, 255),
        2
    )
    
    # Left side (shoulder to hip)
    cv2.line(
        frame,
        keypoints["left_shoulder"],
        keypoints["left_hip"],
        (200, 200, 200),
        2
    )
    
    # Right side (shoulder to hip)
    cv2.line(
        frame,
        keypoints["right_shoulder"],
        keypoints["right_hip"],
        (200, 200, 200),
        2
    )
    
    # Neck line (shoulder midpoint to nose)
    shoulder_mid_x = (keypoints["left_shoulder"][0] + keypoints["right_shoulder"][0]) // 2
    shoulder_mid_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) // 2
    cv2.line(
        frame,
        (shoulder_mid_x, shoulder_mid_y),
        keypoints["nose"],
        (255, 0, 255),
        2
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
