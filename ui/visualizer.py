"""
This module handles all visual feedback rendering for the posture
correction system. It displays posture classification results and
angle measurements on the video frame.
"""

import cv2

# VISUAL CONFIGURATION
# Colors in BGR format (OpenCV standard)
COLOR_GREEN = (0, 255, 0)      # Good posture indicator
COLOR_RED = (0, 0, 255)        # Bad posture indicator
COLOR_WHITE = (255, 255, 255)  # General text

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE = 1.0         # For status text
FONT_SCALE_SMALL = 0.6         # For angle values
FONT_THICKNESS_LARGE = 2
FONT_THICKNESS_SMALL = 1

# Text positioning (top-left corner)
STATUS_POSITION = (20, 40)     # "GOOD/BAD POSTURE" text
NECK_ANGLE_POSITION = (20, 80) # Neck angle display
BACK_ANGLE_POSITION = (20, 110) # Back angle display

def draw_feedback(frame, posture_data: dict):
    """
    Draw posture feedback overlay on the video frame.
    
    This function renders:
    1. Posture status text ("GOOD POSTURE" or "BAD POSTURE")
    2. Numeric angle values for neck and back
    
    Args:
        frame: OpenCV BGR image (numpy array) - modified in place
        posture_data: Dictionary containing classification results:
                      {
                          "neck_angle": float,  # degrees
                          "back_angle": float,  # degrees
                          "is_good": bool
                      }
    
    Returns:
        None (frame is modified in place)
    
    Example:
        >>> posture_data = {"neck_angle": 15.2, "back_angle": 10.5, "is_good": True}
        >>> draw_feedback(frame, posture_data)
    """
    # Extract data from posture_data dictionary
    neck_angle = posture_data["neck_angle"]
    back_angle = posture_data["back_angle"]
    is_good = posture_data["is_good"]
    
    # 1. Draw posture status text
    if is_good:
        status_text = "GOOD POSTURE"
        status_color = COLOR_GREEN
    else:
        status_text = "BAD POSTURE"
        status_color = COLOR_RED
    
    # Draw status text with shadow for better visibility
    # Shadow (black outline effect)
    cv2.putText(
        frame,
        status_text,
        (STATUS_POSITION[0] + 2, STATUS_POSITION[1] + 2),
        FONT,
        FONT_SCALE_LARGE,
        (0, 0, 0), 
        FONT_THICKNESS_LARGE + 1
    )
    # Main text
    cv2.putText(
        frame,
        status_text,
        STATUS_POSITION,
        FONT,
        FONT_SCALE_LARGE,
        status_color,
        FONT_THICKNESS_LARGE
    )
    
    # 2. Draw neck angle value
    neck_text = f"Neck Angle: {neck_angle:.1f} deg"
    
    cv2.putText(
        frame,
        neck_text,
        NECK_ANGLE_POSITION,
        FONT,
        FONT_SCALE_SMALL,
        COLOR_WHITE,
        FONT_THICKNESS_SMALL
    )
    
    # 3. Draw back angle value
    back_text = f"Back Angle: {back_angle:.1f} deg"
    
    cv2.putText(
        frame,
        back_text,
        BACK_ANGLE_POSITION,
        FONT,
        FONT_SCALE_SMALL,
        COLOR_WHITE,
        FONT_THICKNESS_SMALL
    )
