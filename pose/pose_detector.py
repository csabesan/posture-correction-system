"""
pose_detector.py - MediaPipe Pose Keypoint Extraction Module

This module handles pose detection using MediaPipe Pose and extracts
specific keypoints required for posture analysis.

CPS843 - Computer Vision Project
"""

import mediapipe as mp
import numpy as np


# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose

# Create pose detector instance with static configuration
# - static_image_mode=False: Optimized for video stream (tracks between frames)
# - min_detection_confidence: Minimum confidence for initial detection
# - min_tracking_confidence: Minimum confidence for tracking between frames
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe landmark indices for required keypoints
# Reference: https://google.github.io/mediapipe/solutions/pose.html
LANDMARK_INDICES = {
    'nose': mp_pose.PoseLandmark.NOSE,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP
}


def get_keypoints(frame) -> dict | None:
    """
    Extract pose keypoints from an OpenCV frame.
    
    This function processes a BGR image frame through MediaPipe Pose
    and returns pixel coordinates of specific anatomical landmarks
    required for posture analysis.
    
    Args:
        frame: OpenCV BGR image (numpy array of shape HxWx3)
    
    Returns:
        dict: Dictionary containing keypoint names as keys and 
              (x, y) pixel coordinates as values.
              Keys: 'nose', 'left_shoulder', 'right_shoulder', 
                    'left_hip', 'right_hip'
        None: If pose detection fails or no pose is detected
    
    Example:
        >>> keypoints = get_keypoints(frame)
        >>> if keypoints:
        ...     print(keypoints['nose'])  # (x, y) coordinates
    """
    
    # Validate input frame
    if frame is None or frame.size == 0:
        return None
    
    # Get frame dimensions for coordinate conversion
    # MediaPipe returns normalized coordinates [0, 1]
    # We need to convert to pixel coordinates
    frame_height, frame_width = frame.shape[:2]
    
    # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
    frame_rgb = frame[:, :, ::-1]  # Efficient BGR to RGB conversion
    
    # Process frame through MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Check if pose landmarks were detected
    if results.pose_landmarks is None:
        return None
    
    # Extract required keypoints and convert to pixel coordinates
    keypoints = {}
    
    for name, landmark_index in LANDMARK_INDICES.items():
        # Get the landmark object
        landmark = results.pose_landmarks.landmark[landmark_index]
        
        # Check landmark visibility (optional quality filter)
        # Visibility < 0.5 indicates landmark may be occluded or unreliable
        if landmark.visibility < 0.5:
            # Still include the point, but this could be used for filtering
            pass
        
        # Convert normalized coordinates to pixel coordinates
        # x is normalized by width, y is normalized by height
        pixel_x = int(landmark.x * frame_width)
        pixel_y = int(landmark.y * frame_height)
        
        # Store as tuple (x, y)
        keypoints[name] = (pixel_x, pixel_y)
    
    return keypoints


def release():
    """
    Release MediaPipe resources.
    
    Call this function when done with pose detection to free resources.
    """
    pose.close()
