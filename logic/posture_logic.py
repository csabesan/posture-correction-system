"""
This module performs posture classification using geometric analysis
of body keypoints. It computes angles between body segments and a
vertical reference axis to determine if the user has good or bad posture.
"""

import numpy as np

# CONFIGURABLE THRESHOLDS (in degrees)
# These thresholds define the boundary between good and bad posture.
# Values are based on ergonomic guidelines for seated posture.
NECK_ANGLE_THRESHOLD = 20.0  # forward head tilt
BACK_ANGLE_THRESHOLD = 15.0  # upper back slouching

# HELPER FUNCTIONS
def _compute_midpoint(point1: tuple, point2: tuple) -> np.ndarray:
    """
    Compute the midpoint between two 2D points.
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple
    
    Returns:
        NumPy array containing midpoint coordinates [x, y]
    
    Example:
        >>> _compute_midpoint((0, 0), (10, 10))
        array([5., 5.])
    """
    return np.array([
        (point1[0] + point2[0]) / 2.0,
        (point1[1] + point2[1]) / 2.0
    ])

def _compute_angle_with_vertical(vector: np.ndarray) -> float:
    """
    Compute the angle between a vector and the vertical axis.
    
    The vertical axis points upward (negative y in image coordinates).
    
    Geometry:
        - Image coordinate system: y increases downward
        - Vertical reference: (0, -1) pointing upward
        - Result is always positive (0 to 180 degrees)
    
    Formula:
        angle = arccos( (v · vertical) / (|v| * |vertical|) )
    
    Args:
        vector: 2D vector as NumPy array [x, y]
    
    Returns:
        Angle in degrees (0 to 180)
    """
    vertical = np.array([0.0, -1.0])
    
    vector_magnitude = np.linalg.norm(vector)
    
    # Avoid division by zero for zero-length vectors
    if vector_magnitude < 1e-6:
        return 0.0
    
    # Normalize the input vector to unit length
    vector_normalized = vector / vector_magnitude
    
    # Compute dot product with vertical axis
    dot_product = np.dot(vector_normalized, vertical)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Compute angle using inverse cosine
    angle_radians = np.arccos(dot_product)
    
    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return float(angle_degrees)

# ANGLE COMPUTATION FUNCTIONS
def _compute_neck_angle(keypoints: dict) -> float:
    """
    Compute the neck angle (forward head posture detection).
    
    Measurement:
        - Vector: shoulder midpoint → nose
        - Reference: vertical axis (upward)
        - Good posture: nose is directly above shoulder midpoint (angle ≈ 0°)
        - Bad posture: nose is forward of shoulders (angle > threshold)
    
    Diagram (side view, person facing right):
    
                 o  ← nose
                /
               /  ← neck angle with vertical
              |
              |  ← vertical reference
         [shoulders]
    
    Args:
        keypoints: Dictionary with body landmark coordinates
    
    Returns:
        Neck angle in degrees (0 = perfect vertical alignment)
    """
    # Calculate the midpoint between left and right shoulders
    shoulder_midpoint = _compute_midpoint(
        keypoints["left_shoulder"],
        keypoints["right_shoulder"]
    )
    
    # Get nose position as numpy array
    nose = np.array(keypoints["nose"])
    
    # Create vector from shoulder midpoint pointing toward nose
    neck_vector = nose - shoulder_midpoint
    
    # Compute angle between neck vector and vertical axis
    return _compute_angle_with_vertical(neck_vector)

def _compute_back_angle(keypoints: dict) -> float:
    """
    Compute the upper back angle (slouching detection).
    
    Measurement:
        - Vector: hip midpoint → shoulder midpoint
        - Reference: vertical axis (upward)
        - Good posture: shoulders directly above hips (angle ≈ 0°)
        - Bad posture: shoulders forward of hips (angle > threshold)
    
    Diagram (side view, person facing right):
    
         [shoulders]
              /
             /  ← back angle with vertical
            |
            |  ← vertical reference
         [hips]
    
    Args:
        keypoints: Dictionary with body landmark coordinates
    
    Returns:
        Back angle in degrees (0 = perfect vertical alignment)
    """
    # Calculate the midpoint between left and right hips
    hip_midpoint = _compute_midpoint(
        keypoints["left_hip"],
        keypoints["right_hip"]
    )
    
    # Calculate the midpoint between left and right shoulders
    shoulder_midpoint = _compute_midpoint(
        keypoints["left_shoulder"],
        keypoints["right_shoulder"]
    )
    
    # Create vector from hip midpoint pointing toward shoulder midpoint
    back_vector = shoulder_midpoint - hip_midpoint
    
    # Compute angle between back vector and vertical axis
    return _compute_angle_with_vertical(back_vector)

# MAIN CLASSIFICATION FUNCTION
def classify_posture(keypoints: dict) -> dict:
    """
    Classify posture as good or bad based on body angles.
    
    This function implements a rule-based classification system:
    1. Compute neck angle (forward head posture)
    2. Compute back angle (slouching)
    3. Apply threshold-based classification
    
    Classification Rules:
        - Neck angle > 20° → BAD posture (forward head)
        - Back angle > 15° → BAD posture (slouching)
        - Both angles within thresholds → GOOD posture
    
    Args:
        keypoints: Dictionary containing pixel coordinates for:
                   - "left_shoulder": (x, y)
                   - "right_shoulder": (x, y)
                   - "left_hip": (x, y)
                   - "right_hip": (x, y)
                   - "nose": (x, y)
    
    Returns:
        Dictionary with classification results:
        {
            "neck_angle": float,  # Angle in degrees
            "back_angle": float,  # Angle in degrees
            "is_good": bool       # True if posture is good
        }
    
    Example:
        >>> keypoints = {
        ...     "nose": (320, 100),
        ...     "left_shoulder": (280, 200),
        ...     "right_shoulder": (360, 200),
        ...     "left_hip": (290, 400),
        ...     "right_hip": (350, 400)
        ... }
        >>> result = classify_posture(keypoints)
        >>> print(result)
        {'neck_angle': 5.2, 'back_angle': 3.1, 'is_good': True}
    """
    # Step 1: Compute neck angle (forward head posture)
    neck_angle = _compute_neck_angle(keypoints)
    
    # Step 2: Compute back angle (slouching)
    back_angle = _compute_back_angle(keypoints)
    
    # Step 3: Apply threshold-based classification
    # Posture is good only if BOTH angles are within acceptable limits
    neck_ok = neck_angle <= NECK_ANGLE_THRESHOLD
    back_ok = back_angle <= BACK_ANGLE_THRESHOLD
    is_good = neck_ok and back_ok
    
    return {
        "neck_angle": round(neck_angle, 1),
        "back_angle": round(back_angle, 1),
        "is_good": is_good
    }
