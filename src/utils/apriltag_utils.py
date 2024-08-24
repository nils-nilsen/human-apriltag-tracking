import cv2
import numpy as np
from apriltag import apriltag
from collections import deque

class AprilTagDetector:
    """
    A class to handle AprilTag detection.

    Attributes:
        detector (apriltag.Detector): An AprilTag detector instance.
    """

    def __init__(self, tag_family):
        """
        Initializes the AprilTagDetector with a specific tag family.

        Args:
            tag_family (str): The family of AprilTags to detect, e.g., "tag36h11".
        """
        self.detector = apriltag(tag_family)

    def detect(self, image):
        """
        Detects AprilTags in a given image.

        Args:
            image (np.array): The image in which to detect AprilTags.

        Returns:
            list: A list of detected tags, each represented as a dictionary with tag properties.
        """
        return self.detector.detect(image)

def get_pose(tag, mtx, dist, size):
    """
    Calculates the pose (rotation and translation vectors) of an AprilTag.

    Args:
        tag (dict): The detected AprilTag's information.
        mtx (np.array): The camera matrix.
        dist (np.array): The distortion coefficients.
        size (float): The size of the AprilTag in meters.

    Returns:
        tuple: A tuple containing the rotation vector (rvec) and translation vector (tvec).
    """
    corners = tag['lb-rb-rt-lt'].astype(np.float32)
    obj_points = np.array([
        [-size / 2, -size / 2, 0],
        [size / 2, -size / 2, 0],
        [size / 2, size / 2, 0],
        [-size / 2, size / 2, 0]
    ], dtype=np.float32)
    ret, rvec, tvec = cv2.solvePnP(obj_points, corners, mtx, dist)
    return rvec, tvec

def transform_coordinates(coords):
    """
    Transforms the coordinates from the camera's coordinate system to the room's coordinate system.

    Args:
        coords (np.array): The coordinates (x, y, z) to be transformed.

    Returns:
        np.array: The transformed coordinates in the room's coordinate system.
    """
    camera_x = coords[0]
    camera_y = 1.5  # Fixed Y position for the room's coordinate system.
    camera_z = -5.5  # Position of the camera along the Z-axis in the room.
    transformed_coords = np.array([camera_x, camera_y, (camera_z + coords[2]) * -1])
    return transformed_coords

def adjust_rotation_for_shoulder(rvec, is_left_shoulder):
    """
    Adjusts the rotation of a shoulder tag based on its position on the body.

    Args:
        rvec (np.array): The rotation vector of the tag.
        is_left_shoulder (bool): Whether the tag is on the left shoulder.

    Returns:
        np.array: The adjusted rotation vector.
    """
    rmat, _ = cv2.Rodrigues(rvec)
    if is_left_shoulder:
        rotation_adjustment = cv2.Rodrigues(np.array([0, 0, np.pi / 2]))[0]  # 90 degrees for the left shoulder.
    else:
        rotation_adjustment = cv2.Rodrigues(np.array([0, 0, -np.pi / 2]))[0]  # -90 degrees for the right shoulder.
    rmat_adjusted = rotation_adjustment @ rmat
    rvec_adjusted, _ = cv2.Rodrigues(rmat_adjusted)
    return rvec_adjusted

def determine_body_orientation(detections, mtx, dist):
    """
    Determines the orientation of the body based on the detected AprilTags.

    Args:
        detections (list): A list of detected AprilTags.
        mtx (np.array): The camera matrix.
        dist (np.array): The distortion coefficients.

    Returns:
        str: A string describing the orientation of the body ("Front", "Back", "Left Side", "Right Side", or "Unknown").
    """
    min_angle = float('inf')
    orientation = "Unknown"

    for detection in detections:
        if detection['id'] == 2 or detection['id'] == 1:
            rvec, tvec = get_pose(detection, mtx, dist, 0.2)  # Larger tags for front and back.
        elif detection['id'] == 6 or detection['id'] == 0:
            rvec, tvec = get_pose(detection, mtx, dist, 0.15)  # Smaller tags for shoulders.
        else:
            continue

        rmat, _ = cv2.Rodrigues(rvec)
        z_axis = rmat[:, 2]  # Z-axis of the rotation matrix.
        angle = np.arccos(z_axis[2])  # Angle between the tag's Z-axis and the camera's Z-axis.

        if angle < min_angle:
            min_angle = angle
            if detection['id'] == 2:
                orientation = "Front"
            elif detection['id'] == 1:
                orientation = "Back"
            elif detection['id'] == 6:
                orientation = "Left Side"
            elif detection['id'] == 0:
                orientation = "Right Side"

    return orientation

def is_valid_measurement(new_coords, last_coords, threshold=20.0):
    """
    Checks if the new measurement is valid based on the movement threshold.

    Args:
        new_coords (np.array): The new coordinates (x, y, z).
        last_coords (np.array): The last recorded coordinates (x, y, z).
        threshold (float): The maximum allowed change in position to consider the measurement valid.

    Returns:
        bool: True if the measurement is valid, False otherwise.
    """
    if last_coords is None:
        return True
    delta = np.linalg.norm(new_coords - last_coords)
    return delta < threshold

def round_coords(coords, decimals=2):
    """
    Rounds the coordinates to a specified number of decimal places.

    Args:
        coords (np.array): The coordinates (x, y, z) to be rounded.
        decimals (int): The number of decimal places to round to.

    Returns:
        np.array: The rounded coordinates.
    """
    return np.round(coords, decimals)

def moving_average(new_coords, window=deque(maxlen=5)):
    """
    Computes the moving average of the coordinates to smooth the data.

    Args:
        new_coords (np.array): The new coordinates (x, y, z).
        window (deque): A deque used to store recent coordinates and calculate the moving average.

    Returns:
        np.array: The smoothed coordinates using a moving average.
    """
    window.append(new_coords)
    return np.mean(window, axis=0)

def undistort_image(image, mtx, dist):
    """
    Removes distortion from an image based on the camera's calibration parameters.

    Args:
        image (np.array): The distorted input image.
        mtx (np.array): The camera matrix.
        dist (np.array): The distortion coefficients.

    Returns:
        np.array: The undistorted image.
    """
    h, w = image.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, mtx, dist, None, new_mtx)
    return undistorted_image
