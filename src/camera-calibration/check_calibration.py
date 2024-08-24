import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Load the camera calibration parameters
with np.load('camera_calib.npz') as data:
    mtx, dist = data['mtx'], data['dist']

# Open the webcam and set the resolution to 1280x720 for better accuracy
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


def undistort_image(image, mtx, dist):
    """
    Removes distortion from an image based on calibration parameters.

    Args:
    - image: Input image (distorted).
    - mtx: Camera matrix.
    - dist: Distortion coefficients.

    Returns:
    - undistorted_image: Undistorted image.
    """
    h, w = image.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, mtx, dist, None, new_mtx)
    return undistorted_image


def show_distortion_effect(frame, mtx, dist):
    """
    Displays the original and undistorted images side by side.

    Args:
    - frame: Original image frame.
    - mtx: Camera matrix.
    - dist: Distortion coefficients.
    """
    undistorted_frame = undistort_image(frame, mtx, dist)
    combined_image = np.hstack((frame, undistorted_frame))
    cv2.imshow('Distortion Effect (Left: Distorted, Right: Undistorted)', combined_image)


def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Show the distortion effect
        show_distortion_effect(frame, mtx, dist)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
