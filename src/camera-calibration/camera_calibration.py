import cv2
import numpy as np
import glob
import os

# Parameters
chessboard_size = (10, 7)  # Number of inner corners in the chessboard pattern (width, height)
square_size = 0.025  # Size of a square in the chessboard (in meters, e.g., 25 mm)

# Prepare the object points, like (0,0,0), (1,0,0), (2,0,0), ..., (9,6,0)
# These represent the 3D coordinates in the chessboard's local coordinate system.
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale the points according to the actual size of the squares.

# Arrays to store object points (3D points in the real world) and image points (2D points in the image plane)
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images from the specified directory
images = glob.glob('calibration_images/*.jpg')
print(f"Found images: {len(images)}")  # Output the number of images found

if len(images) == 0:
    print("No calibration images found. Please check the path.")
    exit()

image_counter = 0  # Counter to track the number of processed images
valid_detection_counter = 0  # Counter to track valid chessboard detections

for fname in images:
    img = cv2.imread(fname)  # Read the image file
    if img is None:
        print(f"Failed to load image: {fname}")
        continue  # Skip to the next image if loading failed

    image_counter += 1  # Increment the counter
    print(f"Processing image {image_counter}/{len(images)}: {fname}")  # Output the counter and filename

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)  # Find the chessboard corners

    if ret:
        valid_detection_counter += 1
        # If corners are found, add object points and image points
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)  # Display the image with the detected corners for 500ms
    else:
        print(f"Chessboard not detected in image {fname}.")

cv2.destroyAllWindows()  # Close all OpenCV windows

# Check if sufficient points were detected before calibrating
if valid_detection_counter > 0:
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("Rotation vectors:\n", rvecs)
    print("Translation vectors:\n", tvecs)
    np.savez('camera_calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("No valid chessboard corners found. Calibration not possible. Try to take new calibration images.")
