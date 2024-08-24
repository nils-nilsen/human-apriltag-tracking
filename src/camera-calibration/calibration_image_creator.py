import cv2
import os
import time
import numpy as np

# Parameters
chessboard_size = (10, 7)  # Number of inner corners in the chessboard pattern (width, height)
square_size = 0.025  # Size of a square in the chessboard (in meters)
output_dir = 'calibration_images'  # Directory to save the captured images
capture_interval = 2  # Time in seconds between captures
movement_threshold = 10  # Movement threshold in pixels

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set higher resolution for better accuracy
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the webcam was successfully opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

img_count = 0  # Counter for the saved images
last_capture_time = time.time()  # Time of the last captured image
last_corners = None  # Store the last detected corners

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale as it is required for chessboard detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Flags for better detection
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    # Try to find the chessboard corners in the frame
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)

    if ret:
        print("Chessboard detected.")
        # Refine corner locations for better accuracy
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
        # Draw the chessboard corners on the frame for visualization
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        current_time = time.time()  # Get the current time

        # Calculate the movement of the detected corners since the last capture
        if last_corners is not None:
            movement = np.linalg.norm(corners - last_corners)
        else:
            movement = np.inf  # Set to infinity for the first detection

        # Save the image if the capture interval has passed and the chessboard has moved significantly
        if current_time - last_capture_time >= capture_interval and movement > movement_threshold:
            img_count += 1  # Increment the image count
            img_name = os.path.join(output_dir, f'calib_{img_count:02d}.jpg')  # Generate the image filename
            cv2.imwrite(img_name, frame)  # Save the frame as an image
            print(f'Image {img_count} saved: {img_name}')
            last_capture_time = current_time  # Update the last capture time
            last_corners = corners  # Update the last detected corners
        elif movement <= movement_threshold:
            print("Chessboard is stable. Please move it slightly for the next capture.")

    else:
        print("Chessboard not detected. Please adjust the position of the chessboard in the frame.")

    # Display the frame with the drawn chessboard corners
    cv2.imshow('Webcam', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
