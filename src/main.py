import cv2
import numpy as np
import socket
import json
import time
from collections import deque
from utils.apriltag_utils import AprilTagDetector, get_pose, transform_coordinates, adjust_rotation_for_shoulder, \
    determine_body_orientation, is_valid_measurement, round_coords, moving_average, undistort_image
from utils.plotting_utils import plot_path, save_plot_incrementally

# Constants
CAMERA_WIDTH = 1280  # Camera resolution width
CAMERA_HEIGHT = 720  # Camera resolution height
UDP_IP = "127.0.0.1"  # IP address for UDP transmission
UDP_PORT = 5005  # Port for UDP transmission
TAG_SIZE_LARGE = 0.2  # Size of the large AprilTags (in meters)
TAG_SIZE_SMALL = 0.15  # Size of the small AprilTags (in meters)
FRONT_TAG_ID = 2  # ID for the front tag
BACK_TAG_ID = 1  # ID for the back tag
LEFT_SHOULDER_TAG_ID = 6  # ID for the left shoulder tag
RIGHT_SHOULDER_TAG_ID = 0  # ID for the right shoulder tag
MOVEMENT_THRESHOLD = 10  # Minimum movement in pixels to consider significant
CAPTURE_INTERVAL = 2  # Minimum time interval between captures (in seconds)
TEXT_SCALE = 0.75  # Scale for the text overlay in the video
TEXT_THICKNESS = 2  # Thickness for the text overlay in the video

# Load the camera calibration parameters (camera matrix and distortion coefficients).
with np.load('camera-calibration/camera_calib.npz') as data:
    mtx, dist = data['mtx'], data['dist']

# Initialize the AprilTag detector with the specified tag family.
detector = AprilTagDetector("tagStandard41h12")

# Open the webcam and set the resolution to 1280x720 for better accuracy.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Check if the webcam was successfully opened. If not, exit the program.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up the UDP connection for transmitting data.
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize variables and buffers for storing data and smoothing measurements.
last_person_coords = None
last_avg_rotation = None
moving_avg_window = deque(maxlen=5)
path_coords = []

def process_video_stream():
    """
    Function to process the video stream, detect AprilTags, calculate positions and rotations,
    and send the data to a server via UDP. The function also displays the video feed with overlays.
    """
    global last_person_coords, last_avg_rotation

    while True:
        # Read a frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale.
        detections = detector.detect(gray)  # Detect AprilTags in the frame.

        # Initialize variables for the different tag positions.
        front_tag = None
        back_tag = None
        left_shoulder_tag = None
        right_shoulder_tag = None

        # Assign the detected tags to the corresponding body parts.
        for detection in detections:
            if detection['id'] == FRONT_TAG_ID:
                front_tag = detection
                center = tuple(detection['center'].astype(int))
                cv2.putText(frame, "Front", center, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS)
            elif detection['id'] == BACK_TAG_ID:
                back_tag = detection
                center = tuple(detection['center'].astype(int))
                cv2.putText(frame, "Back", center, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS)
            elif detection['id'] == LEFT_SHOULDER_TAG_ID:
                left_shoulder_tag = detection
                center = tuple(detection['center'].astype(int))
                cv2.putText(frame, "Left Shoulder", center, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS)
            elif detection['id'] == RIGHT_SHOULDER_TAG_ID:
                right_shoulder_tag = detection
                center = tuple(detection['center'].astype(int))
                cv2.putText(frame, "Right Shoulder", center, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS)

        # Lists to store the positions and rotations of the tags.
        tag_poses = []
        rotations = []
        person_coords = None
        avg_rotation = None

        # Check if any tags were detected.
        if any([front_tag, back_tag, left_shoulder_tag, right_shoulder_tag]):
            # Calculate the pose (position and rotation) of the detected tags.
            if front_tag is not None:
                front_rvec, front_tvec = get_pose(front_tag, mtx, dist, TAG_SIZE_LARGE)
                tag_poses.append(front_tvec)
                rotations.append(front_rvec)

            if back_tag is not None:
                back_rvec, back_tvec = get_pose(back_tag, mtx, dist, TAG_SIZE_LARGE)
                tag_poses.append(back_tvec)
                rotations.append(back_rvec)

            if left_shoulder_tag is not None:
                left_shoulder_rvec, left_shoulder_tvec = get_pose(left_shoulder_tag, mtx, dist, TAG_SIZE_SMALL)
                left_shoulder_rvec = adjust_rotation_for_shoulder(left_shoulder_rvec, is_left_shoulder=True)
                tag_poses.append(left_shoulder_tvec)
                rotations.append(left_shoulder_rvec)

            if right_shoulder_tag is not None:
                right_shoulder_rvec, right_shoulder_tvec = get_pose(right_shoulder_tag, mtx, dist, TAG_SIZE_SMALL)
                right_shoulder_rvec = adjust_rotation_for_shoulder(right_shoulder_rvec, is_left_shoulder=False)
                tag_poses.append(right_shoulder_tvec)
                rotations.append(right_shoulder_rvec)

            if tag_poses:
                # Calculate the average position and rotation of the detected tags.
                avg_coords = np.mean(tag_poses, axis=0).flatten()
                avg_coords = transform_coordinates(avg_coords)

                # Check if the measurement is valid and smooth the coordinates.
                if is_valid_measurement(avg_coords, last_person_coords):
                    smoothed_coords = round_coords(avg_coords, decimals=2)
                    smoothed_coords = moving_average(smoothed_coords)

                    person_coords = round_coords(avg_coords, decimals=2)
                    avg_rotation = np.mean(rotations, axis=0).flatten()

                    # Display the calculated positions and rotations on the video.
                    avg_rotation_text = f"Avg Rotation: ({avg_rotation[0]:.2f}, {avg_rotation[1]:.2f}, {avg_rotation[2]:.2f})"
                    avg_text = f"Avg Position: ({person_coords[0]:.2f}m, {person_coords[1]:.2f}m, {person_coords[2]:.2f}m)"
                    cv2.putText(frame, avg_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 255), TEXT_THICKNESS)
                    cv2.putText(frame, avg_rotation_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 255), TEXT_THICKNESS)

                    last_person_coords = person_coords
                    last_avg_rotation = avg_rotation

                    # Determine the body orientation based on the detected tags.
                    body_orientation = determine_body_orientation(detections, mtx, dist)
                    print(f"Position: {person_coords}, Rotation: {avg_rotation}, Orientation: {body_orientation}")

                    # Send the calculated data over UDP with error handling.
                    data = {
                        "position": person_coords.tolist(),
                        "rotation": avg_rotation.tolist(),
                        "orientation": body_orientation
                    }
                    message = json.dumps(data)
                    try:
                        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                    except socket.error as e:
                        print(f"Error sending UDP packet: {e}")
                else:
                    print(f"Invalid measurement: {avg_coords} - Exceeds threshold.")
        else:
            # If no new tags are detected, display the last known position and rotation.
            if last_person_coords is not None and last_avg_rotation is not None:
                avg_text = f"Last Position: ({last_person_coords[0]:.2f}m, {last_person_coords[1]:.2f}m, {last_person_coords[2]:.2f}m)"
                avg_rotation_text = f"Last Rotation: ({last_avg_rotation[0]:.2f}, {last_avg_rotation[1]:.2f}, {last_avg_rotation[2]:.2f})"
                cv2.putText(frame, avg_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 255), TEXT_THICKNESS)
                cv2.putText(frame, avg_rotation_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 255), TEXT_THICKNESS)

                person_coords = last_person_coords

                # Send the last known position and rotation data over UDP with error handling.
                data = {
                    "position": person_coords.tolist(),
                    "rotation": last_avg_rotation.tolist(),
                    "orientation": determine_body_orientation(detections, mtx, dist)
                }
                message = json.dumps(data)
                try:
                    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                except socket.error as e:
                    print(f"Error sending UDP packet: {e}")

        # Save the detected positions to the path coordinates list.
        if person_coords is not None:
            path_coords.append(person_coords)

        # Display the video frame with overlays.
        cv2.imshow('AprilTag Person Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    """
    Main function to start processing the video stream and handle cleanup after execution.
    """
    process_video_stream()

    # Release resources after processing is complete.
    cap.release()
    cv2.destroyAllWindows()
    sock.close()

    # Plot the path of the detected coordinates.
    plot_path(path_coords)


if __name__ == "__main__":
    main()
