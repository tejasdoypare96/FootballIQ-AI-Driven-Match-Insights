# Import necessary libraries
import pickle  # To save and load data to/from files
import cv2  # OpenCV for video and image processing
import numpy as np  # For array manipulations
import os  # For handling file paths
import sys  # To manipulate Python runtime environment

# Add the utils module to the Python path (assumed to be in the parent directory)
sys.path.append('../')
# Import custom functions from the utils module for measuring distances
from utils import measure_distance, measure_xy_distance

# Define the CameraMovementEstimator class
class CameraMovementEstimator():
    # Initialize the class with the first video frame
    def __init__(self, frame):
        # Minimum movement distance to consider valid camera movement
        self.minimum_distance = 5

        # Parameters for calculating optical flow (tracking motion between frames)
        self.lk_params = dict(
            winSize = (15, 15),  # Size of the window for tracking
            maxLevel = 2,  # Maximum pyramid level for tracking
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Criteria to stop tracking
        )

        # Convert the first frame to grayscale (used for optical flow calculation)
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask to specify areas where we want to track features
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Track features on the left edge of the frame
        mask_features[:, 900:1050] = 1  # Track features on the right side of the frame

        # Parameters for detecting good features to track in the image
        self.features = dict(
            maxCorners = 100,  # Maximum number of corners to detect
            qualityLevel = 0.3,  # Minimum quality level of the detected corners
            minDistance = 3,  # Minimum distance between detected corners
            blockSize = 7,  # Size of the block used to detect corners
            mask = mask_features  # Use the defined mask for feature detection
        )

    # Method to adjust object positions in the tracks to account for camera movement
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust the position of tracked objects by subtracting the camera movement
        for each frame.
        :param tracks: Object tracking data.
        :param camera_movement_per_frame: Camera movement detected per frame.
        """
        # Loop over all tracked objects (players, referees, etc.)
        for object, object_tracks in tracks.items():
            # Loop over each frame of the object tracks
            for frame_num, track in enumerate(object_tracks):
                # Loop over each track ID in the frame
                for track_id, track_info in track.items():
                    position = track_info['position']  # Get the original position
                    camera_movement = camera_movement_per_frame[frame_num]  # Get camera movement for the current frame
                    # Adjust the object's position by subtracting the camera movement
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # Save the adjusted position in the tracks dictionary
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    # Method to calculate camera movement across video frames
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect camera movement between frames using optical flow.
        :param frames: List of video frames.
        :param read_from_stub: If True, load camera movement data from a stub file.
        :param stub_path: Path to save/load the stub data.
        :return: A list of camera movements for each frame.
        """
        # Check if camera movement data is already saved and read it if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize the list to store camera movement for each frame (starting with no movement)
        camera_movement = [[0, 0]] * len(frames)

        # Convert the first frame to grayscale and detect good features to track
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Loop through all subsequent frames to track camera movement
        for frame_num in range(1, len(frames)):
            # Convert the current frame to grayscale
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # Calculate optical flow (motion) between the old and new frames
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0  # Keep track of the maximum movement distance
            camera_movement_x, camera_movement_y = 0, 0  # Initialize camera movement values

            # Compare feature points between the old and new frames
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()  # Flatten the new feature point array
                old_features_point = old.ravel()  # Flatten the old feature point array

                # Measure the distance between the new and old feature points
                distance = measure_distance(new_features_point, old_features_point)
                # Update camera movement if the current distance is the largest
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # If the maximum movement is greater than the threshold, record the camera movement
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Update the feature points for tracking in the next frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Update the previous frame to be the current frame
            old_gray = frame_gray.copy()

        # Save the camera movement data to a stub file if specified
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        # Return the camera movement data for all frames
        return camera_movement

    # Method to visualize camera movement on the video frames
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draws the camera movement on each frame.
        :param frames: List of video frames.
        :param camera_movement_per_frame: Detected camera movement for each frame.
        :return: List of frames with camera movement annotations.
        """
        output_frames = []  # Initialize list to store output frames

        # Loop through all frames and add the camera movement annotations
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()  # Make a copy of the current frame

            # Draw a semi-transparent overlay rectangle at the top-left corner for the text
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)  # White rectangle
            alpha = 0.6  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend the overlay with the frame

            # Get the camera movement for the current frame (X and Y directions)
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Add text to the frame showing the camera movement in X and Y directions
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            # Append the annotated frame to the output list
            output_frames.append(frame)

        return output_frames  
