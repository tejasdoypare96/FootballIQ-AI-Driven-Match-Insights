# Import necessary libraries
import cv2  # OpenCV for image and video processing
import sys  # To manipulate Python runtime environment

# Adding the parent directory to the Python path to import custom utilities
sys.path.append('../')  
from utils import measure_distance, get_foot_position  # Custom functions for distance measurement and getting the foot position

# Define a class to estimate speed and distance for objects
class SpeedAndDistance_Estimator():
    def __init__(self):
        """
        Initialize the Speed and Distance Estimator with a frame window and frame rate.
        """
        # Set the window size (number of frames to consider at once) for calculations
        self.frame_window = 5
        # Set the frame rate of the video (frames per second)
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Add speed and distance information to tracked objects (except for the ball and referees).
        :param tracks: Dictionary of tracked objects with their positions and frame numbers.
        """
        # Dictionary to keep track of the total distance covered by each object
        total_distance = {}

        # Loop through each tracked object (e.g., players) in the tracks
        for object, object_tracks in tracks.items():
            # Skip tracking for "ball" and "referees"
            if object == "ball" or object == "referees":
                continue 

            # Get the number of frames for which the object has been tracked
            number_of_frames = len(object_tracks)

            # Loop through frames in steps of frame_window size (e.g., every 5 frames)
            for frame_num in range(0, number_of_frames, self.frame_window):
                # Set the last frame to the current frame plus frame_window, but limit it to the total number of frames
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                # Loop through the tracked IDs (individual objects within the current frame)
                for track_id, _ in object_tracks[frame_num].items():
                    # If the object doesn't exist in the last_frame, skip it
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get the transformed (real-world) position of the object in both the first and last frame
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # Skip if either the start or end positions are missing
                    if start_position is None or end_position is None:
                        continue

                    # Calculate the distance covered between the two frames using the custom measure_distance function
                    distance_covered = measure_distance(start_position, end_position)
                    # Calculate the time that has elapsed between the two frames (in seconds)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    # Calculate speed in meters per second and then convert to kilometers per hour
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Initialize total distance tracking for the object if not already set
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    # Add the distance covered in this frame window to the object's total distance
                    total_distance[object][track_id] += distance_covered

                    # Loop through each frame in the current window and update speed/distance in the tracks
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        # Add the calculated speed and distance to the track information for each frame
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """
        Draw the speed and distance information on each frame of the video.
        :param frames: List of frames from the video.
        :param tracks: Dictionary of tracked objects with speed and distance data.
        :return: List of frames with speed and distance drawn on them.
        """
        output_frames = []

        # Loop through each frame
        for frame_num, frame in enumerate(frames):
            # Loop through each tracked object (e.g., players)
            for object, object_tracks in tracks.items():
                # Skip tracking for "ball" and "referees"
                if object == "ball" or object == "referees":
                    continue

                # Loop through each track (object) in the current frame
                for _, track_info in object_tracks[frame_num].items():
                    # Check if the object has speed data in the current frame
                    if "speed" in track_info:
                        # Get the speed and distance of the object
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        # Skip if either speed or distance is missing
                        if speed is None or distance is None:
                            continue
                        
                        # Get the bounding box (bbox) of the object to determine where to display the text
                        bbox = track_info['bbox']
                        # Get the foot position (lower part) of the bounding box
                        position = get_foot_position(bbox)
                        # Adjust the position to display the speed slightly below the bounding box
                        position = list(position)
                        position[1] += 40  # Move the text 40 pixels below the foot position

                        # Convert the position to integer format (for OpenCV text function)
                        position = tuple(map(int, position))

                        # Draw the speed (in km/h) on the frame at the given position
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        # Draw the distance covered (in meters) just below the speed
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Add the modified frame to the output frames list
            output_frames.append(frame)

        # Return the list of frames with speed and distance information drawn on them
        return output_frames
