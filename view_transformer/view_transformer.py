# Import necessary libraries
import numpy as np  # For numerical operations on arrays
import cv2  # OpenCV for image processing and transformation functions

# Define the ViewTransformer class
class ViewTransformer():
    def __init__(self):
        """
        Initialize the transformer by defining the court dimensions and
        calculating the perspective transformation matrix.
        """
        # Define the real-world dimensions of the court in meters
        court_width = 68  # Court width
        court_length = 23.32  # Court length

        # Define the pixel coordinates of the four corner points in the image (camera view)
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom-left corner
            [265, 275],   # Top-left corner
            [910, 260],   # Top-right corner
            [1640, 915]   # Bottom-right corner
        ])

        # Define the real-world target coordinates for the court corners (meters)
        self.target_vertices = np.array([
            [0, court_width],          # Bottom-left (real world)
            [0, 0],                    # Top-left (real world)
            [court_length, 0],         # Top-right (real world)
            [court_length, court_width] # Bottom-right (real world)
        ])

        # Convert pixel and target coordinates to float32 format (required for OpenCV functions)
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Calculate the perspective transformation matrix
        # This matrix helps map pixel coordinates from the camera view to the real-world court view
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    # Method to transform a point from pixel coordinates to real-world court coordinates
    def transform_point(self, point):
        """
        Transforms a given point from the camera's perspective (pixel coordinates)
        to real-world court coordinates.
        :param point: The input point in pixel coordinates.
        :return: The transformed point in real-world coordinates or None if outside the court.
        """
        # Ensure the point is in integer format for pixel-level operations
        p = (int(point[0]), int(point[1]))

        # Check if the point lies within the area defined by the court (pixel_vertices)
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        if not is_inside:
            # If the point is outside the court area, return None
            return None

        # Reshape the point to fit the format required by OpenCV's perspective transformation function
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        # Perform the perspective transformation (convert pixel to real-world coordinates)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        # Reshape the output to a more usable format and return
        return transformed_point.reshape(-1, 2)

    # Method to transform the adjusted positions of objects in a set of tracks
    def add_transformed_position_to_tracks(self, tracks):
        """
        Applies perspective transformation to the adjusted positions of tracked objects.
        :param tracks: The dictionary containing tracking data of objects.
        """
        # Loop through each object in the tracks (e.g., players, referees)
        for object, object_tracks in tracks.items():
            # Loop through the tracked positions of the object in each frame
            for frame_num, track in enumerate(object_tracks):
                # Loop through the tracked data for each object in a given frame
                for track_id, track_info in track.items():
                    # Get the adjusted position of the object in pixel coordinates
                    position = track_info['position_adjusted']
                    position = np.array(position)  # Convert to a numpy array for processing

                    # Transform the position from pixel to real-world coordinates
                    position_transformed = self.transform_point(position)
                    
                    # If the transformation was successful, convert the result to a list
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    # Save the transformed position in the tracks dictionary
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
