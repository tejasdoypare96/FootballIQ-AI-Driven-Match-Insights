# Import necessary functions from the utils module
import sys 
sys.path.append('../')  # Add parent directory to system path for module access
from utils import get_center_of_bbox, measure_distance

# Class to assign the ball to the nearest player
class PlayerBallAssigner():
    def __init__(self):
        """
        Initialize the PlayerBallAssigner class.
        `max_player_ball_distance`: Defines the maximum allowable distance (in pixels or units) 
        between a player and the ball to consider the player as the one "possessing" the ball.
        """
        self.max_player_ball_distance = 70  # Maximum distance threshold to assign the ball to a player
    
    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assigns the ball to the closest player based on the bounding boxes of the players and the ball.
        :param players: Dictionary containing player information with their bounding boxes.
        :param ball_bbox: The bounding box of the ball (top-left and bottom-right corners).
        :return: ID of the player assigned to the ball (or -1 if no player is within the defined threshold).
        """
        # Get the center of the ball bounding box (as the ball's position)
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999  # Initialize the minimum distance to a very large number
        assigned_player = -1  # Initialize the assigned player ID to -1 (no player assigned initially)

        # Iterate through each player and their bounding box information
        for player_id, player in players.items():
            player_bbox = player['bbox']  # Get the player's bounding box

            # Measure the distance from the bottom-left corner of the player's bounding box to the ball's center
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)

            # Measure the distance from the bottom-right corner of the player's bounding box to the ball's center
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)

            # Choose the minimum of the two distances (to account for the player's bounding box size)
            distance = min(distance_left, distance_right)

            # If the player is within the defined distance threshold for ball possession
            if distance < self.max_player_ball_distance:
                # Check if this player is closer to the ball than previously found players
                if distance < miniumum_distance:
                    miniumum_distance = distance  # Update the minimum distance
                    assigned_player = player_id  # Assign the ball to the closest player

        # Return the ID of the player who is closest to the ball (or -1 if no player is close enough)
        return assigned_player
