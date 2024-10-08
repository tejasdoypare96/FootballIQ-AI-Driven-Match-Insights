from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Load video frames from the specified file
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize the tracker with the specified model for player and ball detection
    tracker = Tracker('models/best.pt')

    # Get tracking data for objects in the video (e.g., players, ball)
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,  # Load from pre-saved data if available
                                       stub_path='stubs/track_stubs.pkl')  # Path to saved track data
    
    # Add position information to the tracking data
    tracker.add_position_to_tracks(tracks)

    # Initialize camera movement estimator to adjust tracking for camera shifts
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # Get camera movement across all frames (use pre-saved data if available)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    # Adjust player and ball positions based on camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Initialize view transformer to change tracking data into a bird's-eye view
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Fill in missing ball positions by interpolation (estimate positions when data is missing)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize the speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    
    # Add speed and distance information for each player in the tracking data
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign teams to players based on their appearance (e.g., jersey color)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])  # Use the first frame to determine teams
    
    # For each player, determine which team they belong to and add that info to the tracking data
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], 
                                                 track['bbox'], 
                                                 player_id)
            # Store the assigned team and team color for each player
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign the ball to a player (determine which player has possession)
    player_assigner = PlayerBallAssigner()
    team_ball_control = []  # List to store which team has control of the ball at each frame
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # Get the ball's bounding box for the current frame
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)  # Find the player with the ball

        # If a player has the ball, mark them as having possession
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])  # If no one has the ball, use the last known possession
    
    # Convert team ball control to a numpy array
    team_ball_control = np.array(team_ball_control)

    # Draw annotations on the video (e.g., player tracks, ball possession, team control)
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Overlay the camera movement on the video for visualization
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw the speed and distance data on the video
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the output video with all the annotations and overlays
    save_video(output_video_frames, 'output_videos/output_video.avi')

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
