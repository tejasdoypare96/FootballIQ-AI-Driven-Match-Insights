# Import necessary libraries
from ultralytics import YOLO  # YOLO for object detection
import supervision as sv  # Supervision for object tracking
import pickle  # For saving and loading data
import os  # To handle file paths
import numpy as np  # For array manipulations
import pandas as pd  # For data manipulation and interpolation
import cv2  # OpenCV for video and image processing
import sys  # To manipulate the Python runtime environment

# Append utils module path (assumed to be in the parent directory)
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position  # Custom utility functions

# Tracker class for object detection and tracking
class Tracker:
    def __init__(self, model_path):
        """
        Initialize the tracker with a YOLO model and ByteTrack for tracking objects.
        :param model_path: Path to the pre-trained YOLO model.
        """
        self.model = YOLO(model_path)  # Load YOLO model for object detection
        self.tracker = sv.ByteTrack()  # Initialize ByteTrack tracker

    def add_position_to_tracks(self, tracks):
        """
        Add position (center or foot position) to each tracked object in the tracks.
        :param tracks: Dictionary containing tracks of objects across frames.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    # Determine position based on object type
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates the missing ball positions between frames.
        :param ball_positions: List of ball positions over frames.
        :return: Interpolated ball positions.
        """
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values and fill any NaNs
        df_ball_positions = df_ball_positions.interpolate().bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        """
        Detect objects in video frames using YOLO in batches.
        :param frames: List of frames to detect objects in.
        :return: Detections across all frames.
        """
        batch_size = 20  # Define batch size for detection
        detections = []  # Initialize list to store detections

        for i in range(0, len(frames), batch_size):
            # Detect objects in batch of frames
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch  # Append detections to the list

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect and track objects in video frames. Optionally load from or save to stub.
        :param frames: List of video frames.
        :param read_from_stub: Whether to read tracks from a pre-saved stub.
        :param stub_path: Path to the stub file.
        :return: Object tracks.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Detect objects in frames
        detections = self.detect_frames(frames)

        # Initialize dictionary to store tracks
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Process each frame's detections
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Class names
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Inverse class names dictionary

            # Convert to supervision's Detections format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert "goalkeeper" to "player" class for simplicity
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track objects using ByteTrack
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize dictionary entries for current frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Process tracked objects
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Add player and referee tracks
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Add ball tracks (since ball is not tracked)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save tracks to stub if provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse to represent players or referees on the frame.
        :param frame: The video frame.
        :param bbox: The bounding box of the object.
        :param color: The color to draw the ellipse.
        :param track_id: Optional track ID to display.
        :return: Frame with drawn ellipse.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw an ellipse at the bottom center of the bounding box
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw track ID (if available)
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12

            # Adjust text position if track ID is large
            if track_id > 99:
                x1_text -= 10

            # Draw track ID
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw a triangle to represent ball possession over a player.
        :param frame: The video frame.
        :param bbox: The bounding box of the object.
        :param color: The color of the triangle.
        :return: Frame with drawn triangle.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Define triangle points
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        # Draw triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Display ball possession percentage on the frame for both teams.
        :param frame: The video frame.
        :param frame_num: The current frame number.
        :param team_ball_control: Array representing ball control for each frame.
        :return: Frame with ball control information displayed.
        """
        overlay = frame.copy()

        # Draw a semi-transparent rectangle to display ball control stats
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames

        # Calculate and display ball control percentages
        team_1 = team_1_num_frames / total_frames
        team_2 = team_2_num_frames / total_frames

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draw annotations (players, referees, ball, ball control) on the video frames.
        :param video_frames: List of frames to annotate.
        :param tracks: Tracked objects.
        :param team_ball_control: Array of ball control information for each frame.
        :return: Annotated video frames.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # Draw ball possession indicator
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw ball control information
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
