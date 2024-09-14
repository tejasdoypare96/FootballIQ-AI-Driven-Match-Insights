# Football Match Analysis using AI/ML

This project focuses on analyzing football match videos using advanced machine learning techniques. It tracks players, referees, and the ball in real-time, provides insights like ball possession, team control, player speed, and distance covered, and displays various metrics on the video output.

## Overview

1. **Object Detection & Tracking**:
   - Fine-tuned a **YOLOv8** model on a custom RoboFlow dataset to detect players, referees, and the football.
   - Each detected object is assigned a unique track ID for consistent tracking across frames.

2. **Player & Ball Metrics**:
   - **Player Speed & Distance**: Calculated by transforming pixel positions to real-world field coordinates, adjusted for camera motion using optical flow and perspective transformation.
   - **Ball Control**: The player nearest to the ball is assigned possession based on proximity, with team control calculated by tracking ball possession instances.

3. **Team Classification**:
   - Team colors are extracted from player jerseys using **K-Means Clustering**, classifying players into teams based on the dominant colors.

4. **Camera Motion & View Transformation**:
   - Camera movements are compensated using OpenCV’s optical flow algorithms, ensuring accurate player and ball tracking.
   - Player positions are transformed from pixel space to real field coordinates, enabling accurate speed and distance measurements.

5. **Interpolation**:
   - In case the ball is momentarily lost in a few frames, position interpolation is used to maintain continuous tracking.

6. **Output Visualization**:
   - The processed output video displays player IDs, their speed, distance covered, team classification, and ball possession in real-time.

## Features
- Detects and tracks **players**, **referees**, and the **football**.
- **Classifies players** into teams using jersey colors.
- Calculates **player speeds** and **distances** traveled across the field.
- Determines **which player is in possession** of the ball.
- Displays **team control** over the ball.
- Tracks **camera motion** for accurate positional data.

## Installation & Setup

### Prerequisites
- Python 3.x
- OpenCV
- YOLOv8 (Ultralytics)
- Scikit-learn
- Pandas, Numpy, Matplotlib

## Methodology
------
1. **Model Training**:
   - Fine-tuned YOLOv8 on a Roboflow dataset to detect players, referees, and the ball.

2. **Tracking Data**:
   - Tracks players and ball positions using unique track IDs.
   - Measures the player’s position using the center of their bounding box.
   - Calculates player speed and distance, accounting for camera motion with optical flow and perspective transformation.

3. **Team Classification**:
   - Uses K-means clustering to classify players into teams based on their jersey colors, extracted from the top half of the player's bounding box.

4. **Ball Control**:
   - Assigns ball possession to the player closest to the ball.
   - Tracks overall team control by counting ball possession events.

5. **Data Interpolation**:
   - Uses interpolation for smoother tracking when the ball is temporarily lost in some frames.

6. **Camera Motion & Optical Flow**:
   - Uses OpenCV’s optical flow to measure camera movement and ensure accurate tracking.
   - Transforms player and ball positions to the actual football field using perspective transformation.

## Tools Used
------
- **YOLOv8** for object detection.
- **OpenCV** for video processing and camera motion tracking.
- **Scikit-learn** for team classification using K-means clustering.
- **Pandas, Numpy, Matplotlib** for data handling and visualization.

## Limitations
------
- Misidentification when the ball overlaps with players.
- Detecting goalkeepers and referees could be improved.
- New IDs may be assigned to players who leave and re-enter the frame.

## Future Work
------
- Improve goalkeeper detection by considering field sides.
- Add graphical HUD to display player positions on a field.
- Enhance the representation of data for better clarity.





