# Import the YOLO class from the ultralytics package
from ultralytics import YOLO 

# Load the pre-trained model from the specified path ('models/best.pt')
model = YOLO('models/best.pt')

# Predict objects in the specified video ('input_videos/08fd33_4.mp4')
# `save=True` means the result images with bounding boxes will be saved to disk
results = model.predict('input_videos/08fd33_4.mp4', save=True)

# Print the results of the first frame of the video (results[0])
# This includes the detected objects and related metadata such as bounding boxes and confidence scores
print(results[0])

# Print a separator for clarity
print('=====================================')

# Iterate over each detected object (bounding box) in the first frame of the video
for box in results[0].boxes:
    # Print the information of each bounding box, which usually contains the coordinates, class, and confidence score
    print(box)
