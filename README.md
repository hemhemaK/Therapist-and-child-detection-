# Therapist-and-child-detection
The code provided is a Python script designed to detect and track people in a video using the YOLOv8 model and Kalman filters. The objective is to classify individuals as either 'Children' or 'Adults (Therapists)' based on their height, and to track their movement across the video frames, even handling scenarios where individuals are temporarily occluded (i.e., disappear and reappear).

**Key Components and Concepts**
**YOLOv8 Model:**
The YOLO (You Only Look Once) model is used for object detection. In this script, YOLOv8 is specifically loaded to detect objects in each frame of the video. The model identifies bounding boxes around detected objects, such as people.

**Video Input:**
The script processes a video file (test.mp4). It reads the video frame by frame using OpenCV, which is a library for computer vision tasks.

**Kalman Filter:**
The Kalman filter is a mathematical model used for tracking and predicting the location of moving objects. It's particularly useful in situations where there might be noise or uncertainty in the object's movement. The script initializes a Kalman filter for each detected person, updating their predicted location in every frame.

**Object Tracking and Unique IDs:**
Each detected person is assigned a unique ID for tracking. The script maintains a mapping between object IDs and their bounding boxes across frames. This ensures that the same person is tracked throughout the video, even when they move or are temporarily occluded.

**Classification Based on Height:**
The script classifies individuals as either 'Children' or 'Adults (Therapists)' based on their height (the vertical dimension of the bounding box). An initial height threshold of 150 pixels is used, but this threshold is dynamically adjusted based on the average height of detected individuals.

**Handling Occlusion:**
The script accounts for situations where an individual might be temporarily out of view (occlusion). If an object is lost (i.e., no longer detected), it is tracked for a certain number of frames (occlusion_timeout), and if it reappears within this period, it is re-associated with its original ID. If not, the object is removed from tracking.

**Detailed Code Walkthrough**

**Initialization:**
The YOLO model is loaded using model = YOLO('yolov8s.pt').
The video file is loaded with cap = cv2.VideoCapture(video_path).
Variables are initialized to keep track of object IDs, bounding boxes, and Kalman filters.

**Main Processing Loop:**
The script enters a loop that processes each frame of the video.
For each frame:
YOLO is used to detect objects.
-The script filters detections to only include persons (cls == 0).
-The height of each bounding box is measured, and based on this, the person is classified as either a 'Child' or 'Adult (Therapist)'.
-The script then tries to match these detections with previously tracked objects using Intersection over Union (IoU), a metric that compares the overlap between two bounding boxes.
-If a match is found, the existing ID is reused; otherwise, a new ID is assigned.

**Tracking with Kalman Filters:**
-For each tracked object, the Kalman filter is used to predict the next position of the person. The Kalman filter is then updated with the actual detection from YOLO.
-The bounding box is drawn around the detected person in the frame, with different colors representing children (green) and adults (red).

**Handling Lost Objects:**
-If an object is not detected in a particular frame, it is marked as "lost" but continues to be tracked for a few frames (occlusion_timeout).
-If the object reappears within this time frame, it is re-associated with its original ID; otherwise, it is removed from tracking.

**Dynamic Threshold Adjustment:**
The height threshold for classifying individuals as children or adults is dynamically adjusted based on the average height of detected persons in each frame.

**Saving the Results:**
The processed video, with tracking and classification, is saved to a file (results/tracking_results.mp4).

**Exiting:**
The script releases all resources and closes any windows displaying the video once processing is complete or if the user interrupts by pressing 'q'.

**Purpose and Use Case**
This script is useful for scenarios where there is a need to monitor and track individuals in a video, such as in a therapy session, classroom, or any environment where it's important to differentiate between children and adults. The tracking ensures that each individual is consistently followed throughout the video, and the system can handle temporary occlusions, which are common in real-world situations.
