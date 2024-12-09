# Person Detection and Tracking System for Autism Spectrum Disorder (ASD) Analysis

## Project Overview
This project presents a sophisticated system designed to detect and track individuals, particularly children with Autism Spectrum Disorder (ASD) and therapists, within video footage. The system assigns unique identifiers to each detected person, ensuring accurate tracking of their movements throughout the video. The solution effectively manages scenarios involving re-entries and occlusions, enabling continuous monitoring.

## Objective
The primary objective of this project is to analyze interactions between children with ASD and therapists by tracking their movements over extended video sessions. The collected tracking data can be utilized to study behavioral patterns, emotional responses, and levels of engagement, which are critical for the development of effective therapeutic strategies.

## System Features
- Person Detection: Utilizes the YOLOv5 model for accurate real-time detection of individuals within video frames.
- Unique ID Assignment: Assigns and maintains unique identifiers for each detected person across video frames.
- Re-entry Management: Handles cases where individuals leave and subsequently re-enter the frame, ensuring continuity in tracking.
- Post-Occlusion Tracking: Re-tracks individuals following partial or complete occlusion, maintaining the integrity of the unique identifiers.
- Focused Tracking: Filters and tracks only relevant individuals (children and therapists), excluding non-relevant objects.

## Technical Breakdown

### Model Deployment
The project employs the YOLOv5 model from Ultralytics, pre-trained on the COCO dataset, which is proficient in detecting 80 classes of objects, including "person." The model is seamlessly integrated using the PyTorch `hub.load` function, which facilitates the easy deployment of pre-trained models.

### Video Processing
The video input is captured using OpenCV’s `cv2.VideoCapture`, enabling frame-by-frame analysis necessary for real-time detection and tracking.

### Detection and Filtering
For each frame, the YOLOv5 model outputs bounding boxes, confidence scores, and class labels. Only the detections classified as "person" (class ID 0) are retained. An optional size-based filtering mechanism is included to distinguish between children and therapists, based on the area of the bounding box.

### Tracking with SORT Algorithm
Filtered detections are passed to the SORT (Simple Online and Realtime Tracking) algorithm, which assigns and maintains unique IDs for each person across video frames. SORT utilizes a combination of Kalman Filters and the Hungarian algorithm to predict and update object trajectories, ensuring consistency in ID assignment even during re-entries or occlusions.

### Handling Re-entries and Occlusions
- Re-entries: SORT reassigns the same ID to individuals re-entering the frame if their predicted trajectory is consistent with previous movements.
- Occlusions: During occlusions, SORT predicts and tracks the individual’s movement until they are visible again, preserving the correct ID.

### Overlaying and Saving Results
Bounding boxes and unique IDs are overlaid onto each frame using OpenCV’s `cv2.rectangle` and `cv2.putText`. The processed frames are compiled and saved as an output video, which can be reviewed to analyze interactions between children and therapists.

## Methodology

### Model Selection
YOLOv5, a state-of-the-art object detection model, is employed for detecting individuals in each frame. The model is pre-trained on the COCO dataset and is known for its efficiency and accuracy.

### Tracking Algorithm
The SORT algorithm assigns unique IDs to detected persons and maintains these IDs across the video frames, ensuring consistent and accurate tracking.

### Filtering Mechanism
Detection results are filtered to focus exclusively on "persons" (children and therapists), ignoring other objects. An optional size threshold is available for differentiating between children and therapists based on bounding box dimensions.

## Installation and Requirements
### Prerequisites
The project requires the following software and libraries:
- Python 3.8+
- PyTorch
- OpenCV
- SORT (Simple Online and Realtime Tracking)
- NumPy

### Installation Steps
1. Clone the Repository
   ```bash
   git clone https://github.com/hemhemaK/Therapist-and-child-detection-/tree/main
   cd Computer_Vision_Therapist_and_Child_Detection_and_Tracking
   ```
2. Install Dependencies:
   ```bash
   conda install -c conda-forge opencv
   pip install torch torchvision sort numpy
   ```

## Usage Instructions
1. Video Placement: Place your input video in the project directory or specify its path in the script.
2. Run the Script: Execute the following command:
   ```bash
   python person_tracker.py
   ```
3. Output: The script processes the video and saves the output with detected persons and their unique IDs at the specified output path.

## Customization Options
- Threshold Adjustment: Modify the size threshold or detection logic in `person_tracker.py` to better distinguish between children and therapists.
- Model Fine-Tuning: Fine-tune the detection model on a custom dataset to improve accuracy for specific scenarios.


