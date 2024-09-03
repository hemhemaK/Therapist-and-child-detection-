from collections import defaultdict
from ultralytics import YOLO
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import os

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load video
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

# Define initial height threshold for classification
initial_height_threshold = 150
dynamic_threshold_adjustment = 10

# Initialize unique ID counter
unique_id = 0

# Create dictionaries to keep track of detected objects and their IDs
object_id_map = {}
previous_boxes = {}
tracker_data = defaultdict(lambda: {'last_seen': 0, 'box': None, 'kf': None})

# Define lost objects tracking
lost_objects = defaultdict(lambda: {'last_seen': 0, 'box': None})
occlusion_timeout = 50  # Number of frames an object can be lost before being removed


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    x_left = max(x1, x1_p)
    y_top = max(y1, y1_p)
    x_right = min(x2, x2_p)
    y_bottom = min(y2, y2_p)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000
    kf.R = np.array([[50, 0],
                     [0, 50]])
    kf.Q = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    return kf


# Create results folder if it doesn't exist
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(os.path.join(results_folder, 'tracking_results.mp4'), fourcc, 20.0,
                               (int(cap.get(3)), int(cap.get(4))))

ret = True
frame_count = 0
while ret:
    ret, frame = cap.read()
    frame_count += 1

    if ret:
        # Detect and track objects
        results = model.track(frame, persist=True)

        # Filter detections to include only persons and classify them
        current_boxes = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()

                    height_box = y2 - y1
                    person_type = 'Child' if height_box < initial_height_threshold else 'Adult(Therapist)'
                    current_boxes.append((x1, y1, x2, y2, person_type))

        # Match current detections with previous boxes
        new_object_id_map = {}
        id_assigned = set()
        current_ids = set(previous_boxes.keys())

        for (x1, y1, x2, y2, person_type) in current_boxes:
            best_id = None
            best_iou = 0.0

            for obj_id, (prev_x1, prev_y1, prev_x2, prev_y2) in previous_boxes.items():
                iou = compute_iou((x1, y1, x2, y2), (prev_x1, prev_y1, prev_x2, prev_y2))
                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id

            if best_iou > 0.2:
                label_id = best_id
            else:
                unique_id += 1
                label_id = unique_id

            new_object_id_map[label_id] = (x1, y1, x2, y2)
            id_assigned.add(label_id)

            if tracker_data[label_id]['kf'] is None:
                tracker_data[label_id]['kf'] = initialize_kalman_filter()
                tracker_data[label_id]['kf'].x = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, 0, 0])
                tracker_data[label_id]['last_seen'] = frame_count

            kf = tracker_data[label_id]['kf']
            kf.predict()
            kf.update(np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]))
            (x, y) = kf.x[:2]

            color = (0, 255, 0) if person_type == 'Child' else (0, 0, 255)
            cv2.rectangle(frame, (int(x - (x2 - x1) / 2), int(y - (y2 - y1) / 2)),
                          (int(x + (x2 - x1) / 2), int(y + (y2 - y1) / 2)), color, 2)
            cv2.putText(frame, f'{person_type} ID: {label_id}', (int(x - (x2 - x1) / 2), int(y - (y2 - y1) / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            id_assigned.add(label_id)

        # Handle lost objects
        for obj_id, data in tracker_data.items():
            if obj_id not in id_assigned:
                lost_objects[obj_id] = {'last_seen': frame_count, 'box': data['box']}

        for obj_id, lost_data in list(lost_objects.items()):
            lost_box = lost_data['box']
            if lost_box is None:
                print(f"Warning: Lost box for object ID {obj_id} is None")
                continue

            if frame_count - lost_data['last_seen'] <= occlusion_timeout:
                for (x1, y1, x2, y2, person_type) in current_boxes:
                    iou = compute_iou(lost_box, (x1, y1, x2, y2))
                    if iou > 0.5:
                        new_object_id_map[obj_id] = (x1, y1, x2, y2)
                        lost_objects.pop(obj_id)
                        break
            else:
                tracker_data.pop(obj_id, None)
                lost_objects.pop(obj_id, None)

        # Update previous boxes for the next frame
        previous_boxes = {id_: box for id_, box in new_object_id_map.items() if id_ in id_assigned}

        # Adjust the height threshold dynamically if needed
        if len(current_boxes) > 0:
            height_values = [y2 - y1 for (x1, y1, x2, y2, _) in current_boxes]
            average_height = np.mean(height_values)
            initial_height_threshold = max(150, average_height - dynamic_threshold_adjustment)

        video_writer.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
