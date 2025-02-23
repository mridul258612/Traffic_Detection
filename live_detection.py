import cv2
import torch
from PIL import Image
import numpy as np
from collections import defaultdict
import time

# Load YOLOv5 model for vehicle detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load Haar Cascade for License Plate Detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Open a video file or webcam (0 for default webcam)
cap = cv2.VideoCapture(0)  # Change to '0' for webcam

# Vehicle Counter and Tracking
vehicle_count = defaultdict(int)
counted_vehicles = set()  # Tracks vehicles that have been counted
active_vehicles = set()  # Tracks currently visible vehicles
previous_positions = {}

# Define Vehicle Classes of Interest
vehicle_classes = ["car", "truck", "bus", "motorcycle"]

# Frame dimensions
FRAME_WIDTH = int(cap.get(3))
FRAME_HEIGHT = int(cap.get(4))
line_y = FRAME_HEIGHT // 2  # Define a reference line for vehicle counting

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no frame is read

    # Convert OpenCV image (BGR) to PIL image (RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Run YOLOv5 inference
    results = model(img)

    detected_vehicles = set()

    # Process detection results
    for *xyxy, conf, cls in results.xyxy[0]:  # xyxy (bounding box), conf (confidence), cls (class)
        x1, y1, x2, y2 = map(int, xyxy)
        center_x = (x1 + x2) // 2  # Find center point
        center_y = (y1 + y2) // 2
        detected_class = model.names[int(cls)]
        
        if detected_class in vehicle_classes:
            vehicle_id = f"{detected_class}{x1}{y1}"  # Unique ID for tracking
            
            detected_vehicles.add(vehicle_id)  # Add to currently detected vehicles

            # Count vehicle only if it crosses the middle line and hasn't been counted yet
            if vehicle_id not in counted_vehicles and center_y > line_y:
                vehicle_count[detected_class] += 1  # Increment count
                counted_vehicles.add(vehicle_id)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{detected_class} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # License Plate Detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (px, py, pw, ph) in plates:
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
        cv2.putText(frame, "LICENSE PLATE", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update active vehicles (vehicles currently on screen)
    active_vehicles = detected_vehicles

    # Display the count of currently visible vehicles
    cv2.putText(frame, f"Vehicles on screen: {len(active_vehicles)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display total vehicle counts
    y_offset = 70
    for vehicle, count in vehicle_count.items():
        cv2.putText(frame, f"{vehicle}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30

    # Show the output frame
    cv2.imshow("Traffic Monitoring", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()