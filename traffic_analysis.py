from ultralytics import YOLO
import cv2
import os
import numpy as np

# Reuse the model or load a new one
model = YOLO('yolov8n.pt')

def analyze_traffic(video_path, result_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, "Error", None

    vehicle_counts = []
    frames_count = 0
    
    # Process every 10th frame to speed up (demo constraint)
    # Vehicles in COCO: car(2), motorcycle(3), airplane(4), bus(5), train(6), truck(7), boat(8)
    # We care about: car, motorcycle, bus, truck -> indices 2, 3, 5, 7
    vehicle_classes = [2, 3, 5, 7]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_count += 1
        if frames_count % 10 != 0:
            continue
            
        # Inference
        results = model(frame, classes=vehicle_classes, verbose=False)
        
        # Count vehicles in this frame
        current_count = len(results[0].boxes)
        vehicle_counts.append(current_count)
        
        # Stop after 100 processing steps (approx 30s of video at 30fps if stepping every 10)
        # to keep it fast for demo
        if len(vehicle_counts) > 300: 
            break
            
    cap.release()
    
    if not vehicle_counts:
        return 0, "Low", None

    avg_count = int(np.mean(vehicle_counts))
    
    # Density Logic
    if avg_count <= 5:
        density = "Low"
    elif avg_count <= 15:
        density = "Medium"
    else:
        density = "High"
        
    return avg_count, density
