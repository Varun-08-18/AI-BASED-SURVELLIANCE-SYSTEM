import cv2
import numpy as np
import os

def detect_accident(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video"

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    if not ret:
        return False, "Not enough frames"

    # Heuristic: Monitor the magnitude of change between frames
    # Large sudden changes -> possible crash or fast movement
    
    alert_triggered = False
    max_diff = 0
    
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Count non-zero pixels after thresholding
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total motion area
        motion_area = sum([cv2.contourArea(c) for c in contours])
        
        # Get frame total area
        height, width = frame1.shape[:2]
        total_area = height * width
        
        # If motion covers a significant portion unexpectedly, or just raw high motion
        # This is very naive. A better "Accident" heuristic for demo is sudden acceleration.
        # But for visual "accident", often there is a lot of chaotic motion.
        # Let's set a threshold.
        
        motion_ratio = motion_area / total_area
        
        # Threshold (tuned for demo purposes)
        if motion_ratio > 0.3: # If >30% of screen changes drastically
            alert_triggered = True
            break
            
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
            
    cap.release()
    
    # Return result
    return alert_triggered
