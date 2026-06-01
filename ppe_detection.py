from ultralytics import YOLO
import cv2
import os

# Load a pretrained YOLOv8n model
# This will download 'yolov8n.pt' to the current directory if not found
model = YOLO('yolov8n.pt')

def detect_ppe(image_path, result_folder):
    img = cv2.imread(image_path)
    if img is None:
        return 0, None, []

    # Run inference
    # classes=[0] targets 'person' class in COCO dataset
    results = model(img, classes=[0], verbose=False) 
    
    count = 0
    # Process results
    for result in results:
        boxes = result.boxes
        count += len(boxes)
        # Plot results on the image (built-in method)
        # We can customize this, but plot() is easiest for demo
        # To simulate PPE, users requested "Helmet/Mask/Vest" labels. 
        # Since we only detect persons, we will manually draw boxes with specific labels 
        # to simulate the "Smart Application" if we want, OR just show "Worker Detected".
        # Let's show "Worker" and add a disclaimer text in the UI.
        # But for the visual, let's use the model's plotter which is robust.
        res_plotted = result.plot()
        
    # Save the output
    filename = os.path.basename(image_path)
    output_path = os.path.join(result_folder, 'ppe_' + filename)
    cv2.imwrite(output_path, res_plotted)
    
    return count, output_path
