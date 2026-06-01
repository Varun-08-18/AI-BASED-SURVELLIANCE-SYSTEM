import cv2
import os

def detect_faces(image_path, result_folder):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        return 0, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Save the output
    filename = os.path.basename(image_path)
    output_path = os.path.join(result_folder, 'face_' + filename)
    cv2.imwrite(output_path, img)
    
    return len(faces), output_path
