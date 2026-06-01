import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename, file_type):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if file_type == 'image':
        return ext in ALLOWED_EXTENSIONS_IMG
    elif file_type == 'video':
        return ext in ALLOWED_EXTENSIONS_VID
    return False

from modules import face_detection, ppe_detection, traffic_analysis, accident_detection

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face', methods=['POST'])
def face_detection_route():
    return handle_upload(request, 'face')

@app.route('/ppe', methods=['POST'])
def ppe_detection_route():
    return handle_upload(request, 'ppe')

@app.route('/traffic', methods=['POST'])
def traffic_analysis_route():
    return handle_upload(request, 'traffic')

@app.route('/accident', methods=['POST'])
def accident_detection_route():
    return handle_upload(request, 'accident')

def handle_upload(req, module_name):
    if 'file' not in req.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = req.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    # Determine expected type based on module
    file_type = 'video' if module_name in ['traffic', 'accident'] else 'image'
    
    if file and allowed_file(file.filename, file_type):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result_text = ""
        processed_image = None
        
        # Call specific module function
        if module_name == 'face':
            count, output_path = face_detection.detect_faces(filepath, app.config['RESULT_FOLDER'])
            result_text = f"Detected {count} Faces."
            if output_path:
                processed_image = os.path.basename(output_path)
                
        elif module_name == 'ppe':
            count, output_path = ppe_detection.detect_ppe(filepath, app.config['RESULT_FOLDER'])
            result_text = f"Detected {count} Persons (Potential Workers)."
            if output_path:
                processed_image = os.path.basename(output_path)
                
        elif module_name == 'traffic':
            count, density = traffic_analysis.analyze_traffic(filepath, app.config['RESULT_FOLDER'])
            result_text = f"Average Vehicles per Frame: {count}<br>Traffic Density: <strong>{density}</strong>"
            # Traffic might not produce an output image unless we want to verify one frame
            # For now, let's just show the video and the stats
            
        elif module_name == 'accident':
            alert = accident_detection.detect_accident(filepath)
            if alert:
                result_text = "<span class='text-danger fw-bold'>ALERT: Possible Accident or Sudden Motion Detected!</span>"
            else:
                result_text = "<span class='text-success'>No Abnormal Motion Detected.</span>"
        
        return render_template('result.html', 
                               module=module_name, 
                               filename=filename,
                               result_text=result_text,
                               processed_image=processed_image,
                               file_type=file_type)
    else:
        flash(f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS_IMG if file_type=="image" else ALLOWED_EXTENSIONS_VID}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
