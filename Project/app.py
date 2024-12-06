import base64
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO  # Import YOLO from Ultralytics
import cv2

app = Flask(__name__)

# Load the YOLOv8 model
dir_path = os.path.dirname(os.path.realpath(__file__))
model = YOLO(dir_path + '/Models/model1/weights/best.pt').to('cpu')

# Define the upload folder path (use absolute path to avoid issues)
app.config['UPLOAD_FOLDER'] = dir_path + '/static/uploads/'

# Allowed file extensions (for security reasons)
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS_IMAGE.union(ALLOWED_EXTENSIONS_VIDEO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, output_path):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original FPS (e.g., 29.97)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H.264 codec

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Cannot write to file {output_path}")
        return False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame (apply YOLO detections, etc.)
        results = model(frame)
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = result.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label and confidence score
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Blur the detected region (optional, can be removed if not needed)
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            frame[y1:y2, x1:x2] = blurred_roi

            """ x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)ç
            frame[y1:y2, x1:x2] = blurred_roi """

        out.write(frame)

    cap.release()
    out.release()
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Determine if it's an image or a video
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_EXTENSIONS_IMAGE:
        img_stream = file.stream.read()
        img_array = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        processedImage = processImage(img)
        _, buffer = cv2.imencode('.jpg', processedImage)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return render_template('index.html', image_base64=img_base64, result=True)
    elif ext in ALLOWED_EXTENSIONS_VIDEO:
        output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}")
        process_video(file_path, output_video_path)
        return render_template('index.html', video_path=output_video_path, result=True)

@app.route('/display/<filename>')
def display_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)