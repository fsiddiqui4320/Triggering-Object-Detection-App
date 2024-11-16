import torch
print(torch.cuda.is_available())


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
#print("This is current directoy" + os.getcwd())
#print("This is file directoy" + dir_path)

# Allowed file extensions (for security reasons)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        print("No image part in the request")
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        print("No file selected")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename).lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"Running YOLOv8 inference on: {filepath}")
        results = model(filepath)  # YOLOv8 inference
        result = results[0]
        detections_count = len(result.boxes)  # Get number of detections

        image = cv2.imread(filepath)

        # Draw bounding boxes and save the image (as in previous steps)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = result.names[cls]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            roi = image[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            image[y1:y2, x1:x2] = blurred_roi

        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_annotated.jpg')
        cv2.imwrite(result_image_path, image)

        # Pass the filename and detections_count to the template
        return render_template('index.html', filename='result_annotated.jpg', detections_count=detections_count)

    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)