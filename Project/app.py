from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO  # Import YOLO from Ultralytics

# Load the YOLOv8 model
model = YOLO('/Users/farissiddiqui/Desktop/VS_Code/124_Honors/FA24-Group2/Project/Models/model1/weights/best.pt')

app = Flask(__name__)

# Define the upload folder path (use absolute path to avoid issues)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/uploads/')

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
    
    # Check if a file was selected
    if file.filename == '':
        print("No file selected")
        return redirect(request.url)

    # Validate the file type
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename).lower()  # Sanitize filename and convert to lowercase
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the file to the configured upload folder
        try:
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            print(f"File saved successfully at {filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return redirect(request.url)

        # --- YOLOv8 Inference ---
        try:
            print(f"Running YOLOv8 inference on: {filepath}")
            results = model(filepath, conf=0.25)  # YOLOv8 inference
            result = results[0]  # Access the first result

            print(f"Detections: {len(result.boxes)}")

            # Get the annotated image (bounding boxes, labels) from YOLOv8 as a NumPy array
            annotated_image = result.plot()  # The 'plot()' function returns an image as a NumPy array

            # Save the annotated image using OpenCV (OpenCV is required here)
            import cv2
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')  # Define the path to save the image
            cv2.imwrite(result_image_path, annotated_image)  # Save the result image

            print(f"YOLOv8 inference completed, result saved at {result_image_path}")

            # Redirect to display the processed (annotated) image
            return redirect(url_for('display_image', filename='result.jpg'))
        
        except Exception as e:
            print(f"Error running YOLOv8 inference: {e}")
            return redirect(request.url)

    print("File extension not allowed")
    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)