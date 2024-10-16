from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename

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
        print("No image part in request")
        return redirect(request.url)

    file = request.files['image']
    
    # Check if a file was actually selected
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

        return redirect(url_for('display_image', filename=filename))

    print("File extension not allowed")
    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)