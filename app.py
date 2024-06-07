from flask import Flask, request, send_file, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from normalize import process_image

app = Flask(__name__)
model = load_model('srcnn_model.h5')

def enhance_image(image_path):
    processed_test_image = process_image(image_path)
    x_test = np.expand_dims(processed_test_image, axis=0)
    enhanced_image = model.predict(x_test)
    enhanced_image = np.squeeze(enhanced_image, axis=0) 
    return enhanced_image

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        enhanced_image = enhance_image(filepath)
        enhanced_image = (enhanced_image * 255).astype(np.uint8)  # Convert to uint8
        
        # Save the enhanced image to a file
        enhanced_filename = 'enhanced_' + filename
        enhanced_filepath = os.path.join('uploads', enhanced_filename)
        img = Image.fromarray(enhanced_image)
        img.save(enhanced_filepath)
        
        return render_template('result.html', original_image=filename, enhanced_image=enhanced_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join('uploads', filename)
    response = send_file(file_path)
    
    # Use a callback to delete the file after sending it
    @response.call_on_close
    def delete_file():
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    
    return response

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
