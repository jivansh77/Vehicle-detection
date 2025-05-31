from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import os
from ultralytics import YOLO

app = Flask(__name__)

# Initialize model only when needed
def get_model():
    if not hasattr(get_model, 'model'):
        # Use YOLOv8n model with reduced precision
        get_model.model = YOLO('yolov8n.pt')
        # Set model to use half precision
        get_model.model.to('cpu').half()
    return get_model.model

def process_image(image):
    # Get model instance
    model = get_model()
    
    # Resize image if too large (max 800px on longest side)
    height, width = image.shape[:2]
    max_size = 800
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Run YOLOv8 inference on the image
    results = model(image, conf=0.5)  # Increased confidence threshold
    
    # Initialize car counter
    car_count = 0
    
    # Process the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Check if the detected object is a car (class 2) or truck (class 7)
            if cls in [2, 7]:
                car_count += 1
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label with rounded confidence
                label = f'Car {conf:.1f}'
                cv2.putText(image, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display car count
    cv2.putText(image, f'Cars: {car_count}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

@app.route('/detect', methods=['POST'])
def detect_cars():
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    
    try:
        # Read the image file
        file = request.files['image']
        image_bytes = file.read()
        
        # Check file size (limit to 4MB)
        if len(image_bytes) > 4 * 1024 * 1024:
            return {'error': 'Image too large. Maximum size is 4MB'}, 413
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'error': 'Invalid image format'}, 400
        
        # Process the image
        processed_image = process_image(image)
        
        # Convert the processed image to bytes with reduced quality
        _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buffer.tobytes()
        
        # Create a BytesIO object
        img_io = io.BytesIO(image_bytes)
        img_io.seek(0)
        
        # Return the processed image
        return send_file(
            img_io,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='processed_image.jpg'
        )
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True) 