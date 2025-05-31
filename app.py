from flask import Flask, request, send_file
import cv2
from ultralytics import YOLO
import numpy as np
import io
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')

def process_image(image):
    # Run YOLOv8 inference on the image
    results = model(image)
    
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
            if cls in [2, 7] and conf > 0.5:
                car_count += 1
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label
                label = f'Car {conf:.2f}'
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
    
    # Read the image file
    file = request.files['image']
    image_bytes = file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {'error': 'Invalid image format'}, 400
    
    # Process the image
    processed_image = process_image(image)
    
    # Convert the processed image to bytes
    _, buffer = cv2.imencode('.jpg', processed_image)
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

if __name__ == '__main__':
    app.run(debug=True) 