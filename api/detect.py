from flask import Flask, request, Response
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8n.pt')

def process_image(image):
    try:
        # Run inference
        logger.info("Running YOLO inference...")
        results = model(image)
        logger.info("Inference completed")
        
        # Process the detections
        car_count = 0
        
        # Get the first result (we only process one image at a time)
        result = results[0]
        
        # Draw boxes for cars and trucks (class ids 2 and 7)
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls in [2, 7]:  # car or truck
                car_count += 1
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Car {conf:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        logger.info(f"Detected {car_count} cars")
        cv2.putText(image, f'Cars: {car_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image, None
            
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return None, str(e)

@app.route('/', methods=['POST'])
def detect_cars():
    try:
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        
        # Read the image file
        file = request.files['image']
        image_bytes = file.read()
        logger.info(f"Received image of size: {len(image_bytes)} bytes")
        
        # Check file size
        if len(image_bytes) > 1024 * 1024:  # 1MB limit
            return {'error': 'Image too large. Maximum size is 1MB'}, 413
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'error': 'Invalid image format'}, 400
        
        logger.info(f"Image decoded successfully, shape: {image.shape}")
        
        # Process the image
        processed_image, error = process_image(image)
        
        if error:
            logger.error(f"Error processing image: {error}")
            return {'error': error}, 500
        
        if processed_image is None:
            return {'error': 'Failed to process image'}, 500
        
        # Convert processed image to bytes
        _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buffer.tobytes()
        logger.info(f"Processed image size: {len(image_bytes)} bytes")
        
        # Return the image
        return Response(
            image_bytes,
            mimetype='image/jpeg',
            headers={'Content-Disposition': 'attachment; filename="processed_image.jpg"'}
        )
        
    except Exception as e:
        logger.error(f"Error in detect_cars: {str(e)}")
        return {'error': str(e)}, 500

@app.route('/', methods=['GET'])
def get_info():
    return {'message': 'Car Detection API. Send POST request with image file.'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 