from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import io
import os
import requests
import base64
import json
import logging
from requests.exceptions import RequestException, Timeout

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def process_image(image):
    try:
        # Resize image if too large (reduce to max 200px)
        height, width = image.shape[:2]
        max_size = 200  # Further reduced from 300 to 200
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"Resized image to {new_width}x{new_height}")

        # Convert image to base64 with lower quality
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 40])  # Further reduced quality
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.info("Image converted to base64")
        
        # Prepare the request to Ultralytics Hub API
        api_url = "https://api.ultralytics.com/v1/predict/YOLOv8n"  # Using nano model
        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.getenv('ULTRALYTICS_API_KEY')
        }
        
        payload = {
            "image": image_base64,
            "confidence": 0.9,  # Further increased confidence threshold
            "format": "json",
            "classes": [2, 7],  # Only detect cars (2) and trucks (7)
            "max_det": 5  # Limit maximum detections
        }
        
        logger.info("Sending request to Ultralytics API...")
        # Make the API request with shorter timeout
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=2)  # Reduced timeout to 2s
            logger.info(f"API Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API request failed: {response.text}")
                raise Exception(f"API request failed: {response.text}")
            
            # Process the API response
            result = response.json()
            logger.info("Successfully parsed API response")
            
            # Initialize car counter
            car_count = 0
            
            # Process the detections
            if 'predictions' in result:
                for detection in result['predictions']:
                    # Get class and confidence
                    cls = int(detection.get('class', 0))
                    conf = float(detection.get('confidence', 0))
                    
                    # Check if the detected object is a car (class 2) or truck (class 7)
                    if cls in [2, 7]:
                        car_count += 1
                        # Get box coordinates
                        bbox = detection.get('bbox', [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            # Draw bounding box
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Add label with rounded confidence
                            label = f'Car {conf:.1f}'
                            cv2.putText(image, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            logger.info(f"Detected {car_count} cars")
            
            # Display car count
            cv2.putText(image, f'Cars: {car_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return image, None
            
        except Timeout:
            logger.error("API request timed out")
            return None, "API request timed out. Please try again with a smaller image."
        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None, f"API request failed: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return None, str(e)

@app.route('/detect', methods=['POST'])
def detect_cars():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read the image file
        file = request.files['image']
        image_bytes = file.read()
        logger.info(f"Received image of size: {len(image_bytes)} bytes")
        
        # Check file size (limit to 250KB)
        if len(image_bytes) > 250 * 1024:  # Reduced from 500KB to 250KB
            return jsonify({'error': 'Image too large. Maximum size is 250KB'}), 413
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        logger.info(f"Image decoded successfully, shape: {image.shape}")
        
        # Process the image
        processed_image, error = process_image(image)
        
        if error:
            logger.error(f"Error processing image: {error}")
            return jsonify({'error': error}), 500
        
        if processed_image is None:
            logger.error("Failed to process image")
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Convert the processed image to bytes with reduced quality
        _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 40])  # Further reduced quality
        image_bytes = buffer.tobytes()
        logger.info(f"Processed image size: {len(image_bytes)} bytes")
        
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
        logger.error(f"Error in detect_cars: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 