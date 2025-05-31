import cv2
import numpy as np
import io
import os
import requests
import base64
import json
import logging
from requests.exceptions import RequestException, Timeout
from http.server import BaseHTTPRequestHandler
import cgi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image(image):
    try:
        # Resize image if too large (reduce to max 150px)
        height, width = image.shape[:2]
        max_size = 150  # Further reduced from 200 to 150
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"Resized image to {new_width}x{new_height}")

        # Convert image to base64 with lower quality
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 30])  # Further reduced quality
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
            "confidence": 0.95,  # Further increased confidence threshold
            "format": "json",
            "classes": [2, 7],  # Only detect cars (2) and trucks (7)
            "max_det": 3  # Limit maximum detections
        }
        
        logger.info("Sending request to Ultralytics API...")
        # Make the API request with shorter timeout
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=1)  # Reduced timeout to 1s
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

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse the multipart form data
            content_type = self.headers.get('Content-Type')
            if not content_type or 'multipart/form-data' not in content_type:
                self.send_error(400, "Content-Type must be multipart/form-data")
                return
            
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No data received")
                return
            
            # Read the request body
            post_data = self.rfile.read(content_length)
            
            # Parse multipart data
            boundary = content_type.split('boundary=')[1].encode()
            parts = post_data.split(b'--' + boundary)
            
            image_data = None
            for part in parts:
                if b'Content-Disposition: form-data; name="image"' in part:
                    # Extract image data
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        image_data = part[header_end + 4:]
                        # Remove trailing boundary markers
                        if image_data.endswith(b'\r\n'):
                            image_data = image_data[:-2]
                        break
            
            if image_data is None:
                self.send_error(400, "No image found in request")
                return
            
            logger.info(f"Received image of size: {len(image_data)} bytes")
            
            # Check file size (limit to 100KB)
            if len(image_data) > 100 * 1024:
                self.send_error(413, "Image too large. Maximum size is 100KB")
                return
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                self.send_error(400, "Invalid image format")
                return
            
            logger.info(f"Image decoded successfully, shape: {image.shape}")
            
            # Process the image
            processed_image, error = process_image(image)
            
            if error:
                logger.error(f"Error processing image: {error}")
                self.send_error(500, error)
                return
            
            if processed_image is None:
                logger.error("Failed to process image")
                self.send_error(500, "Failed to process image")
                return
            
            # Convert the processed image to bytes with reduced quality
            _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 30])
            image_bytes = buffer.tobytes()
            logger.info(f"Processed image size: {len(image_bytes)} bytes")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Disposition', 'attachment; filename="processed_image.jpg"')
            self.send_header('Content-Length', str(len(image_bytes)))
            self.end_headers()
            self.wfile.write(image_bytes)
            
        except Exception as e:
            logger.error(f"Error in handler: {str(e)}")
            self.send_error(500, str(e))
    
    def do_GET(self):
        self.send_error(405, "Method not allowed. Use POST.")
    
    def log_message(self, format, *args):
        # Override to use our logger
        logger.info(format % args) 