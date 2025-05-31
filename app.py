from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import numpy as np
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')

def generate_frames():
    # Open the video file
    video_path = 'cars.mp4' 
    cap = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = cap.read()
        if not success:
            # If video ends, restart from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add label
                    label = f'Car {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display car count
        cv2.putText(frame, f'Cars: {car_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True) 