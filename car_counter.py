import cv2
from ultralytics import YOLO
import numpy as np

def process_video(video_path=0):  # 0 for webcam
    # Load the YOLO model
    model = YOLO('yolov8n.pt')
    
    # Open the video stream
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if success:
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
            
            # Display the frame
            cv2.imshow('Car Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Use webcam (0) or provide video file path
    process_video('cars.mp4') 