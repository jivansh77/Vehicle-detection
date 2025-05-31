import requests
import cv2
import numpy as np

def test_detection():
    # Read and resize image
    image = cv2.imread('test_image.jpg')
    if image is None:
        print("Error: Could not read test_image.jpg")
        return
    
    height, width = image.shape[:2]
    max_size = 150  # Match the server's max size
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Save resized image with lower quality
    cv2.imwrite('test_image_resized.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 30])
    
    # Test local endpoint first (if running locally)
    # url = 'http://localhost:3000/api/detect'
    
    # Test deployed endpoint
    url = 'https://vehicle-detection-flame.vercel.app/api/detect'
    
    try:
        with open('test_image_resized.jpg', 'rb') as f:
            files = {'image': f}
            print(f"Sending request to {url}...")
            response = requests.post(url, files=files, timeout=10)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            with open('result.jpg', 'wb') as f:
                f.write(response.content)
            print("Success! Result saved as result.jpg")
            print(f"Result image size: {len(response.content)} bytes")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response text: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_detection() 