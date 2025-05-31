import requests

def test_detection(image_path, api_url):
    # Open the image file
    with open(image_path, 'rb') as f:
        files = {'image': f}
        
        # Send POST request to the endpoint
        response = requests.post(api_url, files=files)
        
        # Check if request was successful
        if response.status_code == 200:
            # Save the processed image
            with open('result.jpg', 'wb') as f:
                f.write(response.content)
            print("Success! Processed image saved as 'result.jpg'")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

# Replace with your Vercel URL
API_URL = "https://your-app.vercel.app/detect"
# Replace with path to your test image
IMAGE_PATH = "test_image.jpg"

test_detection(IMAGE_PATH, API_URL) 