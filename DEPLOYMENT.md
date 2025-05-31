# YOLOv8n Car Detection Deployment Guide

## Option 1: Railway (Recommended) 🚂

Railway offers generous limits and is perfect for ML models.

### Steps:
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "Deploy from GitHub repo"
4. Select this repository
5. Railway will auto-detect the Dockerfile and deploy
6. Your app will be live at `https://your-app-name.railway.app`

### Advantages:
- No size limits for models
- Long execution times
- Auto-scaling
- Free $5/month credit

---

## Option 2: Render 🎨

Free tier available with good performance.

### Steps:
1. Go to [render.com](https://render.com)
2. Connect your GitHub account
3. Create "New Web Service"
4. Select this repository
5. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api/detect.py`
   - **Environment**: Python 3

### Advantages:
- Free tier available
- Easy setup
- Automatic deployments

---

## Option 3: Hugging Face Spaces 🤗

Specifically designed for ML models.

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Choose "Docker" runtime
4. Upload your files
5. Your model will be hosted for free

### Create `app.py` for HF Spaces:
```python
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def detect_cars(image):
    results = model(image)
    result = results[0]
    
    car_count = 0
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls in [2, 7]:  # car or truck
            car_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Car {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(image, f'Cars: {car_count}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

iface = gr.Interface(
    fn=detect_cars,
    inputs=gr.Image(),
    outputs=gr.Image(),
    title="Car Detection with YOLOv8"
)

iface.launch()
```

---

## Option 4: Google Cloud Run ☁️

Pay-per-use, excellent for ML workloads.

### Steps:
1. Install Google Cloud CLI
2. Build and push container:
```bash
gcloud builds submit --tag gcr.io/your-project/car-detection
gcloud run deploy --image gcr.io/your-project/car-detection --platform managed
```

### Advantages:
- Pay only for usage
- Auto-scaling
- High performance
- GPU support available

---

## Option 5: DigitalOcean App Platform 🌊

Simple deployment with good performance.

### Steps:
1. Go to [digitalocean.com/products/app-platform](https://digitalocean.com/products/app-platform)
2. Connect GitHub repository
3. Choose Docker deployment
4. Deploy with auto-detected Dockerfile

---

## Testing Your Deployment

Once deployed, test with:

```bash
curl -X POST -F "image=@test_image.jpg" https://your-app-url.com/ --output result.jpg
```

Or use this Python script:
```python
import requests

url = "https://your-app-url.com/"
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)

with open('result.jpg', 'wb') as f:
    f.write(response.content)
```

## Recommendations by Use Case:

- **For learning/testing**: Hugging Face Spaces (free, easy)
- **For production**: Railway or Google Cloud Run (reliable, scalable)
- **For cost-conscious**: Render (free tier available)
- **For enterprise**: Google Cloud Run or AWS (full control, scaling) 