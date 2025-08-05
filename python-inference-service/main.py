import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO  # ✅ YOLOv8

app = FastAPI(title="Palm Fruit Detection API")

# Enable CORS for all origins to allow requests from mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom_model_lite', 'best.pt')

# Detection threshold
MIN_CONFIDENCE_THRESHOLD = 0.7

# Global model variable
model = None

# Initialize the YOLO model
@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = YOLO(MODEL_PATH)
        print("YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        raise

# Base response model
class DetectionResponse(BaseModel):
    success: bool
    message: str
    detections: List[Dict[str, Any]] = []
    primaryResult: Dict[str, Any] = None
    originalDimensions: Dict[str, int] = None

# Base64 image input model
class Base64ImageInput(BaseModel):
    image: str

# Endpoint for health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Endpoint for object detection using multipart/form-data
@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        return process_image(image)
    except Exception as e:
        return {"success": False, "message": f"Error processing image: {str(e)}", "detections": []}

# Endpoint for object detection using base64 encoded image
@app.post("/detect-base64", response_model=DetectionResponse)
async def detect_objects_base64(input_data: Base64ImageInput):
    try:
        image_data = base64.b64decode(input_data.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        return process_image(image)
    except Exception as e:
        return {"success": False, "message": f"Error processing image: {str(e)}", "detections": []}

# Process the image using YOLO model
def process_image(image):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    imH, imW, _ = image.shape
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0]) + 1  # Shift classId to start from 1
            conf = float(box.conf[0])
            if conf < MIN_CONFIDENCE_THRESHOLD:
                continue

            label = model.names[cls_id - 1]  # model.names is 0-indexed
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": label,
                "classId": cls_id,
                "confidence": conf,
                "boundingBox": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "normalized": {
                        "xmin": x1 / imW,
                        "ymin": y1 / imH,
                        "xmax": x2 / imW,
                        "ymax": y2 / imH
                    }
                }
            })

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "success": True,
        "message": "Detection successful",
        "detections": detections,
        "primaryResult": detections[0] if detections else None,
        "originalDimensions": {
            "width": imW,
            "height": imH
        }
    }

# Run the server when script is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)