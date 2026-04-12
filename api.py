"""
FastAPI Backend for Object Detection
Handles batch processing and REST API endpoints
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import tempfile
import os
import json
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from pathlib import Path
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Object Detection API",
    description="YOLOv8 Real-Time Object Detection API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
try:
    MODEL = YOLO("yolov8s.pt")  # Small model for balance of speed/accuracy
    logger.info("✅ YOLOv8 model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    MODEL = None

# Store for temporary files
TEMP_DIR = Path("./temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Object Detection API",
        "version": "1.0.0",
        "status": "🟢 Running",
        "model": "YOLOv8 Small",
        "endpoints": {
            "health": "/health",
            "detect_image": "/api/detect-image",
            "detect_video": "/api/detect-video",
            "models_info": "/api/models-info"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

# Models info
@app.get("/api/models-info")
async def models_info():
    """Get available YOLO models and classes"""
    return {
        "available_models": ["nano", "small", "medium", "large", "xlarge"],
        "current_model": "yolov8s",
        "num_classes": 80,
        "classes": list(MODEL.names.values()) if MODEL else [],
        "description": "YOLO models trained on COCO dataset"
    }

# Image detection endpoint
@app.post("/api/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """
    Detect objects in an image
    
    Parameters:
    - file: Image file (jpg, png, etc.)
    - conf_threshold: Confidence threshold (0.0-1.0)
    
    Returns:
    - Detected objects with bounding boxes and confidence scores
    """
    
    if not MODEL:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Save uploaded file
        temp_file = TEMP_DIR / file.filename
        with open(temp_file, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Read image
        image = cv2.imread(str(temp_file))
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = MODEL(image, conf=conf_threshold, verbose=False)
        
        # Extract results
        detections = []
        classes_count = defaultdict(int)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = MODEL.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2)
                    }
                })
                
                classes_count[class_name] += 1
        
        # Save annotated image
        annotated_image = results[0].plot()
        output_path = TEMP_DIR / f"annotated_{file.filename}"
        cv2.imwrite(str(output_path), annotated_image)
        
        # Cleanup
        os.remove(temp_file)
        
        return {
            "status": "success",
            "total_detections": len(detections),
            "classes_found": dict(classes_count),
            "detections": detections,
            "annotated_image_path": str(output_path)
        }
    
    except Exception as e:
        logger.error(f"Error in image detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Video detection endpoint
@app.post("/api/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    skip_frames: int = 1
):
    """
    Detect objects in a video
    
    Parameters:
    - file: Video file (mp4, avi, etc.)
    - conf_threshold: Confidence threshold
    - skip_frames: Process every nth frame (1=all, 2=every 2nd, etc.)
    
    Returns:
    - Video statistics and detection summary
    """
    
    if not MODEL:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Save uploaded file
        temp_file = TEMP_DIR / file.filename
        with open(temp_file, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Open video
        cap = cv2.VideoCapture(str(temp_file))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Process video
        frame_count = 0
        processed_frames = 0
        total_detections = 0
        classes_count = defaultdict(int)
        frame_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if requested
            if frame_count % skip_frames != 0:
                continue
            
            processed_frames += 1
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Run detection
            results = MODEL(frame, conf=conf_threshold, verbose=False)
            
            frame_dets = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = MODEL.names[class_id]
                    
                    frame_dets.append({
                        "class": class_name,
                        "confidence": round(conf, 3)
                    })
                    
                    classes_count[class_name] += 1
                    total_detections += 1
            
            frame_detections.append({
                "frame": frame_count,
                "detections_count": len(frame_dets),
                "objects": frame_dets
            })
        
        cap.release()
        os.remove(temp_file)
        
        # Calculate statistics
        avg_detections = total_detections / processed_frames if processed_frames > 0 else 0
        
        return {
            "status": "success",
            "video_info": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": fps,
                "resolution": f"{width}x{height}"
            },
            "statistics": {
                "total_detections": total_detections,
                "average_detections_per_frame": round(avg_detections, 2),
                "classes_found": dict(classes_count)
            },
            "frame_details": frame_detections[:20]  # Return first 20 frames
        }
    
    except Exception as e:
        logger.error(f"Error in video detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch detection endpoint
@app.get("/api/detect-batch")
async def detect_batch_info():
    """
    Get information about batch detection
    """
    return {
        "description": "Send multiple images/videos for batch processing",
        "usage": "POST requests to /api/detect-image or /api/detect-video",
        "example": {
            "endpoint": "/api/detect-image",
            "method": "POST",
            "parameters": {
                "file": "image.jpg",
                "conf_threshold": 0.5
            }
        }
    }

# Statistics endpoint
@app.get("/api/stats")
async def get_stats():
    """
    Get API statistics
    """
    return {
        "model_info": {
            "name": "YOLOv8 Small",
            "classes": 80,
            "inference_time_ms": "~20-50ms per frame",
            "training_dataset": "COCO"
        },
        "supported_formats": {
            "images": ["jpg", "jpeg", "png", "bmp", "webp"],
            "videos": ["mp4", "avi", "mov", "mkv"]
        },
        "parameters": {
            "conf_threshold": "Detection confidence (0.0-1.0)",
            "skip_frames": "Process every nth frame for video"
        }
    }

# Custom error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": "error"},
    )

if __name__ == "__main__":
    # Run with: uvicorn api.py --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
