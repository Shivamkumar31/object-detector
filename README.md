# 🎥 Real-Time Object Detection Dashboard with YOLOv8

A production-ready object detection system built with YOLOv8, Streamlit, and FastAPI. Detect objects in real-time from webcam, videos, or images with a beautiful web interface.

## ✨ Features

- **🎬 Real-Time Webcam Detection** - Detect objects live from your webcam
- **📹 Video Processing** - Upload and process videos with detailed statistics
- **🖼️ Image Detection** - Single image object detection
- **📊 Statistics Dashboard** - Visual charts and detailed detection breakdowns
- **🚀 REST API** - FastAPI backend for batch processing
- **🐳 Docker Ready** - One-command deployment
- **⚡ YOLOv8 Powered** - Ultra-fast object detection (30+ FPS)

## 🤖 What is YOLOv8?

**YOLO (You Only Look Once)** is a state-of-the-art object detection algorithm:

```
Input Image → CNN Feature Extraction → Detection Head → Bounding Boxes + Confidence
```

**Why YOLOv8?**
- ✅ Processes image in **single pass** (real-time performance)
- ✅ Pre-trained on **COCO dataset** (80 object classes)
- ✅ **~50ms inference** on modern hardware
- ✅ 95%+ accuracy on standard benchmarks
- ✅ Can be fine-tuned on custom datasets

**Detectable Objects (80 classes):**
People, vehicles (car, bicycle, motorcycle, bus, truck), animals (dog, cat, horse), sports equipment, furniture, food items, and more!

---

## 📦 Dataset Information

### COCO (Common Objects in Context)
- **Training Data:** 330K images with 2.5M instances
- **Classes:** 80 common object categories
- **Pre-trained Model:** YOLOv8 weights already optimized
- **No Download Needed:** Model auto-downloads on first run (~100 MB)

### Available Model Sizes
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| Nano  | 4.2M | ⚡⚡⚡ Fast | 79.3% | Edge devices, real-time |
| Small | 22M  | ⚡⚡ Medium | 86.6% | **Recommended** |
| Medium | 50M | ⚡ Balanced | 88.7% | Better accuracy |
| Large | 94M  | Slow | 89.0% | Maximum accuracy |
| XLarge | 168M | Very slow | 90.2% | Benchmark |

---



###  Local Installation (Without Docker)

**Prerequisites:**
- Python 3.8+ installed
- pip package manager

**Steps:**

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit dashboard
streamlit run app.py

# 4. In another terminal, run API
uvicorn api:app --reload

# 5. Open http://localhost:8501
```

---

## 📖 Usage Guide

### 🎬 Webcam Detection
1. Go to http://localhost:8501
2. Select "📷 Webcam" from sidebar
3. Check "📹 Start Webcam"
4. Objects detected in real-time with FPS counter
5. View detection summary after

### 📹 Video Processing
1. Select "📹 Upload Video"
2. Upload MP4, AVI, MOV, or MKV file
3. Adjust confidence threshold (0.1-1.0)
4. System processes and shows statistics
5. View object count breakdown

### 🖼️ Image Detection
1. Select "🖼️ Upload Image"
2. Upload JPG, PNG, BMP image
3. See instant detection results
4. View bounding boxes and confidence scores

### 🔌 REST API

**Example: Detect objects in an image**
```bash
curl -X POST "http://localhost:8000/api/detect-image" \
  -F "file=@image.jpg" \
  -F "conf_threshold=0.5"
```

**Response:**
```json
{
  "status": "success",
  "total_detections": 3,
  "classes_found": {
    "person": 2,
    "car": 1
  },
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": {
        "x1": 100, "y1": 50, "x2": 200, "y2": 300,
        "width": 100, "height": 250
      }
    }
  ]
}
```

**API Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `POST /api/detect-image` - Detect in image
- `POST /api/detect-video` - Detect in video
- `GET /api/models-info` - Available models
- `GET /api/stats` - API statistics

**API Documentation:**
Open browser: http://localhost:8000/docs (interactive Swagger UI)

---

## 🔧 Configuration

### Adjust Detection Sensitivity
- **Confidence Threshold:** 0.1-1.0
  - Lower (0.1-0.3): More detections, more false positives
  - Default (0.5): Balanced
  - Higher (0.7-0.9): Fewer detections, higher accuracy

- **IOU Threshold:** Controls bounding box overlap
  - Lower: Stricter NMS (fewer overlapping boxes)
  - Higher: More overlapping boxes allowed

### Model Size Selection
```python
# In app.py, change model_size
model_size = "nano"    # Fastest
model_size = "small"   # Recommended
model_size = "medium"  # Better accuracy
model_size = "large"   # High accuracy
```

---

## 📊 Deep Learning Concepts Explained


```
Input Image (RGB pixels)
    ↓
[Convolutional Layer] - Extracts edges, textures
    ↓
[Feature Maps] - Learned representations
    ↓
[Detection Head] - Predicts objects
    ↓
Output: Bounding Boxes + Class Probabilities
```


### How Detection Works

```
For each grid cell:
  ├─ Bounding box: (center_x, center_y, width, height)
  ├─ Objectness: P(object exists)
  └─ Class probs: P(class1), P(class2), ..., P(class80)

Final: Boxes with high confidence are kept
```

---


## 📈 Performance Optimization

### Speed Up Detection

1. **Lower Resolution**
   ```python
   frame = cv2.resize(frame, (320, 320))  # Instead of 640x640
   ```

2. **Skip Frames** (for video)
   ```python
   if frame_count % 2 == 0:  # Process every 2nd frame
       process(frame)
   ```

3. **Use Smaller Model**
   ```python
   model = YOLO("yolov8n.pt")  # Use nano instead of small
   ```

4. **GPU Acceleration** (if you have NVIDIA GPU)
   ```python
   model = YOLO("yolov8s.pt")
   results = model(image, device=0)  # Use GPU
   ```

### Accuracy Improvement

1. **Higher Confidence Threshold** - Fewer false positives
2. **Use Larger Model** - yolov8x is most accurate
3. **Fine-tune on Custom Dataset** - For domain-specific tasks

---

## 🎓 Project Structure

```
object detector/
├── app.py                    # Streamlit web dashboard
├── api.py                    # FastAPI REST API
├── requirements.txt          # Python dependencies
             

```

---

## 🚢 Deployment Options

### Deploy to Cloud

**Heroku (Free tier)**
```bash
# Install Heroku CLI
heroku login
heroku create my-detector
git push heroku main
```

**Google Cloud Run**
```bash
gcloud run deploy my-detector \
  --source . \
  --platform managed
```

**AWS EC2**
```bash
# SSH into instance
ssh -i key.pem ec2-user@instance

# Clone and run
git clone <repo>
cd my-detector
docker-compose up
```

---

## 🐛 Troubleshooting

### Issue: Docker command not found
**Solution:** Install Docker Desktop from https://www.docker.com

### Issue: Port 8501 already in use
**Solution:** Change port in docker-compose.yml
```yaml
ports:
  - "8502:8501"  # Use 8502 instead
```

### Issue: Out of memory
**Solution:** Use smaller model
```python
model_size = "nano"  # Instead of xlarge
```

### Issue: Webcam not detected
**Solution:** Give Docker permission to access webcam
- Docker Desktop > Preferences > Resources > Allow camera access

### Issue: Model download fails
**Solution:** Manual download
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

---

## 📚 Learning Resources

**Official Documentation:**
- YOLOv8: https://docs.ultralytics.com/
- Streamlit: https://docs.streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/
- Docker: https://docs.docker.com/get-started/

**Tutorials:**
- YOLOv8 YouTube: Search "YOLOv8 tutorial"
- Docker Basics: https://docker-curriculum.com/

---

## 💼 Portfolio Tips

When adding this to your resume/portfolio:

1. **GitHub:**
   - Clear README (like this one)
   - Well-commented code
   - Setup instructions
   - Example outputs

2. **Description:**
   ```
   "Developed real-time object detection system using YOLOv8 
   achieving 30+ FPS on modern hardware. Implemented web dashboard 
   with Streamlit and REST API with FastAPI. Containerized with 
   Docker for easy deployment. Processed video files with detection 
   statistics and visualizations."
   ```

3. **Showcase:**
   - Live demo link (if deployed)
   - Demo video/GIF
   - Performance metrics
   - Code snippets

---

## 📄 License

MIT License - Feel free to use for projects

---

## ❓ FAQ

**Q: Can I use my own trained model?**
A: Yes! Replace the model loading in app.py:
```python
model = YOLO("path/to/your/model.pt")
```

**Q: How accurate is YOLOv8?**
A: 90%+ accuracy on COCO dataset. Real-world may vary.

**Q: Can I detect custom objects?**
A: Yes! Fine-tune on your dataset using Ultralytics documentation.

**Q: What GPU do I need?**
A: Any NVIDIA GPU. CPU works too, but slower.

**Q: How do I deploy to production?**
A: Use cloud platforms (AWS, Google Cloud, Heroku, Railway).

---

**Made with ❤️ for ML enthusiasts**

Need help? Check GitHub issues or reach out!
