
# 🐟 Fisheye Camera Object Detection Using Deep Learning

A Computer Vision project focused on detecting objects in **fisheye camera images** using state-of-the-art object detection and segmentation models. This work includes a **Flask web app** that enables real-time predictions on **images, image batches, and videos**, with support for multiple model selections.

## 📌 Project Overview

Fisheye cameras introduce **extreme spatial distortion**, making object detection more complex. To tackle this, we benchmarked several deep learning models on the **FishEye8K** dataset — containing 8,000 images with 157,000 object annotations — across five classes:
- Car
- Pedestrian
- Truck
- Bus
- Bike

### 🔍 Models Evaluated
| Model          | Type        | Highlights                               |
|----------------|-------------|-------------------------------------------|
| YOLOv8         | One-stage   | Real-time, balanced detection             |
| Faster R-CNN   | Two-stage   | High precision, poor recall               |
| RetinaNet      | One-stage   | Focal Loss, moderate overall performance  |
| YOLOv8-Seg     | Segmentation| Best performance, pixel-level accuracy    |

## 🌐 Web Application

### 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Frontend:** HTML/CSS (Jinja2 templating)
- **Frameworks:** PyTorch, Ultralytics YOLOv8
- **Folders:**
  ```
  fisheye_object_detection/
  ├── app.py
  ├── templates/
  │   ├── index.html
  │   ├── results.html
  │   ├── batch_results.html
  │   └── video_results.html
  ├── static/
  │   └── uploads/
  └── models/
  ```

### 🖼️ Features
- Select models (YOLOv8, Faster R-CNN, RetinaNet, YOLOv8-Seg)
- Upload images (single or batch)
- Upload videos for detection
- Visual feedback for predicted results
- Segmentation overlay for YOLOv8-Seg

## 🧪 Experimental Results

### 🔢 Overall Performance

| Metric          | YOLOv8 | Faster R-CNN | RetinaNet | YOLOv8-Seg |
|-----------------|--------|--------------|-----------|------------|
| Precision (%)   | 60.87  | 80.29        | 67.40     | 62.0       |
| Recall (%)      | 49.05  | 17.60        | 34.66     | 43.5       |
| mAP@0.5 (%)     | 20.24  | 11.70        | 19.15     | **52.9**   |
| mAP@0.75 (%)    | 16.19  | 9.36         | 15.32     | **30.2**   |
| IoU@0.5 (%)     | 37.29  | 16.87        | 29.68     | **57.3**   |

### ✅ Key Findings
- **YOLOv8-Segmentation** significantly outperformed all other models.
- Instance segmentation is **more effective** for distorted fisheye objects.
- Standard detection models struggle with **edge distortion** and **small object detection** (e.g., pedestrians, trucks).

## 🧠 Insights

1. **Segmentation is Superior:** Pixel-level masks align better with distortion than bounding boxes.
2. **Distortion Matters:** Detection is more accurate in image centers than edges.
3. **YOLOv8** had the best balance between recall and precision among detection models.
4. **Faster R-CNN** had high precision but extremely low recall.

## 🚀 How to Run the Web App

### 🔧 Installation
```bash
git clone https://github.com/yourusername/fisheye-object-detection.git](https://github.com/arda92a/fisheye-object-detection.git
cd fisheye-object-detection
pip install -r requirements.txt
```

### ▶️ Run Flask App
```bash
python app.py
```

The app will be available at `http://127.0.0.1:5000`.

## 📁 Dataset

**FishEye8K**  
- 8,000 fisheye images  
- 157,000 bounding box annotations  
- 5 object classes: `car`, `bike`, `pedestrian`, `bus`, `truck`  
- Annotations were converted from VOC XML to COCO and YOLO formats  
- Masks generated using [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)

## 📊 Evaluation Metrics

- **Intersection over Union (IoU)**
- **Precision / Recall / F1 Score**
- **mean Average Precision (mAP@0.5 and mAP@0.75)**

## 📈 Future Work

- Introduce **fisheye-aware anchor generation**
- Design **distortion-specific data augmentations**
- Explore **transformer-based segmentation** tailored to radial distortion
- Collect more diverse fisheye datasets

## 🧠 Author

**Arda Öztüner**  
Computer Engineering, Eskisehir Technical University  
Email: ardaoztuner@ogr.eskisehir.edu.tr
