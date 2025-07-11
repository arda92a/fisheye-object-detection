# Fisheye Camera Object Detection

This project provides a Flask-based web application for object detection and segmentation in fisheye camera images and videos. It supports multiple deep learning models, including YOLO (for object detection and segmentation), RetinaNet, and Faster R-CNN, to detect and classify objects such as bikes, buses, cars, pedestrians, and trucks. The application allows users to upload images or videos, select or upload pre-trained models, and visualize detection results with bounding boxes and optional segmentation masks. Ground truth annotations in YOLO format can also be visualized for comparison.

## Features

- **Model Support**: Supports YOLO (object detection and segmentation), RetinaNet, and Faster R-CNN models.
- **Image Processing**: Upload single or multiple images (JPEG, PNG) for object detection or segmentation.
- **Video Processing**: Upload videos (MP4, AVI) for frame-by-frame object detection.
- **Ground Truth Visualization**: Displays ground truth bounding boxes from YOLO-format `.txt` label files.
- **Web Interface**: User-friendly interface with Tailwind CSS for responsive design and dark mode support.
- **API Endpoint**: Provides a `/api/detect` endpoint for programmatic image detection.
- **Customizable Confidence Threshold**: Adjust the detection confidence threshold (default: 0.5).
- **Color-Coded Classes**: Visualizes detections with class-specific colors (e.g., Bike: Orange, Car: Green).
- **Batch Processing**: Process multiple images simultaneously and view results in a batch format.
- **Downloadable Results**: Download original, ground truth, and processed images.

## Project Structure

```
├── .gitignore
├── app.py                     # Main Flask application
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── models/                    # Directory for pre-trained model files
│   ├── faster_rcnn_model.pth
│   ├── retinanet_model.pth
│   ├── yolo_object_detection.pt
│   ├── yolo_segmentation.pt
├── templates/                 # HTML templates for the web interface
│   ├── batch_results.html     # Displays batch image processing results
│   ├── index.html            # Home page for model and file uploads
│   ├── results.html          # Displays single image processing results
│   ├── video_results.html    # Displays video processing results
├── training_codes/            # Training scripts for models
│   ├── faster_rcnn.py
│   ├── retinanet.py
│   ├── segmentation.py
│   ├── train_yolo_od.py
├── static/uploads/            # Directory for uploaded and processed files
└── static/labels/            # Directory for ground truth label files
```

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup

1. **Install Dependencies**:
   Install the required Python packages using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Model Files**:
   Place pre-trained model files from (https://drive.google.com/drive/folders/1f5sj08ztjm7DjV1sYxSHa-dVm-vnA56z?usp=sharing) (`.pt`, `.pth`, `.weights`) in the `models/` directory. Example models included:
   - `faster_rcnn_model.pth`
   - `retinanet_model.pth`
   - `yolo_object_detection.pt`
   - `yolo_segmentation.pt`

3. **Run the Application**:
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`.

## Usage

### Web Interface

1. **Access the Application**:
   Open `http://localhost:5000` in a web browser.

2. **Load a Model**:
   - **Upload New Model**: Use the "Upload New Model" section to upload a `.pt`, `.pth`, or `.weights` file. Select the model type (YOLO, Segmentation, RetinaNet, or Faster R-CNN) and set a confidence threshold (0.1 to 0.9).
   - **Select Existing Model**: Choose a model from the `models/` directory and specify its type and threshold.

3. **Upload Images**:
   - Upload one or more images (JPEG or PNG) in the "Upload Image(s)" section.
   - For single images, results are displayed on the `/results` page with original, ground truth (if `.txt` labels are provided in `static/labels/`), and processed images.
   - For multiple images, results are shown on the `/batch_results` page.

4. **Upload Videos**:
   - Upload a video (MP4 or AVI) in the "Upload Video" section.
   - Processed video results are displayed on the `/video_results` page.

5. **View Results**:
   - **Single Image Results** (`results.html`): Shows the original image with ground truth boxes (if available), the processed image with detections, and a table of detected objects (class, confidence, bounding box).
   - **Batch Results** (`batch_results.html`): Displays results for multiple images, with each image's original and processed versions and detection details.
   - **Video Results** (`video_results.html`): Shows the processed video with detected objects.

6. **Download Results**:
   - Download original, ground truth, or processed images from the results pages.


## Training Models

The `training_codes/` directory contains scripts for training the supported models:
- `faster_rcnn.py`: Train a Faster R-CNN model.
- `retinanet.py`: Train a RetinaNet model.
- `segmentation.py`: Train a YOLO segmentation model.
- `train_yolo_od.py`: Train a YOLO object detection model.

**Note**: The training scripts cannot be run directly as the dataset is not included in the project directory. These scripts are provided for reference and display purposes only. Actual training was performed and recorded using Kaggle Notebooks. Refer to the Kaggle Notebook documentation for details on the training process and dataset used.

## Supported Models

- **YOLO Object Detection** (`yolo_object_detection.pt`): Detects objects with bounding boxes.
- **YOLO Segmentation** (`yolo_segmentation.pt`): Detects objects and provides segmentation masks.
- **RetinaNet** (`retinanet_model.pth`): Detects objects with bounding boxes using a ResNet50 backbone.
- **Faster R-CNN** (`faster_rcnn_model.pth`): Detects objects with bounding boxes using a ResNet50 backbone.

## Supported Classes

- Bike (Orange)
- Bus (Purple)
- Car (Green)
- Pedestrian (Blue)
- Truck (Red)

## 🧪 Experimental Results

### 🔢 Overall Performance


| Metric          | YOLOv8 | Faster R-CNN | RetinaNet | YOLOv8-Seg |
|-----------------|--------|--------------|-----------|------------|
| Precision (%)   | 60.87  | 80.29        | 67.40     | 62.0       |
| Recall (%)      | 49.05  | 17.60        | 34.66     | 43.5       |
| mAP@0.5 (%)     | 20.24  | 11.70        | 19.15     | **52.9**   |
| mAP@0.75 (%)    | 16.19  | 9.36         | 15.32     | **30.2**   |
<<<<<<< HEAD
=======
| IoU@0.5 (%)     | 37.29  | 16.87        | 29.68     | **57.3**   |


### ✅ Key Findings
- **YOLOv8-Segmentation** significantly outperformed all other models.
- Instance segmentation is **more effective** for distorted fisheye objects.
- Standard detection models struggle with **edge distortion** and **small object detection** (e.g., pedestrians, trucks).

## 🧠 Insights

1. **Segmentation is Superior**: Pixel-level masks align better with distortion than bounding boxes.
2. **Distortion Matters**: Detection is more accurate in image centers than edges.
3. **YOLOv8** had the best balance between recall and precision among detection models.
4. **Faster R-CNN** had high precision but extremely low recall.
