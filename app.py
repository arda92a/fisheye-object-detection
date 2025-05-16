"""
Fisheye Camera Object Detection Web Application
----------------------------------------------
A Flask web application to visualize YOLO object detection and segmentation 
for fisheye camera images.
"""

import os
import uuid
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-key-for-dev')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 16  # 16 MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Define color scheme and class mappings
COLOR_SCHEME = {
    "Bike": (255, 140, 255),     # orange (BGR)
    "Bus": (128, 0, 128),        # purple (BGR)
    "Car": (0, 128, 0),          # green (BGR)
    "Pedestrian": (255, 0, 0),   # blue (BGR)
    "Truck": (0, 0, 255)         # red (BGR)
}

MODEL_CLASS_NAMES = {
    1: "Bike",
    2: "Bus",
    3: "Car",
    4: "Pedestrian",
    5: "Truck"
}

class_names_object_detection = {
    1: 'Bus',
    2: 'Bike',
    3: 'Car',
    4: 'Pedestrian',
    5: 'Truck'
}

class FisheyeDetector:
    """Handler for fisheye camera object detection with different models."""
    
    def __init__(self, model_path, model_type="yolo", threshold=0.5):
        """
        Initialize detector with model path, type and confidence threshold.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model ('yolo', 'retinanet', 'faster_rcnn', 'segmentation')
            threshold: Confidence threshold for detections
        """
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.model_path = model_path  # Store model path for async tasks
        
        try:
            if self.model_type == "yolo":
                self.model = YOLO(model_path)
            elif self.model_type == "retinanet":
                import torch
                from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
                from torchvision.models.detection.retinanet import RetinaNetHead
                from torchvision.ops.misc import FrozenBatchNorm2d
                
                self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
                backbone = self.model.backbone
                anchor_generator = self.model.anchor_generator
                num_classes = len(MODEL_CLASS_NAMES) + 1
                self.model.head = RetinaNetHead(
                    backbone.out_channels,
                    anchor_generator.num_anchors_per_location()[0],
                    num_classes,
                    norm_layer=FrozenBatchNorm2d
                )
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                
            elif self.model_type == "faster_rcnn":
                import torch
                import torchvision
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
                num_classes = len(MODEL_CLASS_NAMES) + 1
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
                
                checkpoint = torch.load(model_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                self.model.to(device)
                self.model.eval()
                
            elif self.model_type == "segmentation":
                self.model = YOLO(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")
            
    def process_image(self, image_path):
        """
        Process an image with the selected model.
        
        Args:
            image_path: Path to the image file or numpy array for video frames
            
        Returns:
            Tuple of (original image, processed image, detection metadata)
        """
        if isinstance(image_path, str):
            try:
                original_image = np.array(Image.open(image_path).convert("RGB"))
            except Exception as e:
                raise ValueError(f"Failed to open image: {str(e)}")
        else:
            original_image = image_path  # For video frames
            
        img_height, img_width = original_image.shape[:2]
        
        if self.model_type == "yolo" or self.model_type == "segmentation":
            task = "segment" if self.model_type == "segmentation" else "detect"
            results = self.model(image_path, task=task)[0]
            if task == "segment":
                boxes, masks, metadata = self._process_yolo_predictions(results, img_width, img_height, class_names=MODEL_CLASS_NAMES)
            else:
                boxes, masks, metadata = self._process_yolo_predictions(results, img_width, img_height, class_names=class_names_object_detection)
        elif self.model_type == "retinanet":
            import torch
            import torchvision.transforms as T
            transform = T.Compose([T.ToTensor()])
            input_tensor = transform(Image.fromarray(original_image)).unsqueeze(0)
            with torch.no_grad():
                results = self.model(input_tensor)[0]
            boxes, masks, metadata = self._process_retinanet_predictions(results, img_width, img_height)
        elif self.model_type == "faster_rcnn":
            import torch
            import torchvision.transforms.functional as F
            device = next(self.model.parameters()).device
            image_tensor = F.to_tensor(Image.fromarray(original_image)).to(device)
            with torch.no_grad():
                results = self.model([image_tensor])[0]
            boxes, masks, metadata = self._process_fasterrcnn_predictions(results, img_width, img_height)
        
        processed_image = self._visualize_predictions(original_image, boxes, masks)
        return original_image, processed_image, metadata
    
    def _process_yolo_predictions(self, results, img_width, img_height, class_names):
        """Process YOLO model predictions to extract bounding boxes and masks."""
        boxes = []
        masks = []
        metadata = {"detections": []}
        
        result_masks = results.masks.data.cpu().numpy() if results.masks is not None else []
        result_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        result_scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
        result_labels = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
        
        for i in range(len(result_labels)):
            if result_scores[i] < self.threshold:
                continue
                
            label_id = result_labels[i] + 1
            if label_id not in MODEL_CLASS_NAMES:
                continue
                
            class_name = class_names[label_id]
            color = COLOR_SCHEME.get(class_name)
            confidence = float(result_scores[i])
            
            x1, y1, x2, y2 = map(int, result_boxes[i])
            box = {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "class_id": int(label_id),
                "class_name": class_name,
                "confidence": confidence,
                "color": color
            }
            boxes.append(box)
            
            metadata["detections"].append({
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            if i < len(result_masks):
                mask = cv2.resize(
                    result_masks[i], 
                    (img_width, img_height), 
                    interpolation=cv2.INTER_NEAREST
                )
                masks.append(mask)
            else:
                masks.append(None)
                
        return boxes, masks, metadata
        
    def _process_retinanet_predictions(self, results, img_width, img_height):
        """Process RetinaNet model predictions."""
        boxes = []
        masks = []
        metadata = {"detections": []}
        
        result_boxes = results["boxes"].cpu().numpy()
        result_scores = results["scores"].cpu().numpy()
        result_labels = results["labels"].cpu().numpy()
        
        for i in range(len(result_labels)):
            if result_scores[i] < self.threshold:
                continue
                
            label_id = int(result_labels[i])
            if label_id not in MODEL_CLASS_NAMES:
                continue
                
            class_name = class_names_object_detection[label_id]
            color = COLOR_SCHEME.get(class_name, (0, 255, 255))
            confidence = float(result_scores[i])
            
            x1, y1, x2, y2 = map(int, result_boxes[i])
            box = {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "class_id": label_id,
                "class_name": class_name,
                "confidence": confidence,
                "color": color
            }
            boxes.append(box)
            
            metadata["detections"].append({
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            masks.append(None)
                
        return boxes, masks, metadata
        
    def _process_fasterrcnn_predictions(self, results, img_width, img_height):
        """Process Faster R-CNN model predictions."""
        boxes = []
        masks = []
        metadata = {"detections": []}
        
        result_boxes = results["boxes"].cpu().numpy()
        result_scores = results["scores"].cpu().numpy()
        result_labels = results["labels"].cpu().numpy()
        
        for i in range(len(result_labels)):
            if result_scores[i] < self.threshold:
                continue
                
            label_id = int(result_labels[i])
            if label_id not in MODEL_CLASS_NAMES:
                continue
                
            class_name = class_names_object_detection[label_id]
            color = COLOR_SCHEME.get(class_name, (0, 255, 255))
            confidence = float(result_scores[i])
            
            x1, y1, x2, y2 = map(int, result_boxes[i])
            box = {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "class_id": label_id,
                "class_name": class_name,
                "confidence": confidence,
                "color": color
            }
            boxes.append(box)
            
            metadata["detections"].append({
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            masks.append(None)
                
        return boxes, masks, metadata
    
    def _visualize_predictions(self, image, boxes, masks):
        """Visualize model predictions with segmentation masks or bounding boxes."""
        output_image = image.copy()
        
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            cv2.rectangle(
                output_image, 
                (box["xmin"], box["ymin"]), 
                (box["xmax"], box["ymax"]), 
                box["color"], 
                2
            )
            
            label_text = f"{box['class_name']} {box['confidence']:.2f}"
            cv2.putText(
                output_image, 
                label_text,
                (box["xmin"], box["ymin"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                box["color"], 
                2
            )
            
            if mask is not None:
                colored_mask = np.zeros_like(output_image)
                for c in range(3):
                    colored_mask[:, :, c] = (mask * box["color"][c]).astype(np.uint8)
                
                mask_indices = mask > 0
                alpha = 0.4
                output_image[mask_indices] = cv2.addWeighted(
                    output_image[mask_indices], 1-alpha, 
                    colored_mask[mask_indices], alpha, 0
                )
                
                contours, _ = cv2.findContours(
                    (mask > 0).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(output_image, contours, -1, box["color"], 2)
        
        return output_image

# Initialize detector
detector = None

def allowed_file(filename):
    """Check if file has an allowed extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_model_file(filename):
    """Check if model file has an allowed extension."""
    ALLOWED_MODEL_EXTENSIONS = {'pt', 'pth', 'weights'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS

def allowed_video_file(filename):
    """Check if video file has an allowed extension."""
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def image_to_base64(img):
    """Convert an image to base64 for HTML display."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

@app.route('/')
def index():
    """Render the home page."""
    model_loaded = detector is not None
    available_models = []
    if os.path.exists(app.config['MODEL_FOLDER']):
        available_models = [f for f in os.listdir(app.config['MODEL_FOLDER']) 
                           if allowed_model_file(f)]
    
    return render_template('index.html', 
                          model_loaded=model_loaded,
                          available_models=available_models)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Handle model upload and initialization."""
    if 'model_file' not in request.files:
        flash('No model file part')
        return redirect(request.url)
        
    model_file = request.files['model_file']
    
    if model_file.filename == '':
        flash('No selected model file')
        return redirect(request.url)
        
    if model_file and allowed_model_file(model_file.filename):
        filename = secure_filename(model_file.filename)
        model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
        model_file.save(model_path)
        
        try:
            model_type = request.form.get('model_type', 'yolo')
            threshold = float(request.form.get('threshold', 0.5))
            global detector
            detector = FisheyeDetector(model_path, model_type=model_type, threshold=threshold)
            flash(f'Model loaded successfully: {filename} (Type: {model_type})')
        except Exception as e:
            flash(f'Failed to load model: {str(e)}')
            
    return redirect(url_for('index'))

@app.route('/select_model', methods=['POST'])
def select_model():
    """Select and initialize a model from the available ones."""
    model_filename = request.form.get('model_filename')
    
    if not model_filename:
        flash('No model selected')
        return redirect(url_for('index'))
        
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
    
    if not os.path.exists(model_path):
        flash('Selected model file does not exist')
        return redirect(url_for('index'))
        
    try:
        model_type = request.form.get('model_type', 'yolo')
        threshold = float(request.form.get('threshold', 0.5))
        global detector
        detector = FisheyeDetector(model_path, model_type=model_type, threshold=threshold)
        flash(f'Model loaded successfully: {model_filename} (Type: {model_type})')
    except Exception as e:
        flash(f'Failed to load model: {str(e)}')
        
    return redirect(url_for('index'))

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle single or batch image uploads."""
    if detector is None:
        flash('Please load a model first')
        return redirect(url_for('index'))
        
    if 'image_file' not in request.files:
        flash('No image file part')
        return redirect(url_for('index'))
        
    image_files = request.files.getlist('image_file')
    results = []
    
    for image_file in image_files:
        if image_file.filename == '':
            continue
        if image_file and allowed_file(image_file.filename):
            original_filename = secure_filename(f"{uuid.uuid4()}_{image_file.filename}")
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            image_file.save(original_path)
            
            try:
                original_image, processed_image, metadata = detector.process_image(original_path)
                processed_filename = secure_filename(f"{uuid.uuid4()}_processed_{image_file.filename}")
                processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, processed_image)
                
                results.append({
                    'original_filename': original_filename,
                    'processed_filename': processed_filename,
                    'metadata': metadata
                })
            except Exception as e:
                flash(f'Error processing image {image_file.filename}: {str(e)}')
    
    if len(results) == 1:
        session['original_image_path'] = results[0]['original_filename']
        session['processed_image_path'] = results[0]['processed_filename']
        session['detection_metadata'] = results[0]['metadata']
        session['model_type'] = detector.model_type
        return redirect(url_for('results'))
    elif results:
        session['batch_results'] = results
        session['model_type'] = detector.model_type
        return redirect(url_for('batch_results'))
    
    flash('No valid images processed')
    return redirect(url_for('index'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and processing."""
    if detector is None:
        flash('Please load a model first')
        return redirect(url_for('index'))
        
    if 'video_file' not in request.files:
        flash('No video file part')
        return redirect(url_for('index'))
        
    video_file = request.files['video_file']
    
    if video_file.filename == '' or not allowed_video_file(video_file.filename):
        flash('Invalid video format. Please upload MP4 or AVI files.')
        return redirect(url_for('index'))
        
    video_filename = secure_filename(f"{uuid.uuid4()}_{video_file.filename}")
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video_file.save(video_path)
    
    try:
        output_filename = secure_filename(f"{uuid.uuid4()}_processed_{video_file.filename}")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        metadata = process_video(video_path, output_path, detector)
        
        session['video_result'] = {
            'original_filename': video_filename,
            'output_filename': output_filename,
            'metadata': metadata,
            'model_type': detector.model_type
        }
        return redirect(url_for('video_results'))
    except Exception as e:
        flash(f'Error processing video: {str(e)}')
        return redirect(url_for('index'))

def process_video(input_path, output_path, detector):
    """Process a video file frame by frame."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    metadata = {"detections": []}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, processed_frame, frame_metadata = detector.process_image(frame_rgb)
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        out.write(processed_frame_bgr)
        metadata["detections"].extend(frame_metadata["detections"])
    
    cap.release()
    out.release()
    return metadata

@app.route('/results')
def results():
    """Render single image results."""
    if 'original_image_path' not in session or 'processed_image_path' not in session:
        flash('No processed images to display')
        return redirect(url_for('index'))
        
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], session['original_image_path'])
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], session['processed_image_path'])
    metadata = session.get('detection_metadata', {"detections": []})
    model_type = session.get('model_type', 'unknown')
    
    original_b64 = image_to_base64(cv2.imread(original_path))
    processed_b64 = image_to_base64(cv2.imread(processed_path))
    
    return render_template('results.html', 
                          original_image=original_b64,
                          processed_image=processed_b64,
                          metadata=metadata,
                          model_type=model_type)

@app.route('/batch_results')
def batch_results():
    """Render batch image processing results."""
    if 'batch_results' not in session:
        flash('No batch results to display')
        return redirect(url_for('index'))
        
    # Preprocess images to base64
    batch_results = session['batch_results']
    for result in batch_results:
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], result['original_filename'])
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], result['processed_filename'])
        try:
            result['original_b64'] = image_to_base64(cv2.imread(original_path))
            result['processed_b64'] = image_to_base64(cv2.imread(processed_path))
        except Exception as e:
            flash(f'Error loading image {result["original_filename"]}: {str(e)}')
            result['original_b64'] = ''
            result['processed_b64'] = ''
    
    model_type = session.get('model_type', 'unknown')
    return render_template('batch_results.html', 
                          batch_results=batch_results,
                          model_type=model_type)

@app.route('/video_results')
def video_results():
    """Render video processing results."""
    if 'video_result' not in session:
        flash('No video results to display')
        return redirect(url_for('index'))
        
    return render_template('video_results.html', 
                          video_result=session['video_result'])

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for image detection."""
    if detector is None:
        return jsonify({
            "success": False,
            "error": "No model loaded"
        }), 400
        
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided"
        }), 400
        
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({
            "success": False,
            "error": "Empty image filename"
        }), 400
        
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(f"{uuid.uuid4()}_{image_file.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)
        
        try:
            _, _, metadata = detector.process_image(image_path)
            return jsonify({
                "success": True,
                "detections": metadata["detections"]
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    return jsonify({
        "success": False,
        "error": "Invalid image format"
    }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)