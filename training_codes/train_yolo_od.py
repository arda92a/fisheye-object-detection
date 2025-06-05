from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data='data/data.yaml',  
    epochs=50,                        
    imgsz=640,                        
    batch=16,                        
    device=0,                     
    project='results/yolov8l_results',  
    name='yolov8l_fisheye_model',  
    exist_ok=True,
    degrees = 180,
    optimizer='Adam',
    lr0=0.01,
    momentum=0.937,
    workers=4,
    verbose=True,
    save=True,
    conf = 0.7
)