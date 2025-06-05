from ultralytics import YOLO

model = YOLO('yolov8l-seg.pt') 


results = model.train(
    data='yolo_segment/dataset.yaml',  
    epochs=50,                         
    imgsz=640,                         
    batch=16,                          
    patience=10,                        
    device=0,                           
    project='segmentation_project'
)