from ultralytics import YOLO
from experiment_controlller import start_new_experiment

class YOLOv8AntDetector:
    
    def __init__(self, dev=True) -> None: 
        
        # Official model
        self.model = YOLO('models\yolov8\yolov8.yaml')
        self.model = YOLO('models\yolov8\yolov8n.pt')

        # Initializing MLFlow experiment if in development mode
        if dev:
            self.experiment_id = start_new_experiment('ant-detection-yolov8')
        


