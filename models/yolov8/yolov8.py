from ultralytics import YOLO
import yaml

class YOLOv8AntDetector:
    
    def __init__(self, data_path:str=None , dev:bool=True) -> None: 
        
        # Official model with pre-trained weights (not fine-tuned)
        self.model = YOLO('models\yolov8\yolov8.yaml')
        self.model = YOLO('models\yolov8\yolov8n.pt')
        self.dev = dev

        if dev:
            
            if not data_path: 
                raise Exception("The data file path is required in development mode.")
             
            self.data_path = data_path
            
        # Loading the number of classes from the data file
        if data_path is not None:
            try: 
                with open(data_path) as file:
                    self.num_classes = str(yaml.safe_load(file)['nc'])
            except FileNotFoundError:
                raise FileNotFoundError("The data file was not found.")
            
    def train(self, project_result_path:str, patience:int=0, epochs:int=200, batch_size:int=16, img_size:int=640) -> None:
        
        if not self.dev: 
            raise Exception("Training is only available in development mode.")
    
                    
        results = self.model.train(data=self.data_path, 
                                    project=project_result_path,
                                    epochs=epochs, 
                                    batch=batch_size, 
                                    patience=patience,
                                    imgsz=img_size)

if __name__ == "__main__":
    ant_detector = YOLOv8AntDetector(data_path='datasets\data\YOLOv8\data.yaml', dev=True)
    
    ant_detector.train(project_result_path='models\\yolov8\\results', 
                       patience=10, 
                       epochs=200, 
                       batch_size=16, 
                       img_size=640)

