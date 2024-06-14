from ultralytics import YOLO
from experiment_controlller import start_new_experiment
import yaml
import mlflow as mlf

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
        
        # Initializing MLFlow experiment if in development mode
        if dev:
            self.experiment_id = start_new_experiment('ant-detection-yolov8')
            
        # Loading the number of classes from the data file
        if data_path is not None:
            try: 
                with open(data_path) as file:
                    self.num_classes = str(yaml.safe_load(file)['nc'])
            except FileNotFoundError:
                raise FileNotFoundError("The data file was not found.")
            
    def train(self, project_result_path:str, run_name:str, patience:int=0, description:str=None, epochs:int=200, batch_size:int=16, img_size:int=640) -> None:
        
        if not self.dev: 
            raise Exception("Training is only available in development mode.")
        
        # Starting a new run in MLFlow
        with mlf.start_run(experiment_id=self.experiment_id, 
                           run_name=run_name, log_system_metrics=True, description=description) as run:
            
            mlf.autolog()
            
            results = self.model.train(data=self.data_path, 
                                       project=project_result_path,
                                       epochs=epochs, 
                                       batch=batch_size, 
                                       patience=patience,
                                       imgsz=img_size)
    


if __name__ == "__main__":
    ant_detector = YOLOv8AntDetector(data_path='data\YOLOv8\data.yaml', dev=True)
    
    ant_detector.train(project_result_path='models\\yolov8\\results', 
                       run_name='yolov8-ant-detector-test-run-2', 
                       patience=10, 
                       description='Training the YOLOv8 model for ant detection. Test run', 
                       epochs=200, 
                       batch_size=16, 
                       img_size=640)

