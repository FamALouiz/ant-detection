import cv2
import numpy as np

class AntHighlighter:
    '''
        A class to highlight the ant in an image by drawing a rectangle around it.
        
        Attributes:
            - color (tuple): The color of the rectangle to be drawn around the ant.
            - width (int): The width of the rectangle to be drawn around the ant.
        
        Methods:
            - highlight_ant(image: np.ndarray, ant_boxes: list) -> np.ndarray: Highlight the ant in the image by drawing a rectangle around it.
            - highlist_ant_from_file_path(image_path: str, ant_box_image_path: str) -> np.ndarray: Highlight the ant in the image by drawing a rectangle around it.
            
        Example:
            ant_highlighter = AntHighlighter(color=(0, 255, 255), width=2)\n
            image_path = r'datasets\dataset1\data\Train_data\images\image0.png'\n
            ant_box_image_path = r'datasets\dataset1\data\Train_data\bboxes\bbox0.txt'\n
            image = ant_highlighter.highlist_ant_from_file_path(image_path, ant_box_image_path)\n
            cv2.imshow('Ant Highlighted Image', image)\n
            cv2.waitKey(0)\n
            cv2.destroyAllWindows()\n
    '''
    
    def __init__(self, color=(0, 255, 0), width=2) -> None: 
        self.color = color
        self.width = width

    def highlight_ant(self, image: np.ndarray, ant_boxes: list) -> np.ndarray: 
        '''
            Highlight the ant in the image by drawing a rectangle around it.
            
            Parameters:
                - image (np.ndarray): The image where the ant is to be highlighted.
                - ant_boxes (list): A list of tuples, where each tuple contains the coordinates of the ant box.
                
            Returns:
                - np.ndarray: The image with the ant highlighted.
        '''
        for ant_box in ant_boxes:
            # Extract the coordinates of the ant box
            x, y, w, h = ant_box

            # Draw a rectangle around the ant
            image = cv2.rectangle(image, (x, y), (x + w, y + h), self.color, self.width)

        return image

    def highlist_ant_from_file_path(self, image_path: str, ant_box_image_path: str)-> np.ndarray:
        '''
        
            Highlight the ant in the image by drawing a rectangle around it.
            
            Parameters:
                - image_path (str): The path to the image where the ant is to be highlighted.
                - ant_box_image_path (str): The path to the file containing the coordinates of the ant box.
            
            Returns:
                - np.ndarray: The image with the ant highlighted. 
        '''
        
        image = cv2.imread(image_path)
        
        ant_boxes = []
        
        # Read the ant box coordinates from the file
        with open(ant_box_image_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines: 
                ant_box = tuple(map(int, line.split()))
                ant_boxes.append(ant_box)
            
        return self.highlight_ant(image, ant_boxes)
    
    
if __name__ == "__main__":
    ant_highlighter = AntHighlighter(color=(0, 255, 255), width=2)
    image_path = r'datasets\dataset1\data\Train_data\images\image0.png'
    ant_box_image_path = r'datasets\dataset1\data\Train_data\bboxes\bbox0.txt'
    image = ant_highlighter.highlist_ant_from_file_path(image_path, ant_box_image_path)
    cv2.imshow('Ant Highlighted Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()