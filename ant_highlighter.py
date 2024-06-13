import cv2
import numpy as np
import sys
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

    def highlist_ant_from_file_path(self, image_path: str, ant_box_image_path: str, start_end_points: bool=True)-> np.ndarray:
        '''
        
            Highlight the ant in the image by drawing a rectangle around it. Each line in the file should represent an ant in the image.
            
            Parameters:
                - image_path (str): The path to the image where the ant is to be highlighted.
                - ant_box_image_path (str): The path to the file containing the coordinates of the ant box.
                - start_end_point (bool): If True, the coordinates in the file should represent the start and end points of the ant box. 
                    If False, the coordinates should represent the start point and the width and height of the ant box.
            
            Returns:
                - np.ndarray: The image with the ant highlighted. 
        '''
        
        image = cv2.imread(image_path)
        
        ant_boxes = []
        
        # Read the ant box coordinates from the file
        with open(ant_box_image_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                try: 
                    ant_box_coordinates = list(map(int, line.split()))
                except ValueError: 
                    
                    # ValueError: If the coordinates are not integers
                    print("Error in reading the ant box coordinates from the file... not an integer")
                    continue
                
                if start_end_points:
                    
                    # Calulate the width and height from the bbox coordinates
                    x = ant_box_coordinates[0]
                    y = ant_box_coordinates[1]
                    w = ant_box_coordinates[2] - ant_box_coordinates[0]
                    h = ant_box_coordinates[3] - ant_box_coordinates[1]
                    
                    ant_boxes.append((x, y, w, h))
                else: 
                    ant_boxes.append(tuple(ant_box_coordinates))
                            
        return self.highlight_ant(image, ant_boxes)
    
    
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python ant_highlighter.py <image_number>")
        sys.exit(0)
        
    if sys.argv[1] == '-h':
        print("Usage: python ant_highlighter.py <image_number>")
        sys.exit(0)
            
    image_number = sys.argv[1]
    
    ant_highlighter = AntHighlighter(color=(0, 255, 0), width=2)
    image_path = r'datasets\dataset1\data\Train_data\images\image' + str(image_number) + '.png'
    ant_box_image_path = r'datasets\dataset1\data\Train_data\bboxes\bbox' + str(image_number) + '.txt'
    
    try: 
        image = ant_highlighter.highlist_ant_from_file_path(image_path, ant_box_image_path)
    except FileNotFoundError:
        print("Error in reading the image or the ant box coordinates from the file... file not found")
        sys.exit(0)
    
    resized_image = cv2.resize(image, (1000, 500))
    cv2.imshow('Ant Highlighted Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()