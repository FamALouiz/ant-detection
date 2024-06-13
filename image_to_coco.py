"""
This code automates the conversion of binary masks representing different 
object categories into the COCO (Common Objects in Context) JSON format. 

The code is based on the following folder structure for training and validation
images and masks. You need to change the code based on your folder structure 
or organize your data to the format below.

For each binary mask, the code extracts contours using OpenCV. 
These contours represent the boundaries of objects within the images.This is a key
step in converting binary masks to polygon-like annotations. 

Convert the contours into annotations, including 
bounding boxes, area, and segmentation information. Each annotation is 
associated with an image ID, category ID, and other properties required by the COCO format.

The code also creates an images section containing 
metadata about the images, such as their filenames, widths, and heights.
In my example, I have used exactly the same file names for all images and masks
so that a given mask can be easily mapped to the image. 

All the annotations, images, and categories are 
assembled into a dictionary that follows the COCO JSON format. 
This includes sections for "info," "licenses," "images," "categories," and "annotations."

Finally, the assembled COCO JSON data is saved to a file, 
making it ready to be used with tools and frameworks that support the COCO data format.


"""

import glob
import json
import os
import cv2

# Label IDs of the dataset representing different categories
category_ids = {
    "Ant": 1,
}

BBOX_EXTENTION = 'txt'
KEYPOINTS_EXTENTION = 'txt'
IMAGE_EXTENTION = 'png'
image_id = 0
annotation_id = 0

def images_annotations_info(keypoints_path: str, bbox_path: str, images_path: str, start_end_points: bool=True) -> tuple[int, int, int]:
    '''
        Process the image data and generate annotations information.

        Parameters:
            - keypoints (str): The path to the keypoints folder.
            - maskpath (str): The path to the folder containing mask images.
            - start_end_points (bool): If True, the coordinates in the file should represent the start and end points of the ant box.
            
        Returns:
            - tuple[int, int, int]: The images, annotations, and annotation count.
    '''
    global image_id, annotation_id
    annotations = []
    images = []

    # Appending images to the images list
    for image_file in glob.glob(os.path.join(images_path, f'*.{IMAGE_EXTENTION}')):
        image_id += 1
        
        read_image = cv2.imread(image_file)
        image = {
            "id": image_id,
            "file_name": image_file.split('/')[-1],
            "width": read_image.shape[1],
            "height": read_image.shape[0],
        }
        
        images.append(image)
    
    # Iterate through categories and corresponding masks
    for bbox_file, keypoints_file in zip(glob.glob(os.path.join(bbox_path, f'*.{BBOX_EXTENTION}')), glob.glob(os.path.join(keypoints_path, f'*.{KEYPOINTS_EXTENTION}'))):
        
        bboxes = []
        total_keypoints = []
        
        # Loading bbox
        with open(bbox_file, 'r') as file:
            
            lines = file.readlines()    
                    
            # Create annotation for each box
            for line in lines:

                # Processing the line 
                try: 
                    bbox_coordinates = list(map(int, line.split()))
                except ValueError: 
                    
                    # ValueError: If the coordinates are not integers
                    print("Error in reading the ant box coordinates from the file... not an integer")
                    continue
                
                if start_end_points:
                    
                    # Calulate the width and height from the bbox coordinates
                    x = min(bbox_coordinates[0], bbox_coordinates[2])
                    y = min(bbox_coordinates[1], bbox_coordinates[3])
                    w = abs(bbox_coordinates[2] - bbox_coordinates[0])
                    h = abs(bbox_coordinates[3] - bbox_coordinates[1])
                    
                    bbox = (x, y, w, h)
                else: 
                    bbox = (tuple(bbox_coordinates))
                
                bboxes.append(bbox)

    
        # Loading keypoints
        with open(keypoints_file, 'r') as file: 
            
            lines = file.readlines()
            
            for line in lines: 
                
                # Processing the line 
                try: 
                    keypoints_coordinates = list(map(int, line.split()))
                except ValueError: 
                    
                    # ValueError: If the coordinates are not integers
                    print("Error in reading the ant box coordinates from the file... not an integer")
                    continue
                
                if start_end_points:
                    
                    # Calulate the width and height from the bbox coordinates
                    x = min(keypoints_coordinates[0], keypoints_coordinates[2])
                    y = min(keypoints_coordinates[1], keypoints_coordinates[3])
                    w = abs(keypoints_coordinates[2] - keypoints_coordinates[0])
                    h = abs(keypoints_coordinates[3] - keypoints_coordinates[1])
                    
                    keypoints = (x, y, w, h)
                else: 
                    keypoints = (tuple(keypoints_coordinates))
                    
                total_keypoints.append(keypoints)
        
        image_id = bbox_file.split('/')[-1].split('.')[0]
        
        for bbox, keypoints in zip(bboxes, total_keypoints):
                    
            annotation = {
                "iscrowd": 0,
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids['Ant'],
                "bbox": bbox,
                "keypoints": keypoints,
            }

            annotations.append(annotation)
            annotation_id += 1

    return images, annotations, annotation_id


def process_masks(bbox_path: str, keypoints_path: str, image_path:str , dest_json: str) -> None:
    global image_id, annotation_id
    image_id = 0
    annotation_id = 0

    info = {
        "description": "Ant Detection Dataset",
        "url": "https://www.kaggle.com/datasets/elizamoscovskaya/ant-2-keypoints-dataset",
        "version": "1.0",
        "year": 2023,
        "contributor": "Eliza Moscovskaya",
    }
    
    # Initialize the COCO JSON format with categories
    coco_format = {
        "info": info,
        "images": [],
        "categories": [{"id": value, "name": key, "supercategory": key} for key, value in category_ids.items()],
        "annotations": [],
    }

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(keypoints_path, bbox_path, image_path)

    # Save the COCO JSON to a file
    with open(dest_json, "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, bbox_path))

if __name__ == "__main__":
    train_bbox_path = "data\\Train_data\\bboxes\\"
    train_json_path = "data\\Train_data\\images\\train.json"
    train_keypoints_path = "data\\Train_data\\keypoints\\"
    train_image_path = "data\\Train_data\\images\\"
    process_masks(train_bbox_path, train_keypoints_path, train_image_path, train_json_path)

    test_mask_path = "data\\Test_data\\bboxes\\"
    test_json_path = "data\\Test_data\\images\\test.json"
    test_keypoints_path = "data\\Test_data\\keypoints\\"
    test_image_path = "data\\Test_data\\images\\"
    process_masks(test_mask_path, test_keypoints_path, test_image_path, test_json_path)