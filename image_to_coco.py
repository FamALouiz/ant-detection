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

# Label IDs of the dataset representing different categories
category_ids = {
    "Ant": 1,
}

BBOX_EXTENTION = 'txt'
ORIGINAL_EXTENTION = 'png'
image_id = 0
annotation_id = 0

def images_annotations_info(bboxpath: str, start_end_points: bool=True) -> tuple[int, int, int]:
    '''
        Process the image data and generate annotations information.

        Parameters:
            - maskpath (str): The path to the folder containing mask images.
            - start_end_points (bool): If True, the coordinates in the file should represent the start and end points of the ant box.
            
        Returns:
            - tuple[int, int, int]: The images, annotations, and annotation count.
    '''
    global image_id, annotation_id
    annotations = []
    images = []

    # Iterate through categories and corresponding masks
    for mask_image in glob.glob(os.path.join(bboxpath, f'*.{BBOX_EXTENTION}')):
        with open(mask_image, 'r') as file:
            
            lines = file.readlines()
        
            # Create annotation for each box
            for line in lines:

                # Processing the line 
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
                    
                    bbox = (x, y, w, h)
                else: 
                    bbox = (tuple(ant_box_coordinates))
                

                image_id = mask_image.split('/')[-1].split('.')[0]
    
                annotation = {
                    "iscrowd": 0,
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids['Ant'],
                    "bbox": bbox,
                }

                # Add annotation if area is greater than zero
                
                annotations.append(annotation)
                annotation_id += 1

    return images, annotations, annotation_id


def process_masks(mask_path, dest_json):
    global image_id, annotation_id
    image_id = 0
    annotation_id = 0

    # Initialize the COCO JSON format with categories
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [{"id": value, "name": key, "supercategory": key} for key, value in category_ids.items()],
        "annotations": [],
    }

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    # Save the COCO JSON to a file
    with open(dest_json, "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))

if __name__ == "__main__":
    train_bbox_path = "datasets\\dataset1\\data\\Train_data\\bboxes\\"
    train_json_path = "datasets\\dataset1\\data\\Train_data\\test.json"
    process_masks(train_bbox_path, train_json_path)

    test_mask_path = "datasets\\dataset1\\data\\Test_data\\bboxes\\"
    test_json_path = "datasets\\dataset1\\data\\Test_data\\test.json"
    process_masks(test_mask_path, test_json_path)