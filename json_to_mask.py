import json
import numpy as np
import cv2
import os

input_folder = "annotations"  
output_folder = "masks"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".json"):
        json_path = os.path.join(input_folder, file)

        with open(json_path) as f:
            data = json.load(f)

        h = data["imageHeight"]
        w = data["imageWidth"]

        mask = np.zeros((h, w), dtype=np.uint8)

        for shape in data["shapes"]:
            points = np.array(shape["points"], dtype=np.int32)

       
            cv2.fillPoly(mask, [points], 255)

       
        name = file.replace(".json", ".jpg")
        cv2.imwrite(os.path.join(output_folder, name), mask)

print(" Masks created successfully!")