import os
import shutil

# Define dataset paths
bdd100k_path = "data/bdd100k"
yolo_dataset_path = "data/yolo_format"

# Ensure YOLO dataset directory exists
os.makedirs(yolo_dataset_path, exist_ok=True)

# Function to copy and rename files in YOLO format
def convert_to_yolo():
    for split in ["train", "val"]:
        img_dir = os.path.join(bdd100k_path, "images", "100k", split)
        lbl_dir = os.path.join(bdd100k_path, "labels", split)

        os.makedirs(os.path.join(yolo_dataset_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset_path, split, "labels"), exist_ok=True)

        for img_file in os.listdir(img_dir):
            shutil.copy(
                os.path.join(img_dir, img_file),
                os.path.join(yolo_dataset_path, split, "images", img_file)
            )
    
    print("Dataset successfully converted to YOLO format!")

if __name__ == "__main__":
    convert_to_yolo()