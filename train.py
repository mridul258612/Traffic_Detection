import torch
from yolov5 import train

def train_model():
    train.run(
        data="data.yaml",  # Path to dataset configuration
        weights="yolov5s.pt",  # Pre-trained YOLOv5 model
        epochs=50,
        batch_size=16,
        img_size=640
    )

if __name__ == "__main__":
    train_model()