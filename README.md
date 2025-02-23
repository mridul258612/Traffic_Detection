# YOLOv5 Traffic Detection System

This project is a real-time traffic monitoring system using YOLOv5 for detecting multiple object classes, such as vehicles, pedestrians, , and lanes. The system is built with FastAPI for API support and includes WebSocket support for live streaming, ONNX export for optimization, and an evaluation script for performance metrics.

## Features
- *Real-time object detection* using YOLOv5
- *FastAPI-based API* for uploading and detecting objects
- *WebSocket support* for live video streaming
- *ONNX export* for optimized model inference
- *Evaluation metrics* including mAP, IoU, and FPS

## Installation and working

1. Clone the repository:
   bash
   git clone https://github.com/mridul258612/traffic-detection.git
   

2. Install dependencies:
   bash
      git clone https://github.com/ultralytics/yolov5.git  # Clone the YOLOv5 repository
      cd yolov5                                            # Change directory to yolov5
      python3 -m venv yolov5-venv                          # Create a virtual environment
      source yolov5-venv/bin/activate                      # Activate the virtual environment
      pip install --upgrade pip                            # Upgrade pip to the latest version
      pip install -r requirements.txt                      # Install dependencies required for YOLOv5
      cd ..
   
3. Download the BDD100K Dataset
   https://dl.cv.ethz.ch/bdd100k/data

   bdd100k_det_20_labels_trainval.zip
   bdd100k_drivable_labels_trainval.zip
   bdd100k_lane_labels_trainval.zip
   bdd100k_sem_seg_labels_trainval.zip
   100k_images_train.zip
   100k_images_val.zip
   100k_images_test.zip

   unzip all in data/
   this created a directory bdd100k in data folder and images and labels subfolder
   /data/bdd100k/images
   /data/bdd100k/labels

3. Prepare the Dataset for YOLO Format
   python prepare_data.py

4. Train the YOLOv5 Model using the BDD100K dataset.
   python3 train.py --data data.yaml --weights yolov5s.pt --epochs 1 --img 640 --batch 4

5. Run WebSocket Live Streaming
   python live_detection.py


## Directory Structure

traffic-detection/
│── data/                     # Dataset
│── live_detection.py         # WebSocket streaming
│── train.py                  # Model training script
│── prepare_data.py           # Prepare the Dataset for YOLO Format
│── README.md                 # Project documentation
yolov5/
│── requirements.txt          # Python dependencies


## Authors
- Mridul Joshi (mriduljoshi30@gmail.com)