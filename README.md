# PV Panel Defect Classification System

## Abstract

This project implements a high-performance deep learning pipeline for the automated, real-time classification of defects in photovoltaic (PV) panels. Transitioning from traditional heavy-weight image classifiers, the system utilizes an ultra-lightweight **YOLOv8 Nano Convolutional Neural Network (CNN)** optimized for edge computing. The architecture is bifurcated into a high-speed training module, a desktop evaluation interface with dynamic masking, and a live-inference deployment script engineered for drone hardware, providing an end-to-end robust solution for aerial defect analysis.

## Methodology

The core methodology employs **transfer learning** using the **ResNet-18** architecture, pre-trained on the ImageNet dataset. The model's final fully-connected layer is re-trained on a custom dataset of PV panel images to classify them into specific defect categories.
The system's architecture is segmented into two primary components:

### 1. Training Pipeline (`train.py`)
This module orchestrates the model's learning phase utilizing the PyTorch engine. It is optimized for maximum GPU utilization (e.g., NVIDIA RTX series) using dynamic batching, data caching, and multi-threaded CPU workers. Features include:
* **Automated Hardware Routing:** Dynamically shifts tensor operations to CUDA cores if available, or falls back to standard CPU execution.
* **Early Stopping:** Implements patience-based monitoring to halt training precisely at peak validation accuracy, preventing network overfitting.

### 2. Inference Application (`app.py`)
A user-facing graphical interface built with Tkinter, designed for localized testing and validation with following features mentioned below:
* **Dynamic ROI Masking:** Integrates an advanced OpenCV Canny-edge and Contour detection pipeline. It actively identifies the geometric boundaries of the physical solar panel in the camera feed, crops the specific Region of Interest (ROI), and overlays a solid protective background (Dummy Color) over the rest of the room to prevent environmental false positives during testing.
* **Confidence Metrics:** Returns the predicted defect class alongside its Softmax confidence percentage.

### 3. Drone Intference Script (`drone.py`)
This script provides a practical implementation of trained object detection model using Yolov8 Classification model on an drone equipped with IMX219 Camera. This script helps to initilaise the practical approach on actual physical drone where the minimal interface returns predicited class i.e, defect type and corresponding confidence score derived from software output. The script provides intensive burst oriented image supply with accuracy of 300 images per second providing a smooth frame rate reducing CPU Stress.

## Training Results

### Model Accuracy
![Training and Validation Accuracy](./images/image.png)
