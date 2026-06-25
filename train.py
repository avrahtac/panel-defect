from ultralytics import YOLO
import torch
import multiprocessing

def main():
    if torch.cuda.is_available():
        device_target = 0
        print("CUDA GPU detected. Using GPU for training.")
    else:
        device_target = 'cpu'
        print("No GPU detected. Using CPU for training.")

    model = YOLO('yolov8n.pt') 

    print("Starting training sequence...")
    model.train(
        data='data.yaml',
        epochs=300,
        patience=50,
        imgsz=640,
        batch=16,
        device=device_target,
        workers=0,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )

    print("Training complete. Weights saved to runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()