import os
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}") 
    # LƯU Ý NẾU DÙNG YOLOV5 THÌ PHẢI TẮT INTERNET KHI CHẠY NẾU KHÔNG THÌ NO TỰ DỘNG TẢI MỚI MÔ HÌNH YOLO11
    model = YOLO("yolov5su.pt") 
    results = model.train(
        data="custom_data.yaml",
        epochs=100,
        imgsz=640,
        device=device,
        batch=2,
        workers=0,
        cache=True,
        amp=True,
        optimizer='AdamW',
    )

    print("Training completed.")