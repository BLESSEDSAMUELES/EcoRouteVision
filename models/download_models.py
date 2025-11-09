#!/usr/bin/env python3
"""
Download pretrained models for EcoRouteVision
"""

from ultralytics import YOLO
import os


def download_models():
    """Download required models"""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    print("Downloading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    model.export(format='onnx')  # Export to ONNX for potential optimization

    print("Models downloaded successfully!")

    # Note: For custom trained models, you would add them here
    # For MVP, we use the pretrained YOLOv8n as a starting point


if __name__ == "__main__":
    download_models()
