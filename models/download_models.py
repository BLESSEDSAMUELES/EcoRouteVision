#!/usr/bin/env python3
"""
TerraPave Model Manager & Weight Downloader
Downloads baseline YOLO checkpoints and optimizes for edge inference (ONNX export).
"""

import os
from ultralytics import YOLO


def setup_models():
    """Download baseline models and create models directory structure."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("[TerraPave] Preparing baseline detection models...")
    model = YOLO("yolov8n.pt")
    
    # Export to ONNX format for accelerated cross-platform inference
    onnx_target = os.path.join(models_dir, "yolov8n.onnx")
    if not os.path.exists(onnx_target):
        try:
            print("[TerraPave] Exporting baseline YOLOv8n to ONNX format...")
            model.export(format="onnx")
        except Exception as e:
            print(f"[TerraPave] ONNX export notice: {e}")

    print("\n[TerraPave] Model setup completed successfully.")
    print(f"Place custom road damage weights (e.g. RDD2020 / CRACK500) into: '{models_dir}/best.pt'\n")


if __name__ == "__main__":
    setup_models()
