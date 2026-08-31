"""
TerraPave Road Defect & Anomaly Detector
Leverages YOLO object detection with dynamic class mapping and temporal filtering.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class RoadDetector:
    # Standard Road Defect Classes
    DEFAULT_DEFECT_CLASSES = [
        "pothole",
        "longitudinal_crack",
        "transverse_crack",
        "alligator_crack",
        "debris",
        "manhole_subsidence",
    ]

    def __init__(self, model_path="models/best.pt", conf_threshold=0.35):
        """
        Initialize the TerraPave road anomaly detector.

        Args:
            model_path (str): Path to trained YOLO model weights.
            conf_threshold (float): Detection confidence threshold.
        """
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.model, self.is_custom_model = self._load_model(model_path)
        self.class_names = self._resolve_class_names()
        self.detection_history = []

    def _load_model(self, model_path):
        """Load YOLO model weights, falling back to base YOLOv8n if custom weights are missing."""
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                print(f"[TerraPave] Loaded custom road defect model from '{model_path}'")
                return model, True
            except Exception as e:
                print(f"[TerraPave] Warning: Failed to load '{model_path}' ({e}). Falling back to YOLOv8n.")

        fallback_path = "yolov8n.pt"
        print(f"[TerraPave] Custom weights not found at '{model_path}'. Using base '{fallback_path}'.")
        model = YOLO(fallback_path)
        return model, False

    def _resolve_class_names(self):
        """Resolve class names dynamically from model metadata."""
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            return [self.model.names[i] for i in sorted(self.model.names.keys())]
        return self.DEFAULT_DEFECT_CLASSES

    def preprocess_frame(self, frame):
        """Preprocess frame for YOLO inference."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or invalid.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect(self, frame):
        """
        Detect road anomalies and debris in a video frame.

        Args:
            frame (numpy.ndarray): Input video frame in BGR format.

        Returns:
            dict: Structured detection results including bounding boxes, quality score, and frame shape.
        """
        from .utils import calculate_road_quality_score

        processed_frame = self.preprocess_frame(frame)
        results = self.model(processed_frame, conf=self.conf_threshold, verbose=False)
        detections = self.parse_detections(results, frame.shape)
        road_quality = calculate_road_quality_score(detections)

        # Record history for temporal tracking
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        self.detection_history.append({
            "detections": detections,
            "road_quality": road_quality,
            "timestamp": timestamp,
        })

        # Maintain a sliding window of recent detections (30 frames)
        if len(self.detection_history) > 30:
            self.detection_history.pop(0)

        return {
            "detections": detections,
            "road_quality": road_quality,
            "frame_shape": frame.shape,
            "is_custom_model": self.is_custom_model,
        }

    def parse_detections(self, results, frame_shape):
        """Parse YOLO prediction tensor into structured detection dictionaries."""
        detections = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        if first_res.boxes is not None and len(first_res.boxes) > 0:
            boxes = first_res.boxes.xyxy.cpu().numpy()
            confidences = first_res.boxes.conf.cpu().numpy()
            class_ids = first_res.boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                cid = int(class_ids[i])
                if self.is_custom_model and cid < len(self.class_names):
                    cname = self.class_names[cid]
                elif not self.is_custom_model:
                    # For base COCO models, preserve actual COCO name (e.g. debris/vehicle)
                    coco_name = self.model.names.get(cid, "object") if hasattr(self.model, "names") else "object"
                    # Map obstacles/debris from COCO
                    cname = coco_name
                else:
                    cname = f"defect_class_{cid}"

                width = max(0.0, float(box[2] - box[0]))
                height = max(0.0, float(box[3] - box[1]))
                area = width * height

                detections.append({
                    "bbox": [float(x) for x in box],
                    "confidence": float(confidences[i]),
                    "class_id": cid,
                    "class_name": str(cname),
                    "area": area,
                })

        return detections

    def get_detection_summary(self, detections):
        """
        Summarize counts and aggregate severity for current detections.

        Args:
            detections (list): List of detection objects.

        Returns:
            tuple: (summary_dict, total_severity_score)
        """
        summary = {}
        severity_score = 0.0

        for detection in detections:
            cname = detection["class_name"].lower()
            conf = detection["confidence"]
            summary[cname] = summary.get(cname, 0) + 1

            if "pothole" in cname or "subsidence" in cname:
                severity_score += conf * 2.5
            elif "crack" in cname:
                severity_score += conf * 1.5
            elif "debris" in cname or "waste" in cname:
                severity_score += conf * 1.0
            else:
                severity_score += conf * 0.5

        return summary, float(severity_score)
