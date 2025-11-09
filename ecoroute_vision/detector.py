import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
from .utils import calculate_road_quality_score, draw_detections


class RoadDetector:
    def __init__(self, model_path='models/best.pt', conf_threshold=0.5):
        """
        Initialize the road anomaly detector

        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.model = self.load_model(model_path)
        self.class_names = ['pothole', 'crack', 'plastic_debris',
                            'rubber_debris', 'construction_waste']
        self.detection_history = []

    def load_model(self, model_path):
        """Load YOLO model, download if not exists"""
        try:
            model = YOLO(model_path)
            return model
        except:
            print("Model not found, using pretrained YOLOv8n")
            return YOLO('yolov8n.pt')

    def preprocess_frame(self, frame):
        """Preprocess frame for detection"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def detect(self, frame):
        """
        Detect road anomalies and debris in frame

        Args:
            frame: Input frame (numpy array)

        Returns:
            dict: Detection results with bounding boxes and scores
        """
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)

        # Run inference
        results = self.model(
            processed_frame, conf=self.conf_threshold, verbose=False)

        # Parse results
        detections = self.parse_detections(results, frame.shape)

        # Calculate road quality score
        road_quality = calculate_road_quality_score(detections)

        # Store in history for temporal analysis
        self.detection_history.append({
            'detections': detections,
            'road_quality': road_quality,
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        })

        # Keep only recent history (last 30 detections)
        if len(self.detection_history) > 30:
            self.detection_history.pop(0)

        return {
            'detections': detections,
            'road_quality': road_quality,
            'frame_shape': frame.shape
        }

    def parse_detections(self, results, frame_shape):
        """Parse YOLO results into structured format"""
        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else 'unknown',
                    'area': (box[2] - box[0]) * (box[3] - box[1])
                }
                detections.append(detection)

        return detections

    def get_detection_summary(self, detections):
        """Get summary of current detections"""
        summary = {class_name: 0 for class_name in self.class_names}
        severity_score = 0

        for detection in detections:
            class_name = detection['class_name']
            if class_name in summary:
                summary[class_name] += 1

            # Calculate severity based on class and confidence
            if class_name == 'pothole':
                severity_score += detection['confidence'] * 2
            elif class_name == 'crack':
                severity_score += detection['confidence'] * 1.5
            else:
                severity_score += detection['confidence'] * 1

        return summary, severity_score
