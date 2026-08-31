"""
Unit tests for TerraPave RoadDetector
"""

import unittest
import numpy as np
from terrapave import RoadDetector


class TestRoadDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize detector with default model (fallback to yolov8n.pt if best.pt absent)
        cls.detector = RoadDetector()

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsInstance(self.detector.class_names, list)
        self.assertGreater(len(self.detector.class_names), 0)

    def test_preprocess_frame(self):
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_rgb = self.detector.preprocess_frame(frame_bgr)
        self.assertEqual(frame_rgb.shape, (480, 640, 3))

    def test_detect_blank_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.detector.detect(frame)
        self.assertIn("detections", res)
        self.assertIn("road_quality", res)
        self.assertIn("frame_shape", res)
        self.assertEqual(res["road_quality"], 1.0)
        self.assertEqual(len(res["detections"]), 0)

    def test_detection_summary(self):
        mock_detections = [
            {"class_name": "pothole", "confidence": 0.9},
            {"class_name": "crack", "confidence": 0.8},
            {"class_name": "debris", "confidence": 0.7},
        ]
        summary, severity = self.detector.get_detection_summary(mock_detections)
        self.assertEqual(summary.get("pothole"), 1)
        self.assertEqual(summary.get("crack"), 1)
        self.assertEqual(summary.get("debris"), 1)
        self.assertGreater(severity, 0.0)


if __name__ == "__main__":
    unittest.main()
