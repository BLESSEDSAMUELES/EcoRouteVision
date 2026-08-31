"""
Unit tests for TerraPave utility and scoring functions
"""

import unittest
import numpy as np
from terrapave import (
    calculate_road_quality_score,
    draw_detections,
    create_sustainability_dashboard,
    generate_synthetic_road_frame,
    SustainabilityAnalyzer,
)


class TestUtils(unittest.TestCase):
    def test_calculate_road_quality_pristine(self):
        score = calculate_road_quality_score([])
        self.assertEqual(score, 1.0)

    def test_calculate_road_quality_with_potholes(self):
        detections = [
            {"class_name": "pothole", "confidence": 0.9, "area": 5000.0},
            {"class_name": "alligator_crack", "confidence": 0.8, "area": 4000.0},
        ]
        score = calculate_road_quality_score(detections)
        self.assertLess(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_draw_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.85, "class_name": "pothole", "area": 10000},
        ]
        annotated = draw_detections(frame, detections, 0.65)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertTrue((annotated != 0).any())

    def test_dashboard_generation(self):
        analyzer = SustainabilityAnalyzer()
        metrics = analyzer.calculate_route_efficiency(0.75, 10.0)
        dashboard = create_sustainability_dashboard(analyzer, metrics)
        self.assertEqual(len(dashboard.shape), 3)
        self.assertEqual(dashboard.shape[2], 3)

    def test_synthetic_frame_generation(self):
        frame = generate_synthetic_road_frame(frame_index=15, width=640, height=480)
        self.assertEqual(frame.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
