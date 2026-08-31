"""
TerraPave - AI-powered road surface assessment, defect intelligence, and eco-routing engine.
"""

__version__ = "0.1.0"
__author__ = "TerraPave Contributors"

from .detector import RoadDetector
from .analyzer import SustainabilityAnalyzer
from .utils import (
    calculate_road_quality_score,
    draw_detections,
    create_sustainability_dashboard,
    generate_synthetic_road_frame,
)

__all__ = [
    "RoadDetector",
    "SustainabilityAnalyzer",
    "calculate_road_quality_score",
    "draw_detections",
    "create_sustainability_dashboard",
    "generate_synthetic_road_frame",
]
