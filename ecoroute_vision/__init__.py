"""
Legacy alias for TerraPave.
Deprecated: Use `terrapave` instead.
"""

import warnings
warnings.warn(
    "The 'ecoroute_vision' module is deprecated and has been renamed to 'terrapave'. "
    "Please update your imports to 'from terrapave import ...'",
    DeprecationWarning,
    stacklevel=2
)

from terrapave import (
    RoadDetector,
    SustainabilityAnalyzer,
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
