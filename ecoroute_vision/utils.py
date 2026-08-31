"""Legacy alias: use terrapave.utils instead."""
from terrapave.utils import (
    calculate_road_quality_score,
    draw_detections,
    create_sustainability_dashboard,
    generate_synthetic_road_frame,
)

__all__ = [
    "calculate_road_quality_score",
    "draw_detections",
    "create_sustainability_dashboard",
    "generate_synthetic_road_frame",
]
