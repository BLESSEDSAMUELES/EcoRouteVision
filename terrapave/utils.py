"""
TerraPave Visualization, Quality Scoring, and Synthetic Testing Utilities
"""

from datetime import datetime
import cv2
import numpy as np


def calculate_road_quality_score(detections):
    """
    Calculate road quality score (0.0 to 1.0) based on detected defect severities.

    Args:
        detections (list): List of detection objects with class names and confidences.

    Returns:
        float: Road quality score (1.0 = pristine pavement, 0.0 = severe failure).
    """
    if not detections:
        return 1.0

    severity_weights = {
        "pothole": 0.45,
        "manhole_subsidence": 0.35,
        "alligator_crack": 0.30,
        "transverse_crack": 0.20,
        "longitudinal_crack": 0.15,
        "crack": 0.20,
        "debris": 0.15,
        "plastic_debris": 0.10,
        "rubber_debris": 0.10,
        "construction_waste": 0.15,
    }

    total_penalty = 0.0
    for detection in detections:
        cname = detection["class_name"].lower()
        conf = float(detection["confidence"])
        weight = 0.10
        for k, w in severity_weights.items():
            if k in cname:
                weight = w
                break

        # Area scale factor: larger defect boxes incur higher penalty
        area = detection.get("area", 1000.0)
        area_scale = min(1.5, max(0.8, (area / 10000.0) ** 0.5))
        total_penalty += weight * conf * area_scale

    # Quality score bounded in [0.0, 1.0]
    return float(np.clip(1.0 - total_penalty, 0.0, 1.0))


def draw_detections(frame, detections, road_quality):
    """
    Render defect bounding boxes, labels, and quality HUD onto the video frame.

    Args:
        frame (numpy.ndarray): Input video frame.
        detections (list): List of detection dictionaries.
        road_quality (float): Road quality score (0-1).

    Returns:
        numpy.ndarray: Annotated video frame.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Palette definition for distinct defect classes
    colors = {
        "pothole": (0, 0, 240),             # Vivid Red
        "manhole_subsidence": (0, 75, 255),  # Deep Orange
        "alligator_crack": (0, 140, 255),   # Bright Orange
        "transverse_crack": (0, 200, 255),  # Gold/Yellow
        "longitudinal_crack": (0, 225, 255),# Yellow
        "crack": (0, 180, 255),             # Amber
        "debris": (0, 220, 100),            # Emerald Green
        "plastic_debris": (100, 255, 100),  # Light Green
        "rubber_debris": (0, 255, 255),     # Yellow
        "construction_waste": (220, 100, 220), # Violet
    }

    for d in detections:
        cname = d["class_name"].lower()
        bbox = d["bbox"]
        conf = d["confidence"]

        # Match color
        color = (255, 255, 255)
        for k, c in colors.items():
            if k in cname:
                color = c
                break

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Draw bounding box with corner accents
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
        if corner_len > 3:
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 4)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 4)

        # Label tag
        label = f"{d['class_name']}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Top HUD Bar for Road Quality
    hud_color = (0, 220, 0) if road_quality >= 0.75 else (0, 180, 255) if road_quality >= 0.50 else (0, 0, 240)
    cv2.rectangle(annotated, (10, 10), (280, 52), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (280, 52), hud_color, 2)

    status_text = "PRISTINE" if road_quality >= 0.85 else "GOOD" if road_quality >= 0.70 else "MODERATE" if road_quality >= 0.50 else "DEGRADED"
    cv2.putText(annotated, f"TerraPave PQI: {road_quality:.2f} ({status_text})", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Timestamp & Telemetry Indicator
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, f"GPS LIVE: {timestamp}", (w - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


def create_sustainability_dashboard(analyzer, current_metrics):
    """
    Render a sleek dark-themed telemetry dashboard with real-time eco metrics.

    Args:
        analyzer (SustainabilityAnalyzer): Analyzer instance with cumulative metrics.
        current_metrics (dict): Current segment efficiency metrics.

    Returns:
        numpy.ndarray: Rendered dashboard image panel.
    """
    panel = np.zeros((320, 720, 3), dtype=np.uint8)
    panel[:] = (24, 28, 34)  # Sleek dark slate background

    # Header
    cv2.rectangle(panel, (0, 0), (720, 48), (35, 42, 52), -1)
    cv2.putText(panel, "TERRAPAVE: ECO-SURFACE INTELLIGENCE DASHBOARD", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 235, 200), 2, cv2.LINE_AA)

    # Divider
    cv2.line(panel, (360, 55), (360, 305), (50, 60, 75), 1)

    # Column 1: Live Trip Impact
    cv2.putText(panel, "LIVE SEGMENT TELEMETRY", (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)

    fuel_wasted = current_metrics.get("excess_fuel_wasted_liters", 0.0)
    co2_excess = current_metrics.get("excess_co2_kg", 0.0)
    quality = current_metrics.get("route_efficiency_score", 1.0)
    reroute = current_metrics.get("suggested_reroute", False)

    cv2.putText(panel, f"Fuel Consumption: {current_metrics.get('current_fuel_liters', 0.0):.2f} L", (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Excess Fuel Loss: {fuel_wasted:.3f} L", (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255) if fuel_wasted > 0 else (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Excess Carbon Emission: {co2_excess:.3f} kg CO2", (25, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 120, 255) if co2_excess > 0 else (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Pavement Quality: {quality:.2f} ({current_metrics.get('pavement_condition', 'N/A')})", (25, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    reroute_text = "ECO-REROUTE SUGGESTED" if reroute else "OPTIMAL ROUTE MAINTAINED"
    reroute_color = (0, 100, 255) if reroute else (0, 220, 100)
    cv2.rectangle(panel, (25, 235), (335, 275), (35, 42, 52), -1)
    cv2.rectangle(panel, (25, 235), (335, 275), reroute_color, 1)
    cv2.putText(panel, reroute_text, (35, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, reroute_color, 1, cv2.LINE_AA)

    # Column 2: Cumulative Fleet Impact
    cv2.putText(panel, "CUMULATIVE SUSTAINABILITY SAVINGS", (385, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)

    cumulative = analyzer.get_sustainability_metrics()
    cv2.putText(panel, f"Total Fuel Prevented: {cumulative['total_fuel_saved_liters']:.2f} L", (385, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Total CO2 Prevented: {cumulative['total_co2_reduced_kg']:.2f} kg", (385, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Equiv Trees Planted: {cumulative['trees_equivalent']:.1f} trees/yr", (385, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 150), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Fleet Avg Road Quality: {cumulative['average_route_quality']:.2f}", (385, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Telemetry Frames Analyzed: {cumulative['samples_analyzed']}", (385, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

    return panel


def generate_synthetic_road_frame(frame_index=0, width=640, height=480):
    """
    Generate a dynamic synthetic road frame with perspective lanes and defects for camera-free testing.

    Args:
        frame_index (int): Time index for motion animation.
        width (int): Frame width.
        height (int): Frame height.

    Returns:
        numpy.ndarray: Simulated road video frame.
    """
    frame = np.ones((height, width, 3), dtype=np.uint8)
    # Asphalt gradient
    for y in range(height):
        asphalt_val = int(55 + (y / height) * 35)
        frame[y, :] = (asphalt_val, asphalt_val, asphalt_val)

    # Road perspective lanes
    horizon_y = int(height * 0.35)
    center_x = width // 2

    # Road edges
    left_edge_top = (center_x - 60, horizon_y)
    left_edge_bot = (40, height)
    right_edge_top = (center_x + 60, horizon_y)
    right_edge_bot = (width - 40, height)

    cv2.line(frame, left_edge_top, left_edge_bot, (220, 220, 220), 4)
    cv2.line(frame, right_edge_top, right_edge_bot, (220, 220, 220), 4)

    # Dashed center line with motion offset
    offset = (frame_index * 12) % 60
    for y in range(horizon_y + offset, height, 60):
        t1 = (y - horizon_y) / max(1, (height - horizon_y))
        t2 = min(1.0, (y + 30 - horizon_y) / max(1, (height - horizon_y)))
        x1 = int(center_x)
        x2 = int(center_x)
        cv2.line(frame, (x1, y), (x2, min(height, y + int(30 * t2))), (0, 215, 255), max(1, int(4 * t1)))

    # Add simulated moving defects periodically
    cycle = frame_index % 120
    if 20 <= cycle <= 80:
        t = (cycle - 20) / 60.0
        defect_y = int(horizon_y + t * (height - horizon_y - 60))
        defect_x = int(center_x + (50 * t))
        radius = int(12 + t * 24)

        # Draw dark crater (simulated pothole)
        cv2.circle(frame, (defect_x, defect_y), radius, (25, 25, 25), -1)
        cv2.ellipse(frame, (defect_x, defect_y), (radius, max(4, radius // 2)), 0, 0, 360, (15, 15, 15), -1)
        # Crack tentacles
        cv2.line(frame, (defect_x - radius, defect_y), (defect_x - radius - 15, defect_y + 10), (20, 20, 20), 2)
        cv2.line(frame, (defect_x + radius, defect_y), (defect_x + radius + 20, defect_y - 8), (20, 20, 20), 2)

    return frame
