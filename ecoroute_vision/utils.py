import cv2
import numpy as np
from datetime import datetime


def calculate_road_quality_score(detections):
    """
    Calculate overall road quality score based on detections

    Args:
        detections: List of detection objects

    Returns:
        float: Road quality score (0-1, 1=best)
    """
    if not detections:
        return 1.0  # Perfect quality if no issues detected

    severity_weights = {
        'pothole': 0.4,
        'crack': 0.3,
        'plastic_debris': 0.1,
        'rubber_debris': 0.1,
        'construction_waste': 0.1
    }

    total_severity = 0
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        weight = severity_weights.get(class_name, 0.1)
        total_severity += weight * confidence

    # Normalize to 0-1 scale (higher = better)
    quality_score = max(0, 1 - min(total_severity, 1))
    return quality_score


def draw_detections(frame, detections, road_quality):
    """
    Draw detection bounding boxes and information on frame

    Args:
        frame: Input frame
        detections: List of detection objects
        road_quality: Road quality score

    Returns:
        numpy.ndarray: Annotated frame
    """
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]

    # Color scheme for different classes
    colors = {
        'pothole': (0, 0, 255),      # Red
        'crack': (0, 165, 255),      # Orange
        'plastic_debris': (0, 255, 0),  # Green
        'rubber_debris': (255, 255, 0),  # Yellow
        'construction_waste': (128, 0, 128)  # Purple
    }

    # Draw each detection
    for detection in detections:
        class_name = detection['class_name']
        bbox = detection['bbox']
        confidence = detection['confidence']

        color = colors.get(class_name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw road quality indicator
    quality_color = (0, 255, 0) if road_quality > 0.7 else (
        0, 165, 255) if road_quality > 0.4 else (0, 0, 255)
    cv2.rectangle(annotated_frame, (10, 10), (150, 40), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Road Quality: {road_quality:.2f}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

    # Draw timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, (width - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_frame


def create_sustainability_dashboard(analyzer, current_metrics):
    """
    Create a visual sustainability dashboard

    Args:
        analyzer: SustainabilityAnalyzer instance
        current_metrics: Current route efficiency metrics

    Returns:
        numpy.ndarray: Dashboard image
    """
    dashboard = np.ones((300, 600, 3), dtype=np.uint8) * \
        50  # Dark gray background

    # Title
    cv2.putText(dashboard, "EcoRouteVision - Sustainability Dashboard",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Current trip metrics
    y_offset = 70
    metrics = [
        f"Current Fuel Saving: {current_metrics.get('fuel_saving_liters', 0):.2f} L",
        f"CO2 Reduction: {current_metrics.get('co2_reduction_kg', 0):.2f} kg",
        f"Route Efficiency: {current_metrics.get('route_efficiency_score', 0):.2f}",
        f"Reroute Suggested: {'Yes' if current_metrics.get('suggested_reroute', False) else 'No'}"
    ]

    for i, metric in enumerate(metrics):
        cv2.putText(dashboard, metric, (20, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Cumulative metrics
    cumulative = analyzer.get_sustainability_metrics()
    y_offset += 120
    cumulative_metrics = [
        f"Total Fuel Saved: {cumulative['total_fuel_saved_liters']:.2f} L",
        f"Total CO2 Reduced: {cumulative['total_co2_reduced_kg']:.2f} kg",
        f"Equivalent Trees: {cumulative['trees_equivalent']:.1f} trees",
        f"Trips Analyzed: {cumulative['trips_analyzed']}"
    ]

    for i, metric in enumerate(cumulative_metrics):
        cv2.putText(dashboard, metric, (20, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return dashboard
