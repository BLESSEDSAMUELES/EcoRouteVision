#!/usr/bin/env python3
"""
EcoRouteVision Demo
Real-time road anomaly detection and sustainability analysis
"""

import cv2
import argparse
import time
from ecoroute_vision import RoadDetector, SustainabilityAnalyzer
from ecoroute_vision.utils import draw_detections, create_sustainability_dashboard
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='EcoRouteVision Demo')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, or file path)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path')
    parser.add_argument('--show-dashboard', action='store_true',
                        help='Show sustainability dashboard')
    args = parser.parse_args()

    # Initialize components
    print("Initializing EcoRouteVision...")
    detector = RoadDetector()
    analyzer = SustainabilityAnalyzer()

    # Open video source
    if args.source == '0':
        source = 0  # Webcam
    else:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("Starting detection... Press 'q' to quit, 'd' to toggle dashboard")

    show_dashboard = args.show_dashboard
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run detection every 5 frames for performance
        if frame_count % 5 == 0:
            results = detector.detect(frame)

            # Calculate sustainability metrics (simulate 10km route)
            sustainability_metrics = analyzer.calculate_route_efficiency(
                results['road_quality'], 10.0
            )

            # Draw detections on frame
            annotated_frame = draw_detections(
                frame, results['detections'], results['road_quality']
            )

            # Show detection summary in console
            if frame_count % 30 == 0:  # Every 30 processed frames
                summary, severity = detector.get_detection_summary(
                    results['detections'])
                print(f"Road Quality: {results['road_quality']:.2f}, "
                      f"Detections: {summary}, Severity: {severity:.2f}")

        else:
            annotated_frame = frame

        # Create and show dashboard if enabled
        if show_dashboard and frame_count % 5 == 0:
            dashboard = create_sustainability_dashboard(
                analyzer, sustainability_metrics)
            # Resize dashboard to match frame width
            dash_height = 300
            dash_width = annotated_frame.shape[1]
            dashboard = cv2.resize(dashboard, (dash_width, dash_height))

            # Combine frame and dashboard
            combined = np.vstack([annotated_frame, dashboard])
            display_frame = combined
        else:
            display_frame = annotated_frame

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(display_frame, f"FPS: {fps:.1f}",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frame
        cv2.imshow('EcoRouteVision', display_frame)

        # Write to output file
        if writer:
            writer.write(annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_dashboard = not show_dashboard
        elif key == ord('r'):
            # Simulate route suggestion
            available_routes = [
                {'name': 'Route A', 'quality': 0.9,
                    'distance_km': 12, 'time_minutes': 18},
                {'name': 'Route B', 'quality': 0.7,
                    'distance_km': 10, 'time_minutes': 15},
                {'name': 'Route C', 'quality': 0.6,
                    'distance_km': 9, 'time_minutes': 14}
            ]
            suggestion = analyzer.suggest_alternative_route(
                results['road_quality'], available_routes
            )
            if suggestion:
                print(f"Route suggestion: {suggestion['best_route']['name']} "
                      f"(Efficiency: {suggestion['best_route']['efficiency_score']:.2f})")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print final sustainability report
    final_metrics = analyzer.get_sustainability_metrics()
    print("\n" + "="*50)
    print("FINAL SUSTAINABILITY REPORT")
    print("="*50)
    for key, value in final_metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()
