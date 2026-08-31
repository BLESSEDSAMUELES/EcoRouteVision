#!/usr/bin/env python3
"""
TerraPave Real-Time Demonstration
Interactive road surface defect detection, pavement quality assessment, and sustainability analytics.
"""

import argparse
import sys
import time
import cv2
import numpy as np

from terrapave import (
    RoadDetector,
    SustainabilityAnalyzer,
    draw_detections,
    create_sustainability_dashboard,
    generate_synthetic_road_frame,
)


def main():
    parser = argparse.ArgumentParser(description="TerraPave: AI Road Defect & Eco-Routing Demo")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for webcam, a video file path, or 'synthetic'")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run in synthetic simulation mode (generates dynamic asphalt and defects)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output annotated video (e.g. output.avi)")
    parser.add_argument("--show-dashboard", action="store_true", default=True,
                        help="Display the real-time sustainability telemetry dashboard panel")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold for defect detections")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process before exiting (useful for batch testing)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without opening GUI windows (useful for headless CI/servers)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TERRAPAVE: ROAD DEFECT & ECO-ROUTING INTELLIGENCE")
    print("=" * 60)
    print(f"Confidence Threshold: {args.conf}")
    print(f"Synthetic Mode:       {args.synthetic or args.source == 'synthetic'}")
    print("Initializing detection models and sustainability analyzer...")

    detector = RoadDetector(conf_threshold=args.conf)
    analyzer = SustainabilityAnalyzer()

    # Determine video source
    use_synthetic = args.synthetic or (args.source == "synthetic")
    cap = None

    if not use_synthetic:
        source_val = 0 if args.source == "0" else args.source
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            print(f"[TerraPave] Warning: Could not open video source '{args.source}'.")
            print("[TerraPave] Automatically falling back to Synthetic Road Simulation Mode.\n")
            use_synthetic = True

    # Setup video writer if output is requested
    writer = None
    frame_width = 640
    frame_height = 480

    if not use_synthetic and cap and cap.isOpened():
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, 20.0, (frame_width, frame_height))
        print(f"[TerraPave] Saving output video to '{args.output}'")

    print("\nControls:")
    print("  [q] Quit demo")
    print("  [d] Toggle real-time sustainability dashboard")
    print("  [r] Run eco-routing recommendation engine")
    print("-" * 60 + "\n")

    show_dashboard = args.show_dashboard
    frame_count = 0
    start_time = time.time()

    # Initialize persistent state variables
    last_results = {
        "detections": [],
        "road_quality": 1.0,
        "frame_shape": (frame_height, frame_width, 3),
    }
    last_metrics = analyzer.calculate_route_efficiency(1.0, 10.0)

    try:
        while True:
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                print(f"[TerraPave] Reached max frames limit ({args.max_frames}). Exiting.")
                break

            if use_synthetic:
                frame = generate_synthetic_road_frame(frame_count, frame_width, frame_height)
                # Inject a simulated detection when defects appear on synthetic road
                cycle = frame_count % 120
                if 25 <= cycle <= 75:
                    t = (cycle - 20) / 60.0
                    horizon_y = int(frame_height * 0.35)
                    defect_y = int(horizon_y + t * (frame_height - horizon_y - 60))
                    defect_x = int(frame_width // 2 + (50 * t))
                    rad = int(12 + t * 24)
                    sim_box = [defect_x - rad - 5, defect_y - rad // 2 - 5, defect_x + rad + 5, defect_y + rad // 2 + 5]
                    sim_det = {
                        "bbox": sim_box,
                        "confidence": 0.88,
                        "class_id": 0,
                        "class_name": "pothole",
                        "area": (sim_box[2] - sim_box[0]) * (sim_box[3] - sim_box[1]),
                    }
                    last_results = {
                        "detections": [sim_det],
                        "road_quality": 0.58,
                        "frame_shape": frame.shape,
                    }
                else:
                    last_results = {
                        "detections": [],
                        "road_quality": 1.0,
                        "frame_shape": frame.shape,
                    }
                # Simulate realistic video frame delay
                time.sleep(0.03)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[TerraPave] Video stream completed or interrupted.")
                    break

                # Run YOLO inference every 3 frames for optimal FPS
                if frame_count % 3 == 0 or frame_count == 1:
                    last_results = detector.detect(frame)

            # Update metrics based on latest road quality
            last_metrics = analyzer.calculate_route_efficiency(last_results["road_quality"], 10.0)

            # Draw HUD & bounding boxes
            annotated_frame = draw_detections(
                frame, last_results["detections"], last_results["road_quality"]
            )

            # Log periodic summaries to console
            if frame_count % 30 == 0:
                summary, severity = detector.get_detection_summary(last_results["detections"])
                fps = frame_count / max(0.001, (time.time() - start_time))
                print(f"[Frame {frame_count:04d} | FPS: {fps:4.1f}] "
                      f"Quality: {last_results['road_quality']:.2f} | "
                      f"Detections: {summary} | Severity: {severity:.2f} | "
                      f"Fuel Loss: {last_metrics['excess_fuel_wasted_liters']:.3f}L")

            # Attach sustainability dashboard panel
            if show_dashboard:
                dash_panel = create_sustainability_dashboard(analyzer, last_metrics)
                dash_resized = cv2.resize(dash_panel, (annotated_frame.shape[1], 280))
                display_frame = np.vstack([annotated_frame, dash_resized])
            else:
                display_frame = annotated_frame

            # Render overlay FPS
            fps = frame_count / max(0.001, (time.time() - start_time))
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, display_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            if writer:
                writer.write(annotated_frame)

            if not args.headless:
                cv2.imshow("TerraPave: AI Road Surface Intelligence", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    show_dashboard = not show_dashboard
                    print(f"[TerraPave] Dashboard toggled: {'ON' if show_dashboard else 'OFF'}")
                elif key == ord("r"):
                    # Candidate alternative routes
                    candidate_routes = [
                        {"name": "Route A (Smooth Highway)", "quality": 0.95, "distance_km": 11.2, "time_minutes": 14.0},
                        {"name": "Route B (Standard Arterial)", "quality": 0.72, "distance_km": 9.8, "time_minutes": 15.5},
                        {"name": "Route C (Short Degraded Road)", "quality": 0.42, "distance_km": 8.0, "time_minutes": 18.0},
                    ]
                    suggestion = analyzer.suggest_alternative_route(last_results["road_quality"], candidate_routes)
                    if suggestion and suggestion["best_route"]:
                        best = suggestion["best_route"]
                        print("\n" + "=" * 55)
                        print("           ECO-ROUTING RECOMMENDATION")
                        print("=" * 55)
                        print(f"Current Road Quality:   {suggestion['current_route_quality']:.2f}")
                        print(f"Recommended Option:     {best['name']}")
                        print(f"Recommendation Level:   {best['recommendation']}")
                        print(f"Est. Fuel Saved:        {best['estimated_fuel_saving_liters']:.3f} Liters")
                        print(f"Est. CO2 Reduced:       {best['co2_saved_kg']:.3f} kg")
                        print(f"Efficiency Score:       {best['efficiency_score']:.2f}")
                        print("=" * 55 + "\n")

    except KeyboardInterrupt:
        print("\n[TerraPave] Interrupted by user.")

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Final Trip Summary
    final_stats = analyzer.get_sustainability_metrics()
    print("\n" + "=" * 60)
    print("          TERRAPAVE FINAL SUSTAINABILITY REPORT")
    print("=" * 60)
    for k, v in final_stats.items():
        print(f"  • {k.replace('_', ' ').title():<30}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
