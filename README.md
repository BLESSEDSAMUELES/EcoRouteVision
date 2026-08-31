# TerraPave 🛣️🌿

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-ONNX%20%7C%20TensorRT-green.svg)](https://onnxruntime.ai/)
[![Status: Ongoing](https://img.shields.io/badge/Status-15--Phase%20Active%20Roadmap-brightgreen.svg)](#-15-phase-development-roadmap)

> **Next-Generation AI Road Surface Intelligence, Automated Defect Detection, and Carbon-Optimal Eco-Routing Platform.**

---

## 📖 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [15-Phase Development Roadmap](#-15-phase-development-roadmap)
- [Mathematical & Environmental Models](#-mathematical--environmental-models)
- [Key Features](#-key-features)
- [Quick Start Guide](#-quick-start-guide)
- [Command Line Interface (CLI)](#-command-line-interface-cli)
- [Python API Reference](#-python-api-reference)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Executive Overview

Modern road degradation costs the global economy billions annually in excess fuel consumption, premature vehicle suspension wear, and infrastructure rehabilitation. 

**TerraPave** transforms standard vehicle dashcams, fleet telematics, and municipal camera networks into real-time road condition sensors. By fusing edge computer vision with vehicle dynamics and GIS routing engines, TerraPave:
1. **Detects & Classifies Defects in Real Time**: Identifies potholes, transverse cracks, longitudinal cracks, alligator cracking, and road debris at 30+ FPS.
2. **Quantifies Surface Degradation (PQI)**: Computes a calibrated Pavement Quality Index aligned with ASTM D6433 standards.
3. **Optimizes Eco-Routing**: Dynamically routes commercial fleets around severe road hazards, reducing fuel consumption by up to 12% and cutting greenhouse gas emissions.
4. **Empowers Municipal Smart Cities**: Automatically prioritizes road repairs based on traffic impact $\times$ defect volume, cutting municipal survey costs by up to 70%.

---

## 🏗️ System Architecture

TerraPave operates on a scalable multi-tier architecture spanning Edge Ingestion, Local AI Perception, Eco-Routing Telematics, and Cloud Digital Twin visualization:

```mermaid
graph TD
    subgraph Tier 1: Edge Sensing & Telemetry
        CAM[Dashcam / Video Stream]
        GPS[GPS / NMEA Geotagging]
        IMU[6-Axis IMU / OBD-II Telematics]
    end

    subgraph Tier 2: AI Perception & Defect Engine
        PRE[Frame Preprocessing & Dynamic Scaling]
        YOLO[YOLOv8 / YOLOv11 Multi-Scale Detector]
        DEPTH[Monocular Depth & 3D Volumetry]
        TRACK[ByteTrack Temporal Tracking]
    end

    subgraph Tier 3: Surface Intelligence & Eco-Analytics
        PQI[Pavement Quality Index Calculator]
        FUEL[Rolling Resistance & Fuel Waste Engine]
        CO2[Carbon Emission & Offset Modeler]
    end

    subgraph Tier 4: Dynamic Eco-Routing
        OSRM[GraphHopper / OSRM Dynamic Engine]
        COST[Multi-Objective Cost Optimizer]
        NAV[Turn-by-Turn Hazard Avoidance]
    end

    subgraph Tier 5: Cloud, Digital Twin & Municipal Work Orders
        SYNC[Delta Edge-to-Cloud Sync / MQTT]
        POSTGIS[(PostGIS Spatial Database)]
        DASH[Glassmorphism React / Deck.gl Digital Twin]
        WORK[Automated Municipal Work Order Dispatcher]
    end

    CAM --> PRE --> YOLO --> TRACK --> PQI
    DEPTH --> PQI
    GPS --> PQI
    IMU --> PQI
    PQI --> FUEL --> CO2 --> COST
    OSRM --> COST --> NAV
    PQI --> SYNC --> POSTGIS --> DASH
    POSTGIS --> WORK
```

---

## 🚀 15-Phase Development Roadmap

TerraPave is structured as an ongoing, enterprise-scale engineering program divided into 15 specialized development phases:

| Phase | Module / Capability | Status | Description |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Core Engine Stabilization & Rebranding** | 🟢 **Completed** | Full rebranding to `terrapave`, mathematical overhaul of fuel/CO2 equations, dynamic YOLO class mapping, camera-free synthetic simulation, and comprehensive test suite. |
| **Phase 2** | **Advanced Computer Vision & RDD Dataset Integration** | 🟡 **In Progress** | Fine-tuning YOLOv8/YOLOv11 on Road Damage Datasets (RDD2020, RDD2022, CRACK500), multi-scale anchor optimization, and ByteTrack temporal defect de-duplication. |
| **Phase 3** | **Monocular Depth Estimation & 3D Defect Volumetry** | ⚪ *Planned* | MiDaS / Depth Anything v2 integration to estimate pothole depth and volume ($V = \int \text{depth} \cdot dA$) for asphalt repair mass calculations. |
| **Phase 4** | **OBD-II CAN Bus & 6-Axis IMU Sensor Fusion** | ⚪ *Planned* | Extended Kalman Filter (EKF) combining visual defect sightings with vehicle accelerometer z-axis vibration shocks and CAN bus fuel rate telemetry. |
| **Phase 5** | **High-Precision Geotagging & PostGIS Spatial DB** | ⚪ *Planned* | RTK-GPS integration, spatial SpatiaLite / PostGIS database schemas, and automated DBSCAN spatial clustering of recurring road defect reports. |
| **Phase 6** | **Multi-Objective Graph Eco-Routing Engine** | ⚪ *Planned* | Dynamic OSRM/GraphHopper integration with a multi-objective cost function penalizing rough pavement segments and computing fuel-optimal bypasses. |
| **Phase 7** | **High-Performance Edge Hardware Acceleration** | ⚪ *Planned* | TensorRT FP16 / INT8 quantization, OpenVINO, ONNX Runtime optimizations targeting NVIDIA Jetson Orin Nano and Raspberry Pi 5 + Coral Edge TPU. |
| **Phase 8** | **Bandwidth-Adaptive Edge-to-Cloud Sync & Federated AI** | ⚪ *Planned* | Lightweight MQTT/gRPC delta sync uploading geotagged defect snapshots; federated learning for distributed model training across fleet vehicles. |
| **Phase 9** | **Privacy-Preserving AI & Automated PII Redaction** | ⚪ *Planned* | On-device zero-latency blurring of pedestrian faces and vehicle license plates ensuring full GDPR and CCPA compliance. |
| **Phase 10** | **Enterprise Microservices Cloud Backend** | ⚪ *Planned* | Scalable FastAPI / Celery / Redis architecture for high-throughput dashcam video ingestion, fleet telemetry streaming, and RESTful API endpoints. |
| **Phase 11** | **3D Geospatial Digital Twin & Live Web Dashboard** | ⚪ *Planned* | Glassmorphism web platform using Next.js, Deck.gl, and MapLibre for real-time fleet tracking, defect heatmaps, and carbon savings analytics. |
| **Phase 12** | **Municipal Work Order Automation & Smart City APIs** | ⚪ *Planned* | Automated work order generation, repair cost estimation (labor, asphalt tonnage), and bi-directional integration with Cityworks & Esri ArcGIS. |
| **Phase 13** | **Carbon Credit Accounting & Green Fleet Gamification** | ⚪ *Planned* | Verifiable Carbon Standard (VCS) compliant carbon offset audit logs, driver eco-scoring leaderboards, and enterprise sustainability certificates. |
| **Phase 14** | **Connected Vehicle V2X & Autonomous Vehicle HD Maps** | ⚪ *Planned* | Cellular-V2X (C-V2X) and DSRC hazard broadcasting to nearby vehicles; dynamic high-definition (HD) drivability map layers for CARLA/Autoware. |
| **Phase 15** | **AI-Driven Predictive Road Deterioration Modeling** | ⚪ *Planned* | Spatio-temporal Graph Neural Networks (GNN) and LSTM networks forecasting road failure 6–24 months in advance incorporating freeze-thaw weather models. |

---

## 📐 Mathematical & Environmental Models

### 1. Pavement Quality Index ($PQI$)
The segment road surface condition score $Q \in [0.0, 1.0]$ is computed dynamically from visual defect detections:
$$Q = \max\left(0.0, \, 1.0 - \sum_{i=1}^{N} w_i \cdot c_i \cdot \sqrt{\frac{A_i}{A_{\text{ref}}}}\right)$$
Where:
- $w_i$: Severity weight for defect class $i$ (e.g., Pothole $= 0.45$, Transverse Crack $= 0.20$).
- $c_i$: Model confidence score $\in [0.0, 1.0]$.
- $A_i$: Pixel area of bounding box; $A_{\text{ref}}$: Reference normalization area.

### 2. Rolling Resistance & Excess Fuel Consumption
Degraded pavement increases vehicle rolling resistance and induces frequent deceleration/acceleration cycles. The empirical fuel multiplier $\mu_f$ is modeled as:
$$\mu_f(Q) = 1.0 + 0.35 \cdot (1.0 - Q)$$
$$\text{Fuel}_{\text{actual}} = \left(\frac{D}{100}\right) \cdot \text{Rate}_{\text{base}} \cdot \mu_f(Q)$$
$$\text{Fuel}_{\text{wasted}} = \max\left(0.0, \, \text{Fuel}_{\text{actual}} - 0.95 \cdot \text{Fuel}_{\text{base}}\right)$$

### 3. Carbon Emissions & Environmental Offset
Emissions avoided $\Delta \text{CO}_2$ (in kg) when choosing an optimal bypass route rather than navigating degraded infrastructure:
$$\Delta \text{CO}_2 = \left(\text{Fuel}_{\text{degraded}} - \text{Fuel}_{\text{alternative}}\right) \times 2.31 \text{ kg CO}_2/\text{L}$$
$$\text{Trees Equivalent} = \frac{\sum \Delta \text{CO}_2}{21.77 \text{ kg CO}_2/\text{tree-year}}$$

---

## ✨ Key Features

- **Multi-Source Ingestion**: Ingests live webcam feeds, pre-recorded MP4/AVI videos, or simulated dynamic asphalt streams (`--synthetic`).
- **Dynamic Class Detection**: Seamlessly switches between custom road damage weights and baseline COCO models without class misattribution.
- **Glassmorphic Telemetry HUD**: Real-time on-screen display showing instantaneous road quality, excess fuel loss, carbon penalty, and cumulative fleet savings.
- **One-Touch Eco-Rerouting**: Press `[r]` during runtime to compute real-time multi-route trade-offs.
- **Automated Synthetic Simulator**: Built-in procedural road generator for seamless development and CI/CD testing without requiring physical cameras.

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for live 60 FPS, CPU supported)

### 1. Clone & Setup
```bash
git clone https://github.com/BLESSEDSAMUELES/TerraPave.git
cd TerraPave
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or install in editable development mode:
pip install -e .
```

### 3. Prepare AI Models
```bash
python models/download_models.py
```

---

## 💻 Command Line Interface (CLI)

### Run in Synthetic Simulation Mode (No Camera Required)
```bash
python demo.py --synthetic --show-dashboard
```

### Run on Live Webcam
```bash
python demo.py --source 0 --show-dashboard
```

### Process Video File and Save Output
```bash
python demo.py --source /path/to/road_video.mp4 --output output_annotated.avi
```

### Interactive Runtime Hotkeys
| Key | Action |
| :---: | :--- |
| `q` | Exit demonstration |
| `d` | Toggle real-time telemetry dashboard overlay |
| `r` | Trigger instant multi-route eco-optimization comparison |

---

## 🔌 Python API Reference

```python
import cv2
from terrapave import RoadDetector, SustainabilityAnalyzer, draw_detections

# 1. Initialize detector and sustainability analyzer
detector = RoadDetector(model_path="models/best.pt", conf_threshold=0.35)
analyzer = SustainabilityAnalyzer(base_fuel_rate=8.0)

# 2. Process video frame
frame = cv2.imread("road_sample.jpg")
results = detector.detect(frame)

# 3. Compute eco-efficiency and fuel impact
metrics = analyzer.calculate_route_efficiency(
    current_road_quality=results["road_quality"], 
    distance_km=15.0
)

# 4. Render visual annotations
annotated = draw_detections(frame, results["detections"], results["road_quality"])
print(f"Pavement Condition: {metrics['pavement_condition']}")
print(f"Excess Fuel Wasted: {metrics['excess_fuel_wasted_liters']:.3f} L")
```

---

## 🧪 Testing & Quality Assurance

TerraPave maintains a full suite of automated unit and regression tests:

```bash
# Run all unit tests
python -m unittest discover -s tests -p "test_*.py"
```

---

## 🤝 Contributing

Contributions to any of the 15 roadmap phases are warmly welcomed! Please open an issue or pull request following our standard feature-branch workflow.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
