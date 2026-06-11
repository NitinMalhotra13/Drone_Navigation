# Autonomous 3D Multi-Drone Cooperative Area Coverage

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://github.com/DLR-RM/stable-baselines3)
[![Interactive 3D](https://img.shields.io/badge/Visuals-Plotly%203D-purple.svg)](https://plotly.com/)

An advanced hierarchical control system for multi-drone cooperative maximum area coverage inside a complex 3D environment with physical constraints (wind, thermals, wake turbulence, static/dynamic obstacles, battery drainage, and recharge pads).

---

## 🚀 System Architecture

The project implements a **3-Tier Hierarchical Control System** to coordinate a fleet of 6 drones:

```
┌────────────────────────────────────────────────────────┐
│             TIER 1: Global Path Planner                │
│       Firefly Algorithm (FA) Sweep Optimization       │
└──────────────────────────┬─────────────────────────────┘
                           │ Coordinated 3D Waypoints
                           ▼
┌────────────────────────────────────────────────────────┐
│             TIER 2: Adaptive Replanner                 │
│      Raven Roosting Algorithm (RRA) Checkpoints        │
└──────────────────────────┬─────────────────────────────┘
                           │ Missed Gap / Shared Map Updates
                           ▼
┌────────────────────────────────────────────────────────┐
│             TIER 3: Low-Level Controller               │
│   RL Policy (PPO) + Autopilot Fallback Repulsion       │
└────────────────────────────────────────────────────────┘
```

1. **Global Path Planner (Firefly Algorithm - FA)**: Plans optimal, fanned-out forward sweeps and return paths to maximize coverage and minimize battery usage/path overlap.
2. **Adaptive Replanner (Raven Roosting Algorithm - RRA)**: Mid-mission checkpoint optimizer that dynamically updates remaining waypoints based on real-time fleet coverage (utilizing a **Shared Coverage Map**).
3. **Low-Level Controller (RL PPO + Autopilot Fallback)**: A Proximal Policy Optimization policy handles navigation. If a drone encounters obstacle collision dangers, an autopilot repulsion override takes control to steer around the obstacle before handing control back.

---

## 🌪️ Environmental Physics Modeled

* **Wind & Turbulences**: Spatially-varying, temporally-evolving wind field with sudden gust spikes.
* **Aerodynamic Drag**: Quadratic speed drag models simulating energy consumption.
* **Wake Turbulence**: Neighboring drones create downward airflow perturbations.
* **Thermal Updrafts & Downdrafts**: Sun-warmed ground cells create localized lifts.
* **Battery Degradation & Recharge**: Drones actively monitor battery status and navigate to the nearest recharge station, docking to recharge before resuming flight.
* **Dynamic Obstacles**: 25 dynamic obstacles move in full 3D space.
* **Ground Effect**: Near-ground cushioning reduces drag.

---

## 📁 Repository Structure

```
├── dataset/                     # Generated terrain, obstacles, and waypoint configs
├── models/                      # Trained RL policies, saved MP4 videos and HTML views
│   ├── drone_coverage.mp4       # 2D visualization video showing flight sweeps
│   └── interactive_3d_view.html # Rotatable Plotly 3D flight dashboard
├── src/
│   ├── fa_coverage.py           # Firefly Algorithm global planner
│   ├── ra_coverage.py           # Raven Roosting Algorithm replanner
│   ├── train_multi_drone_ppo.py # Stable-Baselines3 PPO training script
│   ├── multi_drone_coverage_env.py # Gymnasium 3D drone physics environment
│   ├── generate_terrain.py      # Perlin noise terrain generator
│   └── generate_static_obstacles.py # Static trees/rocks placement
├── quick_results.py             # Unified script to run simulation, render 2D & 3D logs
└── requirements.txt             # Project Python dependencies
```

---

## 🔧 Installation & Setup

### 1. Prerequisites
* **Python 3.10**
* **FFmpeg** (Required for matplotlib MP4 export)
  * *Windows:* `choco install ffmpeg`
  * *Mac:* `brew install ffmpeg`
  * *Ubuntu:* `sudo apt install ffmpeg`

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Execution Guide

### 1. Generate Environment Map
Create the Perlin terrain and moderate static obstacle layout (trees and rocks, no buildings):
```bash
python src/generate_terrain.py
python src/generate_static_obstacles.py
```

### 2. Train Low-Level RL Controller (Optional)
If you want to train the PPO model from scratch:
```bash
python src/train_multi_drone_ppo.py --timesteps 500000
```

### 3. Run Unified Flight Simulation
Run the integrated global/adaptive sweep system. This script simulates the mission, prints performance metrics, and exports synchronized visualizations:
```bash
python quick_results.py
```

---

## 📊 Outputs & Visualizations

Upon completing the simulation, the script generates two synchronized flight logs:

### 1. 2D MP4 Animation (`models/drone_coverage.mp4`)
* Displays the fanned-out forward sweep and coordinated return sweeps.
* Displays dynamic obstacles (royalblue circles) and coverage heatmaps (uncovered cells and obstacles are transparent).
* Highlights recharging phases and successful touchdown sequences on start pads.

### 2. Interactive 3D Plotly Twin (`models/interactive_3d_view.html`)
* **Interactive 3D Terrain**: Rotatable, zoomable surface showing Perlin height contours.
* **Transparent Coverage Heatmap**: Draped directly over the terrain surface.
* **Wood & Leaf Trees**: Rendered as brown trunks topped with fluffy, volumetric green foliage (bushes).
* **Launch/Goal Pads**: Large start pads labeled `START i` sitting on the terrain.
* **Checkpoints**: Planned intermediate waypoints shown along paths.

*Double-click `models/interactive_3d_view.html` to explore the flight logs directly in your browser!*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
