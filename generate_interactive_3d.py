"""
generate_interactive_3d.py
Runs the integrated FA+RA+PPO(waypoint) simulation and generates a fully interactive,
rotatable, zoomable 3D visualization using Plotly, saved as a standalone HTML file.
"""
import os
import sys
import time
import math
import numpy as np
import plotly.graph_objects as go

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)

from multi_drone_coverage_env import MultiDroneCoverageEnv, DRONE_COLORS, RECHARGE_STATIONS_XY
from fa_coverage import FireflyPlanner
from ra_coverage import RavenReplanner

# ── Config ──────────────────────────────────────────────────────────────────
N_STEPS        = 2000         # full mission length
N_DRONES       = 6
DYN_COUNT      = 25           # dynamic obstacles
REPLAN_EVERY   = 999999       # Disable RA replanning — use robust FA sweeps
FA_FIREFLIES   = 50           # more fireflies → better coverage
FA_ITERS       = 70           # more iterations → higher fitness
FA_WP_FILE     = os.path.join(os.path.dirname(__file__), "dataset", "fa_waypoints.npy")
HTML_OUT       = os.path.join(os.path.dirname(__file__), "models", "interactive_3d_view.html")

# ── Helpers ─────────────────────────────────────────────────────────────────
def waypoint_action(positions, waypoints, return_waypoints, return_mode, wp_idx, max_speed, n_drones):
    action = np.zeros((n_drones, 3))
    for i in range(n_drones):
        idx    = min(max(0, int(wp_idx[i])), waypoints.shape[1] - 1)
        wpts   = return_waypoints if return_mode[i] else waypoints
        diff   = wpts[i, idx] - positions[i]
        dist   = np.linalg.norm(diff) + 1e-9
        action[i] = np.clip((diff / dist) * max_speed, -1, 1)
    return action.flatten()

# ── Environment Setup ───────────────────────────────────────────────────────
print("\n[INIT] Building environment...")
_static_path = os.path.join(os.path.dirname(__file__), "dataset", "static_obstacles.npy")
# Force regenerate to make sure we load the correct moderate density
if os.path.exists(_static_path):
    os.remove(_static_path)

from src.generate_static_obstacles import generate_static_obstacles
_static_grid = generate_static_obstacles()
_n_obs = int(_static_grid.sum())
print(f"[INIT] Regenerated static obstacles: {_n_obs} occupied 3D cells")

env = MultiDroneCoverageEnv(
    n_drones=N_DRONES, wind_enabled=True,
    thermal_enabled=True, sensor_noise_std=0.04,
    dyn_count=DYN_COUNT, sensor_radius=4.5
)
obs, _ = env.reset()

start_positions = env.positions.copy()
start_positions[:, 2] = 6.0  # Set planning start height to cruise altitude

goal_positions = np.array([
    [94.0, 94.0, 6.0],   # Drone 0
    [94.0, 96.0, 6.0],   # Drone 1
    [96.0, 94.0, 6.0],   # Drone 2
    [96.0, 96.0, 6.0],   # Drone 3
    [95.0, 94.0, 6.0],   # Drone 4
    [95.0, 96.0, 6.0],   # Drone 5
])[:N_DRONES]

# ── FA Waypoints ────────────────────────────────────────────────────────────
print(f"[FA] Running Firefly Algorithm ({FA_FIREFLIES} fireflies x {FA_ITERS} iters)...")
fa = FireflyPlanner(
    n_drones=N_DRONES, n_waypoints=20,
    n_fireflies=FA_FIREFLIES, max_iter=FA_ITERS,
    start_positions=start_positions,
    goal_positions=goal_positions
)
fa.optimize(verbose=False)
waypoints = fa.get_best_waypoints()

# Compute shared direction/perp vectors
V_ref   = goal_positions[0] - start_positions[0]
D_ref   = np.linalg.norm(V_ref[:2]) + 1e-9
dir_vec = V_ref[:2] / D_ref
perp_vec = np.array([-dir_vec[1], dir_vec[0]])

# Compute return waypoints
return_waypoints = waypoints.copy()
for i in range(N_DRONES):
    for w in range(1, waypoints.shape[1] - 1):
        return_waypoints[i, w, :2] += 13.0 * perp_vec
        return_waypoints[i, w, 0]   = np.clip(return_waypoints[i, w, 0], 1.0, 99.0)
        return_waypoints[i, w, 1]   = np.clip(return_waypoints[i, w, 1], 1.0, 99.0)

# ── Simulation Flight Loop ──────────────────────────────────────────────────
IMPORTANT_POINTS = np.concatenate([
    np.array(RECHARGE_STATIONS_XY, dtype=float),
    [[5.0, 5.0], [95.0, 95.0]]
], axis=0)

wp_idx       = np.zeros(N_DRONES, dtype=int)
return_mode  = np.zeros(N_DRONES, dtype=bool)
wpt_steps    = np.zeros(N_DRONES, dtype=int)
drone_paths  = [[] for _ in range(N_DRONES)]

print("\n[RUN] Simulating flight trajectories (fast CPU execution)...")
t0 = time.time()
for step in range(N_STEPS):
    action = waypoint_action(
        env.positions, waypoints, return_waypoints, return_mode, wp_idx,
        env.max_speed, N_DRONES
    )
    obs, reward, term, trunc, info = env.step(action)

    for i in range(N_DRONES):
        if env.phases[i] in ["cruise", "return_home"]:
            while True:
                idx = min(max(0, int(wp_idx[i])), waypoints.shape[1] - 1)
                wpt = return_waypoints[i, idx] if return_mode[i] else waypoints[i, idx]
                cx, cy = int(round(wpt[0])), int(round(wpt[1]))
                r = int(math.ceil(env.sensor_radius))
                x_lo = max(0, cx - r); x_hi = min(env.GX, cx + r + 1)
                y_lo = max(0, cy - r); y_hi = min(env.GY, cy + r + 1)
                
                near_important = False
                for ip in IMPORTANT_POINTS:
                    if np.linalg.norm(wpt[:2] - ip) < 12.0:
                        near_important = True
                        break
                        
                if not near_important and x_hi > x_lo and y_hi > y_lo:
                    wpt_area = env.coverage_grid[x_lo:x_hi, y_lo:y_hi]
                    coverage_pct = wpt_area.sum() / wpt_area.size
                    if coverage_pct > 0.85:
                        if return_mode[i] and wp_idx[i] > 0:
                            wp_idx[i] -= 1
                            wpt_steps[i] = 0
                            continue
                break

            idx = min(max(0, int(wp_idx[i])), waypoints.shape[1] - 1)
            wpt = return_waypoints[i, idx] if return_mode[i] else waypoints[i, idx]
            wpt_steps[i] += 1
            dist_to_wpt = np.linalg.norm(env.positions[i] - wpt)
            
            if dist_to_wpt < 5.0 or wpt_steps[i] > 40:
                wpt_steps[i] = 0
                if not return_mode[i]:
                    if wp_idx[i] < waypoints.shape[1] - 1:
                        wp_idx[i] += 1
                    else:
                        return_mode[i] = True
                        env.phases[i] = "return_home"
                        wp_idx[i] = waypoints.shape[1] - 1
                else:
                    if wp_idx[i] > 0:
                        wp_idx[i] -= 1
                    else:
                        env.phases[i] = "descend_land"

    # Record drone positions
    for i in range(N_DRONES):
        drone_paths[i].append(env.positions[i].copy())

    if term or trunc:
        break

print(f"[RUN] Simulation finished in {time.time() - t0:.1f}s. Coverage reached: {env.coverage_ratio()*100:.2f}%")

# ── Generate Plotly Interactive 3D ──────────────────────────────────────────
print("\n[VISUAL] Constructing Plotly 3D Scene...")
fig = go.Figure()

# 1. Terrain Surface with vibrant blue contour lines (matching 2D exactly)
X_t = np.arange(100)
Y_t = np.arange(100)
fig.add_trace(go.Surface(
    x=X_t, y=Y_t, z=env.terrain.T,
    colorscale="earth",
    opacity=0.7,
    showscale=False,
    hoverinfo="z",
    name="Terrain Elevation",
    contours=dict(
        z=dict(show=True, usecolormap=False, color="#4b92db", width=2.0, start=0.5, end=14.5, size=2.0)
    )
))

# 1b. Translucent Coverage Heatmap draped over the 3D Terrain
# Matches the 2D coverage overlay exactly (uncovered cells and obstacles are transparent NaNs)
z_cov = env.terrain.copy()
# Set uncovered cells to NaN
z_cov[~env.coverage_grid] = np.nan
# Mask out obstacles to match 2D masking
z_cov[env.static[:, :, 6]] = np.nan

fig.add_trace(go.Surface(
    x=X_t, y=Y_t, z=z_cov.T + 0.1,  # 10cm offset above terrain to prevent Z-fighting
    colorscale=[[0, "#2e8b57"], [1, "#2e8b57"]],  # Solid translucent forestgreen
    opacity=0.55,
    showscale=False,
    hoverinfo="skip",
    name="Area Coverage Heatmap"
))

# 2. Parse 3D Static Grid to render realistic Trees (with leafy bushes) and Rocks (no buildings)
tx, ty, tz = env.static.shape
tree_trunks_x, tree_trunks_y, tree_trunks_z = [], [], []
tree_leaves_x, tree_leaves_y, tree_leaves_z = [], [], []
rocks_x, rocks_y, rocks_z = [], [], []

for x in range(tx):
    for y in range(ty):
        blocked_zs = np.where(env.static[x, y])[0]
        if len(blocked_zs) > 0:
            z_min = int(blocked_zs.min())
            z_max = int(blocked_zs.max())
            height = z_max - z_min + 1
            
            if height >= 5:  # Tree!
                # Trunk: from ground_height (z_min) up to z_max - 2
                for z_val in range(z_min, z_max - 1):
                    tree_trunks_x.append(x + 0.5)
                    tree_trunks_y.append(y + 0.5)
                    tree_trunks_z.append(z_val + 0.5)
                # Foliage / Crown (Proper Fluffy Bush): cluster at the top
                tree_leaves_x.append(x + 0.5)
                tree_leaves_y.append(y + 0.5)
                tree_leaves_z.append(z_max + 0.5)
                # Scatter leaf offsets to make a volumetric bush shape
                offsets = [
                    (0.3, 0, -0.2), (-0.3, 0, -0.2), (0, 0.3, -0.2), (0, -0.3, -0.2),
                    (0.2, 0.2, -0.5), (-0.2, -0.2, -0.5), (0.2, -0.2, -0.5), (-0.2, 0.2, -0.5)
                ]
                for dx, dy, dz_off in offsets:
                    tree_leaves_x.append(x + 0.5 + dx)
                    tree_leaves_y.append(y + 0.5 + dy)
                    tree_leaves_z.append(z_max + 0.5 + dz_off)
            else:  # Rock!
                for z_val in blocked_zs:
                    rocks_x.append(x + 0.5)
                    rocks_y.append(y + 0.5)
                    rocks_z.append(z_val + 0.5)

# Render parsed obstacles
if len(tree_trunks_x) > 0:
    fig.add_trace(go.Scatter3d(
        x=tree_trunks_x, y=tree_trunks_y, z=tree_trunks_z,
        mode="markers",
        marker=dict(size=3, symbol="square", color="#5a3d28", opacity=0.9),
        name="Tree Trunks (Wood)"
    ))

if len(tree_leaves_x) > 0:
    fig.add_trace(go.Scatter3d(
        x=tree_leaves_x, y=tree_leaves_y, z=tree_leaves_z,
        mode="markers",
        marker=dict(size=6.5, symbol="circle", color="#2e8b57", opacity=0.6, line=dict(width=0)),
        name="Tree Leaves (Proper Bush)"
    ))

if len(rocks_x) > 0:
    fig.add_trace(go.Scatter3d(
        x=rocks_x, y=rocks_y, z=rocks_z,
        mode="markers",
        marker=dict(size=4.5, symbol="diamond", color="#696969", opacity=0.8),
        name="Static Rocks / Boulders"
    ))

# 3. Recharge Stations (Placed sitting precisely on the terrain surface)
rs_xy = np.array(RECHARGE_STATIONS_XY, dtype=float)
rs_z = []
for rx, ry in rs_xy:
    rx_idx = int(np.clip(rx, 0, 99))
    ry_idx = int(np.clip(ry, 0, 99))
    rs_z.append(float(env.terrain[rx_idx, ry_idx]) + 0.1)

fig.add_trace(go.Scatter3d(
    x=rs_xy[:, 0], y=rs_xy[:, 1], z=rs_z,
    mode="markers+text",
    marker=dict(size=9, symbol="cross", color="magenta", line=dict(width=1.2, color="white")),
    text=["Recharge Station"] * len(rs_xy),
    textposition="top center",
    textfont=dict(size=8, color="magenta"),
    name="Recharge Stations"
))

# 3b. Dynamic Obstacles (Blue spheres at actual coordinates matching 2D royalblue circles)
dyn_pos = np.array([p.copy() for p in env.dyn.positions])
if len(dyn_pos) > 0:
    fig.add_trace(go.Scatter3d(
        x=dyn_pos[:, 0], y=dyn_pos[:, 1], z=dyn_pos[:, 2],
        mode="markers",
        marker=dict(size=6.0, symbol="circle", color="royalblue", line=dict(width=1.0, color="white")),
        name="Dynamic Obstacles"
    ))

# 4. Drone Flight Paths, Starts, Checkpoints, and Goals
for i in range(N_DRONES):
    pts = np.array(drone_paths[i])
    col = DRONE_COLORS[i]
    
    # 3D line representing trajectory path
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(color=col, width=4.5),
        name=f"Drone {i} Path"
    ))
    
    # Takeoff Landing Pad (Start Zone) - Large, prominent, sitting on terrain
    start_pos = start_positions[i]
    sx_idx = int(np.clip(start_pos[0], 0, 99))
    sy_idx = int(np.clip(start_pos[1], 0, 99))
    start_ground_z = float(env.terrain[sx_idx, sy_idx])
    
    fig.add_trace(go.Scatter3d(
        x=[start_pos[0]], y=[start_pos[1]], z=[start_ground_z],
        mode="markers+text",
        marker=dict(size=13, symbol="square", color="cyan", line=dict(width=1.5, color="white")),
        text=f"START {i}",
        textposition="top center",
        textfont=dict(size=8.5, color="cyan"),
        name=f"Start Pad {i}",
        legendgroup=f"drone_{i}",
        showlegend=True if i == 0 else False
    ))
    
    # Intermediate planned Checkpoints (Waypoints) - highly visible spheres
    wpts_3d = waypoints[i]  # shape (n_waypoints, 3)
    fig.add_trace(go.Scatter3d(
        x=wpts_3d[:, 0], y=wpts_3d[:, 1], z=wpts_3d[:, 2],
        mode="markers",
        marker=dict(size=5.5, symbol="circle", color=col, line=dict(width=1.2, color="white")),
        name=f"Drone {i} Checkpoints",
        legendgroup=f"drone_{i}",
        showlegend=True if i == 0 else False
    ))
    
    # Goal/Landing Pad - prominent colored diamond, sitting on terrain
    goal_pos = goal_positions[i]
    gx_idx = int(np.clip(goal_pos[0], 0, 99))
    gy_idx = int(np.clip(goal_pos[1], 0, 99))
    goal_ground_z = float(env.terrain[gx_idx, gy_idx])
    
    fig.add_trace(go.Scatter3d(
        x=[goal_pos[0]], y=[goal_pos[1]], z=[goal_ground_z],
        mode="markers+text",
        marker=dict(size=10, symbol="diamond", color=col, line=dict(width=1.2, color="white")),
        text=f"GOAL {i}",
        textposition="top center",
        textfont=dict(size=8.5, color=col),
        name=f"Goal Pad {i}",
        legendgroup=f"drone_{i}",
        showlegend=True if i == 0 else False
    ))

# ── Dark Premium Layout Styling ─────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text="<b>Multi-Drone Autonomous Coverage — Interactive 3D Flight Log</b>",
        x=0.5, y=0.95,
        xanchor="center", yanchor="top",
        font=dict(size=18, color="white", family="Inter, Roboto, Arial")
    ),
    template="plotly_dark",
    scene=dict(
        xaxis=dict(
            title="X (meters)", range=[0, 100],
            gridcolor="#2c2c4d", backgroundcolor="#0e0e22", showbackground=True
        ),
        yaxis=dict(
            title="Y (meters)", range=[0, 100],
            gridcolor="#2c2c4d", backgroundcolor="#0e0e22", showbackground=True
        ),
        zaxis=dict(
            title="Z Altitude (m)", range=[0, 15],
            gridcolor="#2c2c4d", backgroundcolor="#0f0f2a", showbackground=True
        ),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.35)  # Squash vertical axis slightly for optimal visual proportions
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    legend=dict(
        x=0.02, y=0.98,
        bgcolor="rgba(19, 19, 40, 0.85)",
        bordercolor="#3a3a6a",
        borderwidth=1,
        font=dict(size=10)
    )
)

os.makedirs(os.path.dirname(HTML_OUT), exist_ok=True)
fig.write_html(HTML_OUT, include_plotlyjs="cdn")
print(f"[DONE] Standalone HTML saved to {HTML_OUT}  ({os.path.getsize(HTML_OUT)/1024:.1f} KB)")
print("\nDouble-click this file on your machine to rotate, zoom, and explore in your browser!")
