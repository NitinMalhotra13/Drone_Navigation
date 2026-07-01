"""
quick_results.py
Runs the integrated FA+RA+PPO(waypoint) system for N steps,
prints a full results table, and saves a GIF/MP4 video.
Designed to complete in under 3 minutes on CPU.
"""
import os, sys, time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)

from multi_drone_coverage_env import MultiDroneCoverageEnv, DRONE_COLORS, RECHARGE_STATIONS_XY
from fa_coverage import FireflyPlanner
from ra_coverage import RavenReplanner

# ── Config (BEST QUALITY — no time constraints) ───────────────────────────
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\ffmpeg-2026-02-09-git-9bfa1635ae-essentials_build\bin\ffmpeg.EXE'

N_STEPS        = 2000         # full mission length
N_DRONES       = 6
DYN_COUNT      = 25           # dynamic obstacles (was 12)
REPLAN_EVERY   = 350          # Replan paths adaptively using RRA every 350 steps
FA_FIREFLIES   = 50           # more fireflies → better coverage solutions (was 30)
FA_ITERS       = 70           # more iterations → higher fitness convergence (was 40)
RA_RAVENS      = 15
RA_ITERS       = 20
FA_WP_FILE     = os.path.join(os.path.dirname(__file__), "dataset", "fa_waypoints.npy")
VIDEO_OUT      = os.path.join(os.path.dirname(__file__), "models", "drone_coverage.gif")
MP4_OUT        = os.path.join(os.path.dirname(__file__), "models", "drone_coverage.mp4")
FRAME_SKIP     = 5            # capture every 5th frame for smooth video
FPS            = 10           # slower playback for easier human tracking
VIDEO_DPI      = 90           # optimized DPI for fast render

# ── Helpers ────────────────────────────────────────────────────────────────
def waypoint_action(positions, waypoints, return_waypoints, return_mode, wp_idx, max_speed, n_drones):
    """
    Wind-compensated waypoint action: drives drones toward targets smoothly.
    Applies a proportional slowdown zone below 8.0m to prevent overshoot,
    oscillations, and sharp zigzags, while preserving a minimum 0.25 speed
    cushion to overcome opposing wind drift.
    """
    action = np.zeros((n_drones, 3))
    for i in range(n_drones):
        idx    = min(max(0, int(wp_idx[i])), waypoints.shape[1] - 1)
        wpts   = return_waypoints if return_mode[i] else waypoints
        diff   = wpts[i, idx] - positions[i]
        dist   = np.linalg.norm(diff) + 1e-9
        
        # Proportional slowdown zone to guarantee smooth transitions with zero oscillations
        if dist < 8.0:
            speed_factor = max(0.35, dist / 8.0)  # smooth decay, min 0.35 to fight wind
            action[i] = np.clip((diff / dist) * max_speed * speed_factor, -1, 1)
        else:
            action[i] = np.clip((diff / dist) * max_speed, -1, 1)
    return action.flatten()

# ── Environment ────────────────────────────────────────────────────────────
print("\n[INIT] Building environment...")

# Force-regenerate static obstacles so the dense obstacle config takes effect
# CRITICAL: must regenerate BEFORE creating the env, otherwise env loads empty grid
_static_path = os.path.join(os.path.dirname(__file__), "dataset", "static_obstacles.npy")
if os.path.exists(_static_path):
    os.remove(_static_path)
    print("[INIT] Removed stale static_obstacles.npy")

# Regenerate fresh obstacle file
from src.generate_static_obstacles import generate_static_obstacles
_static_grid = generate_static_obstacles()
_n_obs = int(_static_grid.sum())
print(f"[INIT] Regenerated static_obstacles.npy — {_n_obs} obstacle cells")

env = MultiDroneCoverageEnv(n_drones=N_DRONES, wind_enabled=True,
                             thermal_enabled=True, sensor_noise_std=0.04,
                             dyn_count=DYN_COUNT,
                             sensor_radius=4.5)   # wider footprint: 63.6 cells/step vs 38.5 (+65%)
obs, _ = env.reset()

# Define explicit start and goal positions for unified cluster-to-cluster coverage
start_positions = env.positions.copy()
start_positions[:, 2] = 6.0  # Set planning start height to cruise altitude

# Original clustered goal: all drones head toward top-right, no crossing paths.
# The FA assigns each drone a distinct lateral corridor via corridor_center.
goal_positions = np.array([
    [94.0, 94.0, 6.0],   # Drone 0
    [94.0, 96.0, 6.0],   # Drone 1
    [96.0, 94.0, 6.0],   # Drone 2
    [96.0, 96.0, 6.0],   # Drone 3
    [95.0, 94.0, 6.0],   # Drone 4
    [95.0, 96.0, 6.0],   # Drone 5
])[:N_DRONES]

# Keep starts at actual env positions (no staggering — avoids mismatch collisions)
start_positions = env.positions.copy()
start_positions[:, 2] = 6.0

# ── FA Waypoints ───────────────────────────────────────────────────────────
print(f"[FA] Running Firefly Algorithm ({FA_FIREFLIES} fireflies x {FA_ITERS} iters)...")
fa = FireflyPlanner(
    n_drones=N_DRONES, n_waypoints=20,   # 20 waypoints for richer sweeps
    n_fireflies=FA_FIREFLIES, max_iter=FA_ITERS,
    start_positions=start_positions,
    goal_positions=goal_positions
)
fa.optimize(verbose=True)
waypoints = fa.get_best_waypoints()
np.save(FA_WP_FILE, waypoints)

# PASS 2: return sweep planned by FA (no hand-coding, pure algorithmic fanning-out!)
print(f"[FA] Planning return sweep waypoints ({FA_FIREFLIES} fireflies x {FA_ITERS} iters)...")
fa_return = FireflyPlanner(
    n_drones=N_DRONES, n_waypoints=20,
    n_fireflies=FA_FIREFLIES, max_iter=FA_ITERS,
    start_positions=goal_positions,     # starts at goal
    goal_positions=start_positions       # returns to start
)
fa_return.optimize(verbose=False)
return_waypoints = fa_return.get_best_waypoints()

stats = fa.get_coverage_stats()
print(f"[FA] Done. Coverage est: {stats['coverage_ratio']*100:.1f}%  "
      f"Path: {stats['total_path_length']:.0f}m  Battery: {stats['total_battery']:.1f}")

# ── RA Replanner ───────────────────────────────────────────────────────────
# ── RA Replanner (best quality: more ravens, more iters) ────────────────
ra = RavenReplanner(
    n_drones=N_DRONES, n_ravens=RA_RAVENS, max_iter=RA_ITERS,
    goal_positions=goal_positions
)
print(f"[RA] Raven Replanner ready ({RA_RAVENS} ravens x {RA_ITERS} iters per replan)")

# ── Run loop ───────────────────────────────────────────────────────────────
IMPORTANT_POINTS = np.concatenate([
    np.array(RECHARGE_STATIONS_XY, dtype=float),
    [[5.0, 5.0], [95.0, 95.0]]
], axis=0)

wp_idx       = np.zeros(N_DRONES, dtype=int)
return_mode  = np.zeros(N_DRONES, dtype=bool)
wpt_steps    = np.zeros(N_DRONES, dtype=int)
drone_paths  = [[] for _ in range(N_DRONES)]
frames       = []

cov_hist, col_hist, bat_hist, path_hist, wind_hist = [], [], [], [], []

# Per-step multi-objective tracking
step_collisions_hist = []
step_path_hist       = []
step_battery_hist    = []

print(f"\n[RUN] Integrated FA+RA system | {N_STEPS} steps | {N_DRONES} drones")
print(f"      Grid: 100x100x15  |  Wind: ON  |  Thermals: ON")
print(f"      FA: {FA_FIREFLIES} fireflies x {FA_ITERS} iters  |  RA: {RA_RAVENS} ravens x {RA_ITERS} iters")
print(f"      Dynamic obstacles: {DYN_COUNT}  |  RA replanning every {REPLAN_EVERY} steps")
print("=" * 65)

t0 = time.time()
for step in range(N_STEPS):

    # RRA replan at specific mid-flight milestones (step 450 and 950)
    # This dynamically targets uncovered gaps using RRA, while keeping paths clean, smooth, and sweeping.
    if step in [450, 950]:
        if not return_mode.any():
            print(f"\n  [RRA] Dynamic replanning of forward sweep at step {step}...")
            print(f"        Shared Map Coverage: {env.coverage_ratio()*100:.2f}%")
            try:
                min_idx = int(wp_idx.min())
                n_rem = waypoints.shape[1] - min_idx
                if n_rem > 2:
                    new_wp = ra.replan(
                        current_positions=env.positions.copy(),
                        coverage_grid=env.coverage_grid.copy(),
                        n_remaining_waypoints=n_rem,
                    )
                    waypoints[:, min_idx:, :] = new_wp
                    wp_idx = np.clip(wp_idx, 0, waypoints.shape[1] - 1)
                    print("  [RRA] Forward sweep path optimized successfully.")
            except Exception as e:
                print(f"  [RRA] Forward replan error: {e}")
        else:
            print(f"\n  [RRA] Dynamic replanning of return sweep at step {step}...")
            print(f"        Shared Map Coverage: {env.coverage_ratio()*100:.2f}%")
            try:
                min_idx = int(wp_idx.min())
                n_rem = return_waypoints.shape[1] - min_idx
                if n_rem > 2:
                    # Plan return using RRA with start_positions as targets!
                    ra_return = RavenReplanner(
                        n_drones=N_DRONES, n_ravens=RA_RAVENS, max_iter=RA_ITERS,
                        goal_positions=start_positions
                    )
                    new_wp = ra_return.replan(
                        current_positions=env.positions.copy(),
                        coverage_grid=env.coverage_grid.copy(),
                        n_remaining_waypoints=n_rem,
                    )
                    return_waypoints[:, min_idx:, :] = new_wp
                    wp_idx = np.clip(wp_idx, 0, return_waypoints.shape[1] - 1)
                    print("  [RRA] Return sweep path optimized successfully.")
            except Exception as e:
                print(f"  [RRA] Return replan error: {e}")

    action = waypoint_action(env.positions, waypoints, return_waypoints, return_mode, wp_idx,
                              env.max_speed, N_DRONES)
    obs, reward, term, trunc, info = env.step(action)

    # Advance waypoint index per drone when close to target or if already covered
    for i in range(N_DRONES):
        if env.phases[i] in ["cruise", "return_home"]:
            # Skip already covered waypoints (excluding important points)
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

            # Increment step counter for current waypoint
            wpt_steps[i] += 1

            # Standard distance-based transition
            dist_to_wpt = np.linalg.norm(env.positions[i] - wpt)
            
            # Timeout or distance-based transition
            # 5.0m radius → reliably reachable even with strong wind gusts
            # 40-step timeout → 3× faster unsticking vs. previous 120-step window
            if dist_to_wpt < 5.0 or wpt_steps[i] > 40:
                if wpt_steps[i] > 40:
                    print(f"[SKIP] Drone {i} advancing past waypoint {idx} (dist={dist_to_wpt:.1f}m, {wpt_steps[i]} steps)")
                
                wpt_steps[i] = 0
                if not return_mode[i]:
                    if wp_idx[i] < waypoints.shape[1] - 1:
                        wp_idx[i] += 1
                    else:
                        # Forward sweep completed: initiate return home reverse sweep!
                        return_mode[i] = True
                        env.phases[i] = "return_home"
                        wp_idx[i] = waypoints.shape[1] - 1
                        print(f"[INFO] Drone {i} sweep completed. Initiating fanned-out return sweep...")
                else:
                    if wp_idx[i] > 0:
                        wp_idx[i] -= 1
                    else:
                        # Reverse return sweep completed: land safely!
                        env.phases[i] = "descend_land"
                        print(f"[INFO] Drone {i} returned home safely. Commencing touchdown...")

    for i in range(N_DRONES):
        drone_paths[i].append(env.positions[i].copy())

    cov_hist.append(info["coverage_pct"])
    col_hist.append(info["total_collisions"])
    bat_hist.append(info["total_battery_used"])
    path_hist.append(info["total_path_length"])
    wind_hist.append(info["wind_speed"])
    step_collisions_hist.append(info["step_collisions"])
    step_path_hist.append(info["step_path"])
    step_battery_hist.append(info["step_battery"])

    # Capture frame every FRAME_SKIP steps
    if step % FRAME_SKIP == 0:
        frames.append({
            "step":         step,
            "positions":    env.positions.copy(),
            "coverage":     env.coverage_grid.copy(),
            "batteries":    list(info["batteries"]),
            "phases":       list(info["phases"]),
            "motor_health": list(info["motor_health"]),
            "dyn_pos":      [p.copy() for p in env.dyn.positions],
            "cov_pct":      info["coverage_pct"],
            "collisions":   info["total_collisions"],
            "bat_used":     info["total_battery_used"],
            "path_len":     info["total_path_length"],
            "wind":         info["wind_speed"],
            "drone_paths":  [list(drone_paths[i]) for i in range(N_DRONES)],
            "wp_idx":       wp_idx.copy(),
            "waypoints":    waypoints.copy(),
            "active_wpt":   np.array([
                return_waypoints[i, min(max(0, int(wp_idx[i])), waypoints.shape[1] - 1)] if return_mode[i]
                else waypoints[i, min(max(0, int(wp_idx[i])), waypoints.shape[1] - 1)]
                for i in range(N_DRONES)
            ]),
        })

    if step % 100 == 0:
        elapsed = time.time() - t0
        print(f"  step={step:4d} | cov={info['coverage_pct']:5.1f}% "
              f"| col={info['total_collisions']:3d} "
              f"| bat_used={info['total_battery_used']:6.1f} "
              f"| path={info['total_path_length']:7.1f} "
              f"| wind={info['wind_speed']:.3f} "
              f"| t={elapsed:.1f}s")

    if term or trunc:
        print(f"\n  Episode ended at step {step}")
        break

elapsed_total = time.time() - t0
final = frames[-1]

# ══════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 65)
print("  MULTI-DRONE COVERAGE RESULTS  |  FA + RA + Physics Sim")
print("=" * 65)
print(f"  Grid Size          : 100 x 100 x 15")
print(f"  Number of Drones   : {N_DRONES}")
print(f"  Steps Completed    : {env.step_count}")
print(f"  Simulation Time    : {elapsed_total:.1f} s")
print(f"  [PRIMARY]  Area Coverage     : {final['cov_pct']:.2f} %")
print("-" * 65)
for i in range(N_DRONES):
    print(f"  Drone {i} | Phase: {final['phases'][i]:15s} | Battery: {final['batteries'][i]:5.1f} | wp_idx: {final['wp_idx'][i]:2d}")
print("-" * 65)
# ══════════════════════════════════════════════════════════════════════════
# VIDEO / GIF RENDERING (BLITTED / HIGH PERFORMANCE)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[VIDEO] Rendering {len(frames)} frames using optimized in-place updates...")
t_vid_start = time.time()

fig = plt.figure(figsize=(24, 11), facecolor="#0d0d1a")
fig.suptitle("Multi-Drone Coverage  |  FA + RA + PPO  |  6 Drones  |  100x100 Grid",
             color="white", fontsize=14, fontweight="bold", y=0.98)

gs      = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30,
                   left=0.05, right=0.98, top=0.92, bottom=0.06)
ax_map  = fig.add_subplot(gs[0, 0])
ax_map_3d = fig.add_subplot(gs[0, 1], projection='3d')
ax_met  = fig.add_subplot(gs[0, 2])
ax_bat  = fig.add_subplot(gs[1, 0])
ax_stat = fig.add_subplot(gs[1, 1:])

for ax in [ax_map, ax_met, ax_bat, ax_stat]:
    ax.set_facecolor("#131328")
    for spine in ax.spines.values():
        spine.set_color("#3a3a6a")
    ax.tick_params(colors="#aaaacc")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

# ── [0,0] Map Panel (Initialize STATIC Artists) ───────────────────────────
rs_xy = np.array(RECHARGE_STATIONS_XY, dtype=float)
# Terrain contour (static) — vibrant bold lines and filled contours so they stand out clearly
X_t = np.arange(100); Y_t = np.arange(100)
ax_map.contour(X_t, Y_t, env.terrain.T, levels=8,
               colors=["#4b92db"], linewidths=1.3, alpha=0.8)
ax_map.contourf(X_t, Y_t, env.terrain.T, levels=8,
                cmap="terrain", alpha=0.25)  # more visible filled terrain tint

# ── 3D Map Panel (Initialize Static Surface, Contours, and Obstacles) ──────
ax_map_3d.set_facecolor("#131328")
try:
    ax_map_3d.xaxis.set_pane_color((0.07, 0.07, 0.15, 1.0))
    ax_map_3d.yaxis.set_pane_color((0.07, 0.07, 0.15, 1.0))
    ax_map_3d.zaxis.set_pane_color((0.09, 0.09, 0.18, 1.0))
except AttributeError:
    ax_map_3d.w_xaxis.set_pane_color((0.07, 0.07, 0.15, 1.0))
    ax_map_3d.w_yaxis.set_pane_color((0.07, 0.07, 0.15, 1.0))
    ax_map_3d.w_zaxis.set_pane_color((0.09, 0.09, 0.18, 1.0))
ax_map_3d.tick_params(colors="#aaaacc")
ax_map_3d.xaxis.label.set_color("white")
ax_map_3d.yaxis.label.set_color("white")
ax_map_3d.zaxis.label.set_color("white")
ax_map_3d.title.set_color("white")
ax_map_3d.grid(True, color="#2c2c4d")

# 3D terrain surface
ax_map_3d.plot_surface(X_t, Y_t, env.terrain.T, rstride=4, cstride=4, cmap="terrain", alpha=0.3, zorder=1)
# 3D contours
ax_map_3d.contour(X_t, Y_t, env.terrain.T, levels=8, zdir='z', offset=0, colors=["#4b92db"], linewidths=1.0, alpha=0.6, zorder=2)

# ── Static obstacles as SCATTER MARKERS (large, clearly visible) ───────────
# Extract obstacle cell positions from the 3D static grid (specifically at flight cruise altitude)
obs_2d = env.static[:, :, 6]  # (100, 100) bool
obs_mask_2d = obs_2d  # kept for coverage masking in animate()
obs_coords = np.argwhere(obs_2d)  # (N, 2) array of [x, y] obstacle positions

if len(obs_coords) > 0:
    ox, oy = obs_coords[:, 0], obs_coords[:, 1]
    # Draw 2D obstacles
    ax_map.scatter(ox, oy, s=18, marker="s", c="#8B4513",
                   edgecolors="#5a2d0c", linewidths=0.3,
                   alpha=0.85, zorder=5, label=f"Static Obstacles ({len(obs_coords)})")
    print(f"[VIDEO] Drawing {len(obs_coords)} static obstacle cells as scatter markers")
else:
    print("[VIDEO] WARNING: No static obstacles found in env.static!")

# Parse 3D Static Grid to render realistic Trees (with leafy bushes) and Rocks on 3D view
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
                for z_val in range(z_min, z_max - 1):
                    tree_trunks_x.append(x + 0.5)
                    tree_trunks_y.append(y + 0.5)
                    tree_trunks_z.append(z_val + 0.5)
                # Volumetric crown dome (leafy crown)
                tree_leaves_x.append(x + 0.5)
                tree_leaves_y.append(y + 0.5)
                tree_leaves_z.append(z_max + 0.5)
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

# Render static obstacles in 3D
if len(tree_trunks_x) > 0:
    ax_map_3d.scatter(tree_trunks_x, tree_trunks_y, tree_trunks_z, s=4, c="#5a3d28", marker="s", alpha=0.9, depthshade=True, zorder=3)
if len(tree_leaves_x) > 0:
    ax_map_3d.scatter(tree_leaves_x, tree_leaves_y, tree_leaves_z, s=15, c="#2e8b57", marker="o", alpha=0.6, depthshade=True, zorder=4)
if len(rocks_x) > 0:
    ax_map_3d.scatter(rocks_x, rocks_y, rocks_z, s=6, c="#696969", marker="d", alpha=0.8, depthshade=True, zorder=3)

# Draw Important Home & Landing Zones (2D circles)
ax_map.add_patch(mpatches.Circle((5, 5), 10, fill=False, edgecolor="cyan", linestyle="--", linewidth=1.2, hatch="//", alpha=0.5, zorder=4))
ax_map.add_patch(mpatches.Circle((95, 95), 10, fill=False, edgecolor="cyan", linestyle="--", linewidth=1.2, hatch="//", alpha=0.5, zorder=4))

# Recharge stations (2D + 3D)
ax_map.scatter(rs_xy[:, 0], rs_xy[:, 1], s=130, marker="s",
               c="magenta", edgecolors="white", linewidths=0.9, zorder=6)
for rx, ry in rs_xy:
    ax_map.text(rx, ry + 2.5, "R", color="magenta", fontsize=7,
                ha="center", va="bottom", fontweight="bold")

# Recharge stations in 3D
rs_z = []
for rx, ry in rs_xy:
    rx_idx = int(np.clip(rx, 0, 99))
    ry_idx = int(np.clip(ry, 0, 99))
    rs_z.append(float(env.terrain[rx_idx, ry_idx]) + 0.15)
ax_map_3d.scatter(rs_xy[:, 0], rs_xy[:, 1], rs_z, s=80, c="magenta", marker="X", depthshade=True, zorder=5)

# 3D Goals
for i in range(N_DRONES):
    goal = goal_positions[i]
    gx_idx = int(np.clip(goal[0], 0, 99))
    gy_idx = int(np.clip(goal[1], 0, 99))
    gz = float(env.terrain[gx_idx, gy_idx])
    ax_map_3d.scatter([goal[0]], [goal[1]], [gz], s=90, c=DRONE_COLORS[i], marker="*", alpha=0.9, zorder=5)

# Legend
from matplotlib.lines import Line2D
leg_obstacles = Line2D([0],[0], marker="s", color="w", markerfacecolor="#8B4513",
                       markersize=7, linestyle="None", label=f"Static Obstacles ({len(obs_coords)})")
leg_dyn       = Line2D([0],[0], marker="o", color="w", markerfacecolor="royalblue",
                       markersize=7, linestyle="None", label="Dynamic Obstacles")
leg_recharge  = Line2D([0],[0], marker="s", color="w", markerfacecolor="magenta",
                       markersize=7, linestyle="None", label="Recharge Pad")
leg_home      = mpatches.Patch(color="cyan", fill=False, hatch="//", label="Home/Goal Zone")
leg_terrain   = Line2D([0],[0], color="#5577aa", linewidth=1.2, label="Terrain Contour")
ax_map.legend(handles=[leg_obstacles, leg_dyn, leg_recharge, leg_home, leg_terrain],
              loc="upper left", fontsize=6.5,
              facecolor="#1a1a35", labelcolor="white", framealpha=0.85)

# ── [0,0] Map Panel (Initialize DYNAMIC Artists) ───────────────────────────
# Coverage heatmap at zorder=2 — below obstacles (zorder=6) and drones (zorder=7)
init_cov = np.full((100, 100), np.nan)
im_cov = ax_map.imshow(init_cov, origin="lower", cmap="YlGn",
                       vmin=0, vmax=1, alpha=0.55,
                       extent=[0, 100, 0, 100], aspect="auto", zorder=2)

sc_dyn = ax_map.scatter([], [], s=50, c="royalblue",
                        alpha=0.75, edgecolors="white", linewidths=0.4,
                        zorder=4, marker="o")

# ── 3D Map Panel (Initialize DYNAMIC Artists) ──────────────────────────────
sc_dyn_3d = ax_map_3d.scatter([], [], [], s=25, c="royalblue",
                            alpha=0.75, edgecolors="white", linewidths=0.4, zorder=5)

# Create arrays of per-drone path lines, dot markers, waypoints, and labels
line_paths = []
sc_drones  = []
txt_drones = []
sc_wpts    = []
line_wpts  = []
sc_goals   = []

# 3D counterparts
line_paths_3d = []
sc_drones_3d  = []
sc_wpts_3d    = []
line_wpts_3d  = []

for i in range(N_DRONES):
    col = DRONE_COLORS[i]
    # 2D Trajectory path line
    lp, = ax_map.plot([], [], color=col, linewidth=1.1, alpha=0.55, zorder=3)
    line_paths.append(lp)
    
    # 3D Trajectory path line
    lp3d, = ax_map_3d.plot([], [], [], color=col, linewidth=1.5, alpha=0.6, zorder=4)
    line_paths_3d.append(lp3d)
    
    # Current 2D drone coordinate triangle marker
    sd = ax_map.scatter([], [], s=100, c=col, edgecolors="white", linewidths=0.9, zorder=6, marker="^")
    sc_drones.append(sd)
    
    # Current 3D drone coordinate cone marker
    sd3d = ax_map_3d.scatter([], [], [], s=60, c=col, edgecolors="white", linewidths=0.8, marker="^", zorder=6)
    sc_drones_3d.append(sd3d)
    
    # Drone label text
    td = ax_map.text(0, 0, f"D{i}", fontsize=6.5, color=col, fontweight="bold", zorder=7)
    txt_drones.append(td)
    
    # Active 2D target marker
    sw = ax_map.scatter([], [], s=45, marker="x", c=col, linewidths=1.3, alpha=0.8, zorder=5)
    sc_wpts.append(sw)
    
    # Active 3D target marker
    sw3d = ax_map_3d.scatter([], [], [], s=30, marker="x", c=col, linewidths=1.1, alpha=0.8, zorder=5)
    sc_wpts_3d.append(sw3d)
    
    # 2D guidance line
    lw, = ax_map.plot([], [], color=col, linewidth=0.5, alpha=0.35, linestyle="--", zorder=4)
    line_wpts.append(lw)
    
    # 3D guidance line
    lw3d, = ax_map_3d.plot([], [], [], color=col, linewidth=0.6, alpha=0.4, linestyle="--", zorder=4)
    line_wpts_3d.append(lw3d)
    
    # Final unified goal star (2D)
    goal = goal_positions[i]
    sg = ax_map.scatter(goal[0], goal[1], s=130, marker="*",
                        facecolors="none", edgecolors=col, linewidths=1.5, zorder=5)
    ax_map.annotate(f"Goal {i}", (goal[0], goal[1]),
                    textcoords="offset points", xytext=(0, -10),
                    fontsize=6.0, color=col, fontweight="bold", ha="center")
    sc_goals.append(sg)

ax_map.set_xlim(0, 100); ax_map.set_ylim(0, 100)
ax_map.set_xlabel("X (m)", fontsize=8); ax_map.set_ylabel("Y (m)", fontsize=8)
ax_map.set_title("2D Map View", fontsize=9, color="white")

ax_map_3d.set_xlim(0, 100); ax_map_3d.set_ylim(0, 100); ax_map_3d.set_zlim(0, 15)
ax_map_3d.set_xlabel("X (m)", fontsize=8); ax_map_3d.set_ylabel("Y (m)", fontsize=8); ax_map_3d.set_zlabel("Z (m)", fontsize=8)
ax_map_3d.set_title("3D Map View", fontsize=9, color="white")
ax_map_3d.view_init(elev=28, azim=-45)


# ── [0,1] Metrics Panel (Initialize Artists) ──────────────────────────────
line_cov, = ax_met.plot([], [], color="#00ff88", linewidth=1.5, label="Coverage %")
ax_met.set_ylabel("Coverage %", color="#00ff88", fontsize=8)
ax_met.tick_params(axis="y", labelcolor="#00ff88", colors="#aaaacc")
ax_met.set_ylim(0, 105)

ax2 = ax_met.twinx()
ax2.set_facecolor("#131328")
for sp in ax2.spines.values():
    sp.set_color("#3a3a6a")
ax2.tick_params(axis="y", labelcolor="white", colors="#aaaacc")
ax2.set_ylim(0, 115)

line_col, = ax2.plot([], [], color="#ff4455", linewidth=1.1, linestyle="--", label="Collisions (norm)")
line_bat, = ax2.plot([], [], color="#ffaa00", linewidth=1.0, linestyle=":",  label="Battery (norm)")
line_path, = ax2.plot([], [], color="#44ccff", linewidth=1.0, linestyle="-.", label="Path (norm)")

ax_met.set_xlabel("Step", fontsize=8)
ax_met.set_title("Multi-Objective Metrics  (Obj1: Collision | Obj2: Path | Obj3: Battery)",
                 fontsize=8.5, color="white")

# ── [1,0] Battery Panel (Initialize Artists) ───────────────────────────────
bar_rects = []
bar_texts = []
for i in range(N_DRONES):
    # Create horizontal bar using default 100% width
    rect = ax_bat.barh(i, 100, color="#00ff88", alpha=0.85, height=0.65,
                       edgecolor="#3a3a6a", linewidth=0.6)[0]
    txt = ax_bat.text(101, i, "100%", va="center", fontsize=7.5, color="white")
    bar_rects.append(rect)
    bar_texts.append(txt)

ax_bat.set_xlim(0, 115)
ax_bat.set_yticks(range(N_DRONES))
ax_bat.set_yticklabels(
    [f"D{i}  mh=100%" for i in range(N_DRONES)],
    fontsize=7.5, color="white"
)
ax_bat.set_xlabel("Battery %", fontsize=8)
ax_bat.set_title("Per-Drone Battery & Motor Health", fontsize=9, color="white")
ax_bat.axvline(20, color="#ff4455", linestyle="--", linewidth=0.8, alpha=0.6)
ax_bat.axvline(50, color="#ffaa00", linestyle=":",  linewidth=0.7, alpha=0.5)

# ── [1,1] Scorecard Panel (Initialize Artists) ─────────────────────────────
ax_stat.axis("off")
txt_scorecard = ax_stat.text(0.05, 0.95, "", transform=ax_stat.transAxes,
                             fontsize=9.0, color="white", va="top", fontfamily="monospace")
ax_stat.set_title("System Status", fontsize=9, color="white")

# Legend once outside
lines1, lbl1 = ax_met.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax_met.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left",
              fontsize=6.5, facecolor="#1a1a35",
              labelcolor="white", framealpha=0.8)

# ── ANIMATION UPDATE LOOP (artist-in-place) ────────────────────────────────
def animate(fi):
    fd = frames[fi]
    step_n = fd["step"]
    t_end  = min(step_n + 1, len(cov_hist))
    xs = list(range(t_end))

    # 1. Update Map Panel
    # Mask obstacle cells and uncovered cells to be fully transparent (NaN) so they don't cover background/terrain
    cov_display = fd["coverage"].astype(float).copy()
    cov_display[cov_display == 0.0] = np.nan
    cov_display[obs_mask_2d] = np.nan
    im_cov.set_data(cov_display.T)
    # Update dynamic obstacle positions
    if fd["dyn_pos"]:
        dp = np.array(fd["dyn_pos"])
        sc_dyn.set_offsets(dp[:, :2])
        # Update 3D dynamic obstacles
        sc_dyn_3d._offsets3d = (dp[:, 0], dp[:, 1], dp[:, 2])
    
    for i in range(N_DRONES):
        col  = DRONE_COLORS[i]
        path = fd["drone_paths"][i]
        if len(path) > 1:
            pts = np.array(path)
            line_paths[i].set_data(pts[:, 0], pts[:, 1])
            # Update 3D drone path
            line_paths_3d[i].set_data(pts[:, 0], pts[:, 1])
            line_paths_3d[i].set_3d_properties(pts[:, 2])
        pos = fd["positions"][i]
        sc_drones[i].set_offsets(pos[:2])
        txt_drones[i].set_position((pos[0] + 3, pos[1] + 3))
        # Update 3D drone position
        sc_drones_3d[i]._offsets3d = (np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]))
        
        # waypoint target
        wpt = fd["active_wpt"][i]
        sc_wpts[i].set_offsets(wpt[:2])
        line_wpts[i].set_data([pos[0], wpt[0]], [pos[1], wpt[1]])
        # Update 3D waypoint target and guidance line
        sc_wpts_3d[i]._offsets3d = (np.array([wpt[0]]), np.array([wpt[1]]), np.array([wpt[2]]))
        line_wpts_3d[i].set_data([pos[0], wpt[0]], [pos[1], wpt[1]])
        line_wpts_3d[i].set_3d_properties([pos[2], wpt[2]])

    ax_map.set_title(
        f"Step {fd['step']:4d}  |  Coverage: {fd['cov_pct']:.1f}%"
        f"  |  Wind: {fd['wind']:.3f}",
        fontsize=9, color="white"
    )

    # 2. Update Metrics Panel
    line_cov.set_data(xs, cov_hist[:t_end])
    max_col  = max(col_hist[-1],  1)
    max_bat  = max(bat_hist[-1],  1)
    max_path = max(path_hist[-1], 1)
    line_col.set_data(xs, [c / max_col * 100  for c in col_hist[:t_end]])
    line_bat.set_data(xs, [b / max_bat * 100  for b in bat_hist[:t_end]])
    line_path.set_data(xs, [p / max_path * 100 for p in path_hist[:t_end]])

    ax_met.set_xlim(0, max(N_STEPS, t_end))
    ax2.set_xlim(0, max(N_STEPS, t_end))

    # 3. Update Battery Panel
    bats = fd["batteries"]
    for i in range(N_DRONES):
        pct = bats[i] / env.battery_capacity * 100
        bar_color = ("#00ff88" if pct > 50 else
                     "#ffaa00" if pct > 20 else "#ff4455")
        bar_rects[i].set_width(pct)
        bar_rects[i].set_facecolor(bar_color)
        bar_texts[i].set_text(f"{pct:.0f}%  [{fd['phases'][i][:3].upper()}]")
        bar_texts[i].set_x(min(pct + 1, 104))

    ax_bat.set_yticklabels(
        [f"D{i}  mh={fd['motor_health'][i]*100:.0f}%" for i in range(N_DRONES)],
        fontsize=7.5, color="white"
    )

    # 4. Update Scorecard Panel
    scorecard_str = f"""LIVE SCORECARD
--------------
Coverage    :  {fd['cov_pct']:.2f} %
Grid Cells  :  {int(fd['cov_pct']/100*10000):,} / 10,000

[OBJ 1] Collisions   :  {fd['collisions']}
[OBJ 2] Path Length  :  {fd['path_len']:.1f} m
[OBJ 3] Battery Used :  {fd['bat_used']:.0f} units

Wind Speed  :  {fd['wind']:.4f} m/step
Step        :  {fd['step']} / {N_STEPS}

ALGORITHMS
----------
  FA  - Firefly (global planner)
  RA  - Raven Roosting (replan)
  PPO - Waypoint controller
"""
    txt_scorecard.set_text(scorecard_str)

    # No return needed — blit=False redraws everything each frame

print(f"  Animating {len(frames)} frames (blit=False for full static layer rendering)...")
anim = FuncAnimation(fig, animate, frames=len(frames),
                     interval=1000 // FPS, blit=False)

# Try MP4 first, fall back to GIF
saved_path = None
try:
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=FPS, metadata={"title": "Drone Coverage"}, bitrate=2800)
    anim.save(MP4_OUT, writer=writer, dpi=VIDEO_DPI)
    saved_path = MP4_OUT
    print(f"\n[VIDEO] MP4 saved -> {os.path.abspath(MP4_OUT)}")
except Exception as e:
    print(f"[VIDEO] ffmpeg unavailable ({type(e).__name__}). Saving GIF...")
    writer = PillowWriter(fps=FPS)
    anim.save(VIDEO_OUT, writer=writer, dpi=VIDEO_DPI)
    saved_path = VIDEO_OUT
    print(f"\n[VIDEO] GIF saved -> {os.path.abspath(VIDEO_OUT)}")

plt.close(fig)
size_mb = os.path.getsize(saved_path) / 1024 / 1024
print(f"[VIDEO] File size: {size_mb:.1f} MB")
print(f"[VIDEO] Rendering complete in {time.time() - t_vid_start:.1f} s")

# ── Generate Plotly Interactive 2D & 3D Web Dashboard ──────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("\n[VISUAL] Constructing side-by-side synchronized 2D & 3D Plotly Web Dashboard...")
    
    # Create subplots
    fig3d = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("2D Top-Down View", "Interactive 3D View")
    )
    
    # Downsample frames for smooth, fast HTML playback without ballooning file sizes
    total_anim_frames = len(frames)
    plotly_step_skip = max(1, total_anim_frames // 60)
    plotly_frames = frames[::plotly_step_skip]
    if len(plotly_frames) > 0 and plotly_frames[-1]["step"] != frames[-1]["step"]:
        plotly_frames.append(frames[-1])
        
    f0 = plotly_frames[0]
    X_t = np.arange(100)
    Y_t = np.arange(100)
    
    # ── LEFT SUBPLOT (2D View) ────────────────────────────────────────────────
    
    # 0. 2D Terrain Contours
    fig3d.add_trace(go.Contour(
        x=X_t, y=Y_t, z=env.terrain.T,
        colorscale="Earth",
        opacity=0.6,
        showscale=False,
        contours=dict(coloring="heatmap", showlabels=False),
        hoverinfo="z",
        name="2D Terrain"
    ), row=1, col=1)
    
    # 1. 2D Recharge Stations
    fig3d.add_trace(go.Scatter(
        x=rs_xy[:, 0], y=rs_xy[:, 1],
        mode="markers+text",
        marker=dict(size=12, symbol="square", color="magenta", line=dict(width=1, color="white")),
        text=["R"] * len(rs_xy),
        textposition="middle center",
        textfont=dict(size=8, color="white", weight="bold"),
        name="Recharge stations",
        showlegend=False
    ), row=1, col=1)
    
    # 2. 2D Home/Goal Zones
    fig3d.add_trace(go.Scatter(
        x=[5, 95], y=[5, 95],
        mode="markers",
        marker=dict(size=30, symbol="circle-open", color="cyan", line=dict(width=2, dash="dash")),
        name="Home/Goal Zones",
        showlegend=False
    ), row=1, col=1)
    
    # 3. 2D Coverage Heatmap
    cov_2d_z_init = f0["coverage"].astype(float).copy()
    cov_2d_z_init[obs_mask_2d] = np.nan
    fig3d.add_trace(go.Heatmap(
        x=X_t, y=Y_t, z=cov_2d_z_init.T,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(46, 139, 87, 0.6)"]],
        zmin=0, zmax=1,
        showscale=False,
        hoverinfo="skip",
        name="2D Coverage Heatmap"
    ), row=1, col=1)
    
    # 4. 2D Dynamic Obstacles
    dyn_pos_init = np.array(f0["dyn_pos"]) if f0["dyn_pos"] else np.empty((0, 3))
    if len(dyn_pos_init) > 0:
        fig3d.add_trace(go.Scatter(
            x=dyn_pos_init[:, 0], y=dyn_pos_init[:, 1],
            mode="markers",
            marker=dict(size=8, symbol="circle", color="royalblue", line=dict(width=1, color="white")),
            name="2D Dynamic Obstacles",
            showlegend=False
        ), row=1, col=1)
    else:
        fig3d.add_trace(go.Scatter(
            x=[], y=[],
            mode="markers",
            marker=dict(size=8, symbol="circle", color="royalblue", line=dict(width=1, color="white")),
            name="2D Dynamic Obstacles",
            showlegend=False
        ), row=1, col=1)
        
    # Drone Traces in 2D
    for i in range(N_DRONES):
        col = DRONE_COLORS[i]
        path_init = np.array(f0["drone_paths"][i])
        pos_init = f0["positions"][i]
        wpt_init = f0["active_wpt"][i]
        
        # Path trace (5 + 3*i)
        fig3d.add_trace(go.Scatter(
            x=path_init[:, 0] if len(path_init) > 0 else [pos_init[0]],
            y=path_init[:, 1] if len(path_init) > 0 else [pos_init[1]],
            mode="lines",
            line=dict(color=col, width=2),
            name=f"Drone {i} Path",
            legendgroup=f"drone_{i}",
            showlegend=True
        ), row=1, col=1)
        
        # Position trace (6 + 3*i)
        fig3d.add_trace(go.Scatter(
            x=[pos_init[0]], y=[pos_init[1]],
            mode="markers+text",
            marker=dict(size=10, symbol="triangle-up", color=col, line=dict(width=1, color="white")),
            text=f"D{i}",
            textposition="top center",
            textfont=dict(size=9, color=col),
            name=f"Drone {i} 2D",
            legendgroup=f"drone_{i}",
            showlegend=False
        ), row=1, col=1)
        
        # Waypoint trace (7 + 3*i)
        fig3d.add_trace(go.Scatter(
            x=[wpt_init[0]], y=[wpt_init[1]],
            mode="markers",
            marker=dict(size=8, symbol="x-thin", color=col, line=dict(width=1.5)),
            name=f"D{i} Target 2D",
            legendgroup=f"drone_{i}",
            showlegend=False
        ), row=1, col=1)
        
    # ── RIGHT SUBPLOT (3D View) ───────────────────────────────────────────────
    
    # 23. 3D Terrain Surface
    fig3d.add_trace(go.Surface(
        x=X_t, y=Y_t, z=env.terrain.T,
        colorscale="Earth",
        opacity=0.7,
        showscale=False,
        hoverinfo="z",
        name="3D Terrain",
        contours=dict(
            z=dict(show=True, usecolormap=False, color="#4b92db", width=2.0, start=0.5, end=14.5, size=2.0)
        )
    ), row=1, col=2)
    
    # 24. 3D Coverage Heatmap Surface
    cov_3d_z_init = env.terrain.T.copy()
    cov_3d_z_init[~f0["coverage"].T] = np.nan
    cov_3d_z_init[obs_mask_2d.T] = np.nan
    fig3d.add_trace(go.Surface(
        x=X_t, y=Y_t, z=cov_3d_z_init + 0.1,
        colorscale=[[0, "#2e8b57"], [1, "#2e8b57"]],
        opacity=0.55,
        showscale=False,
        hoverinfo="skip",
        name="3D Coverage"
    ), row=1, col=2)
    
    # 25. Tree Trunks
    if len(tree_trunks_x) > 0:
        fig3d.add_trace(go.Scatter3d(
            x=tree_trunks_x, y=tree_trunks_y, z=tree_trunks_z,
            mode="markers",
            marker=dict(size=2.5, symbol="square", color="#5a3d28", opacity=0.9),
            name="Tree Trunks",
            showlegend=False
        ), row=1, col=2)
        
    # 26. Tree Leaves
    if len(tree_leaves_x) > 0:
        fig3d.add_trace(go.Scatter3d(
            x=tree_leaves_x, y=tree_leaves_y, z=tree_leaves_z,
            mode="markers",
            marker=dict(size=5.5, symbol="circle", color="#2e8b57", opacity=0.6, line=dict(width=0)),
            name="Bushes",
            showlegend=False
        ), row=1, col=2)
        
    # 27. Rocks
    if len(rocks_x) > 0:
        fig3d.add_trace(go.Scatter3d(
            x=rocks_x, y=rocks_y, z=rocks_z,
            mode="markers",
            marker=dict(size=4.0, symbol="diamond", color="#696969", opacity=0.8),
            name="Rocks",
            showlegend=False
        ), row=1, col=2)
        
    # 28. Recharge Stations
    fig3d.add_trace(go.Scatter3d(
        x=rs_xy[:, 0], y=rs_xy[:, 1], z=rs_z,
        mode="markers",
        marker=dict(size=8, symbol="cross", color="magenta", line=dict(width=1.0, color="white")),
        name="Recharge Pads",
        showlegend=False
    ), row=1, col=2)
    
    # 29. Dynamic Obstacles
    if len(dyn_pos_init) > 0:
        fig3d.add_trace(go.Scatter3d(
            x=dyn_pos_init[:, 0], y=dyn_pos_init[:, 1], z=dyn_pos_init[:, 2],
            mode="markers",
            marker=dict(size=5.0, symbol="circle", color="royalblue", line=dict(width=0.8, color="white")),
            name="3D Dynamic Obstacles",
            showlegend=False
        ), row=1, col=2)
    else:
        fig3d.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode="markers",
            marker=dict(size=5.0, symbol="circle", color="royalblue", line=dict(width=0.8, color="white")),
            name="3D Dynamic Obstacles",
            showlegend=False
        ), row=1, col=2)
        
    # Drone Traces in 3D
    for i in range(N_DRONES):
        col = DRONE_COLORS[i]
        path_init = np.array(f0["drone_paths"][i])
        pos_init = f0["positions"][i]
        wpt_init = f0["active_wpt"][i]
        
        # 3D Path trace (30 + 5*i)
        fig3d.add_trace(go.Scatter3d(
            x=path_init[:, 0] if len(path_init) > 0 else [pos_init[0]],
            y=path_init[:, 1] if len(path_init) > 0 else [pos_init[1]],
            z=path_init[:, 2] if len(path_init) > 0 else [pos_init[2]],
            mode="lines",
            line=dict(color=col, width=3.5),
            name=f"D{i} Path 3D",
            legendgroup=f"drone_{i}",
            showlegend=False
        ), row=1, col=2)
        
        # 3D Start Pad (31 + 5*i)
        start_pos = start_positions[i]
        sx_idx = int(np.clip(start_pos[0], 0, 99))
        sy_idx = int(np.clip(start_pos[1], 0, 99))
        sz = float(env.terrain[sx_idx, sy_idx])
        fig3d.add_trace(go.Scatter3d(
            x=[start_pos[0]], y=[start_pos[1]], z=[sz],
            mode="markers+text",
            marker=dict(size=10, symbol="square", color="cyan", line=dict(width=1, color="white")),
            text=f"S{i}",
            textposition="top center",
            textfont=dict(size=8, color="cyan"),
            name=f"D{i} Start 3D",
            legendgroup=f"drone_{i}",
            showlegend=False
        ), row=1, col=2)
        
        # 3D Goal Pad (32 + 5*i)
        goal_pos = goal_positions[i]
        gx_idx = int(np.clip(goal_pos[0], 0, 99))
        gy_idx = int(np.clip(goal_pos[1], 0, 99))
        gz = float(env.terrain[gx_idx, gy_idx])
        fig3d.add_trace(go.Scatter3d(
            x=[goal_pos[0]], y=[goal_pos[1]], z=[gz],
            mode="markers+text",
            marker=dict(size=8, symbol="diamond", color=col, line=dict(width=1, color="white")),
            text=f"G{i}",
            textposition="top center",
            textfont=dict(size=8, color=col),
            name=f"D{i} Goal 3D",
            legendgroup=f"drone_{i}",
            showlegend=False
        ), row=1, col=2)
        
        # 3D Position trace (33 + 5*i)
        fig3d.add_trace(go.Scatter3d(
            x=[pos_init[0]], y=[pos_init[1]], z=[pos_init[2]],
            mode="markers",
            marker=dict(size=8, symbol="circle", color=col, line=dict(width=0.8, color="white")),
            name=f"D{i} Pos 3D",
            legendgroup=f"drone_{i}",
            showlegend=False
        ), row=1, col=2)
        
        # 3D Waypoint trace (34 + 5*i)
        fig3d.add_trace(go.Scatter3d(
            x=[wpt_init[0]], y=[wpt_init[1]], z=[wpt_init[2]],
            mode="markers",
            marker=dict(size=6, symbol="x", color=col, line=dict(width=1.2)),
            name=f"D{i} Target 3D",
            legendgroup=f"drone_{i}",
            showlegend=False
        ), row=1, col=2)
        
    # ── CONSTRUCT ANIMATION FRAMES ───────────────────────────────────────────
    plotly_animation_frames = []
    
    # Active trace indices list (matching the order of frame updates)
    active_indices = [3, 4]
    for i in range(N_DRONES):
        active_indices.extend([5 + 3*i, 6 + 3*i, 7 + 3*i])
    active_indices.extend([24, 29])
    for i in range(N_DRONES):
        active_indices.extend([30 + 5*i, 33 + 5*i, 34 + 5*i])
        
    for fd in plotly_frames:
        step_val = fd["step"]
        cov_grid_val = fd["coverage"]
        dyn_pos_val = np.array(fd["dyn_pos"]) if fd["dyn_pos"] else np.empty((0, 3))
        positions_val = fd["positions"]
        drone_paths_val = fd["drone_paths"]
        active_wpt_val = fd["active_wpt"]
        
        frame_data = []
        
        # 1. 2D Coverage Heatmap (Trace 3)
        cov_2d_z = cov_grid_val.astype(float).copy()
        cov_2d_z[obs_mask_2d] = np.nan
        frame_data.append(go.Heatmap(z=cov_2d_z.T))
        
        # 2. 2D Dynamic Obstacles (Trace 4)
        if len(dyn_pos_val) > 0:
            frame_data.append(go.Scatter(x=dyn_pos_val[:, 0], y=dyn_pos_val[:, 1]))
        else:
            frame_data.append(go.Scatter(x=[], y=[]))
            
        # 3. 2D Drones
        for i in range(N_DRONES):
            pts_2d = np.array(drone_paths_val[i])
            pos_val = positions_val[i]
            wpt_val = active_wpt_val[i]
            
            frame_data.append(go.Scatter(x=pts_2d[:, 0], y=pts_2d[:, 1]))
            frame_data.append(go.Scatter(x=[pos_val[0]], y=[pos_val[1]]))
            frame_data.append(go.Scatter(x=[wpt_val[0]], y=[wpt_val[1]]))
            
        # 4. 3D Coverage Heatmap Surface (Trace 24)
        cov_3d_z = env.terrain.T.copy()
        cov_3d_z[~cov_grid_val.T] = np.nan
        cov_3d_z[obs_mask_2d.T] = np.nan
        frame_data.append(go.Surface(z=cov_3d_z + 0.1))
        
        # 5. 3D Dynamic Obstacles (Trace 29)
        if len(dyn_pos_val) > 0:
            frame_data.append(go.Scatter3d(x=dyn_pos_val[:, 0], y=dyn_pos_val[:, 1], z=dyn_pos_val[:, 2]))
        else:
            frame_data.append(go.Scatter3d(x=[], y=[], z=[]))
            
        # 6. 3D Drones
        for i in range(N_DRONES):
            pts_3d = np.array(drone_paths_val[i])
            pos_val = positions_val[i]
            wpt_val = active_wpt_val[i]
            
            frame_data.append(go.Scatter3d(x=pts_3d[:, 0], y=pts_3d[:, 1], z=pts_3d[:, 2]))
            frame_data.append(go.Scatter3d(x=[pos_val[0]], y=[pos_val[1]], z=[pos_val[2]]))
            frame_data.append(go.Scatter3d(x=[wpt_val[0]], y=[wpt_val[1]], z=[wpt_val[2]]))
            
        plotly_animation_frames.append(go.Frame(
            data=frame_data,
            name=f"frame_{step_val}",
            traces=active_indices
        ))
        
    fig3d.frames = plotly_animation_frames
    
    # ── CONTROLS (Buttons and Slider) ────────────────────────────────────────
    updatemenus = [dict(
        type="buttons",
        showactive=False,
        direction="left",
        x=0.08, y=0.0,
        xanchor="right", yanchor="top",
        pad=dict(t=25, r=10),
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, transition=dict(duration=0))]
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
            )
        ]
    )]
    
    sliders = [dict(
        active=0,
        steps=[],
        x=0.09, y=0.0,
        len=0.91,
        xanchor="left", yanchor="top",
        pad=dict(t=25, l=10),
        currentvalue=dict(visible=True, prefix="Mission Step: ", xanchor="right", font=dict(size=12, color="white")),
        transition=dict(duration=0)
    )]
    
    for f in plotly_frames:
        step_val = f["step"]
        slider_step = dict(
            args=[[f"frame_{step_val}"], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
            label=str(step_val),
            method="animate"
        )
        sliders[0]["steps"].append(slider_step)
        
    fig3d.update_layout(
        title=dict(
            text="<b>Multi-Drone Autonomous Coverage — Synchronized 2D & 3D Interactive Web Dashboard</b>",
            x=0.5, y=0.97,
            xanchor="center", yanchor="top",
            font=dict(size=18, color="white", family="Inter, Roboto, Arial")
        ),
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="X (meters)", range=[0, 100], gridcolor="#2c2c4d", backgroundcolor="#0e0e22", showbackground=True),
            yaxis=dict(title="Y (meters)", range=[0, 100], gridcolor="#2c2c4d", backgroundcolor="#0e0e22", showbackground=True),
            zaxis=dict(title="Z Altitude (m)", range=[0, 15], gridcolor="#2c2c4d", backgroundcolor="#0f0f2a", showbackground=True),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.35)
        ),
        xaxis=dict(title="X (meters)", range=[0, 100], gridcolor="#2c2c4d", scaleanchor="y"),
        yaxis=dict(title="Y (meters)", range=[0, 100], gridcolor="#2c2c4d"),
        margin=dict(l=40, r=40, b=100, t=80),
        updatemenus=updatemenus,
        sliders=sliders,
        legend=dict(
            x=1.02, y=0.98,
            bgcolor="rgba(19, 19, 40, 0.85)",
            bordercolor="#3a3a6a",
            borderwidth=1,
            font=dict(size=9)
        )
    )
    
    HTML_OUT = os.path.join(os.path.dirname(__file__), "models", "interactive_3d_view.html")
    fig3d.write_html(HTML_OUT, include_plotlyjs="cdn")
    print(f"[DONE] Synchronized Interactive 2D/3D HTML saved to {HTML_OUT}  ({os.path.getsize(HTML_OUT)/1024:.1f} KB)")
except Exception as e:
    import traceback
    print(f"[WARN] Failed to write interactive 3D twin: {e}")
    traceback.print_exc()

print("\n[DONE] All results complete.")
