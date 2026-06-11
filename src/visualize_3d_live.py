# src/visualize_3d_live.py
"""
Visualizer for Drone3DEnv (3D + 2D top-down)
- Works with Stable-Baselines3 VecNormalize (VecEnv)
- Handles VecEnv step return (obs, reward, done, info)
- Stops on goal_reached and snaps final position into path
- Shows recharge stations in both 3D and 2D
- Red safety bubble only when very near static trees
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_env_3d import Drone3DEnv


# -------------------------
# Draw helpers
# -------------------------
def draw_cylinder(ax, x, y, z0, z1, radius=0.30, color="saddlebrown"):
    theta = np.linspace(0, 2 * np.pi, 18)
    z = np.linspace(z0, z1, 2)
    TH, ZZ = np.meshgrid(theta, z)
    X = x + radius * np.cos(TH)
    Y = y + radius * np.sin(TH)
    ax.plot_surface(X, Y, ZZ, color=color, linewidth=0, antialiased=True)


def draw_sphere(ax, x, y, z, radius=0.8, color="green", alpha=1.0):
    u = np.linspace(0, np.pi, 20)
    v = np.linspace(0, 2 * np.pi, 20)
    U, V = np.meshgrid(u, v)
    X = x + radius * np.sin(U) * np.cos(V)
    Y = y + radius * np.sin(U) * np.sin(V)
    Z = z + radius * np.cos(U)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def draw_translucent_sphere(ax, x, y, z, radius, rgba):
    u = np.linspace(0, np.pi, 20)
    v = np.linspace(0, 2 * np.pi, 20)
    U, V = np.meshgrid(u, v)
    X = x + radius * np.sin(U) * np.cos(V)
    Y = y + radius * np.sin(U) * np.sin(V)
    Z = z + radius * np.cos(U)
    ax.plot_surface(X, Y, Z, color=rgba[:3], alpha=rgba[3], linewidth=0)


def draw_recharge_station(ax, x, y, z=3.5):
    draw_cylinder(ax, x, y, z, z + 1.2, radius=0.8, color="magenta")
    draw_sphere(ax, x, y, z + 1.3, radius=1.0, color="magenta", alpha=0.85)


# -------------------------
# RL loader
# -------------------------
def prepare_rl():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_drone_final.zip")
    vec_path = os.path.join(os.path.dirname(__file__), "..", "models", "vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model missing: run train_rl_3d.py first")

    def make_env():
        return Drone3DEnv(demo_mode=False)

    raw_env = DummyVecEnv([make_env])
    # Load VecNormalize wrapper and model
    venv = VecNormalize.load(vec_path, raw_env)
    model = PPO.load(model_path, env=venv)

    real_env = venv.venv.envs[0]
    return venv, model, real_env


# -------------------------
# Visualizer
# -------------------------
def visualize(max_frames=2000):
    venv, model, env = prepare_rl()
    obs = venv.reset()  # returns normalized obs for vecenv (shape: (n_envs, obs_dim))

    terrain = env.terrain
    static = env.static
    grid = env.grid

    fig = plt.figure(figsize=(14, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    drone_path = []

    # Precompute static tree XY centers (for quick min-distance checks)
    static_idx = np.argwhere(static)
    static_xy = (static_idx[:, :2].astype(float) + 0.5) if static_idx.size > 0 else np.zeros((0, 2))

    # Terrain mesh used by 3D and 2D
    Xg, Yg = np.meshgrid(np.arange(terrain.shape[0]), np.arange(terrain.shape[1]))
    Zimg = terrain.T

    for frame in range(max_frames):
        # Clear axes
        ax3d.clear()
        ax2d.clear()

        # 3D terrain surface
        ax3d.plot_surface(Xg, Yg, Zimg, cmap="terrain", linewidth=0, alpha=0.9)

        # Draw static trees in 3D
        sx, sy, sz = static.shape
        for xi in range(sx):
            for yi in range(sy):
                col = static[xi, yi, :]
                if np.any(col):
                    z_idx = np.where(col)[0]
                    z0 = z_idx[0]
                    z1 = z_idx[-1]
                    draw_cylinder(ax3d, xi, yi, z0, z1)
                    draw_sphere(ax3d, xi, yi, z1 + 0.6, radius=0.9, color="forestgreen")

        # Draw dynamic obstacles in 3D
        for (px, py, pz) in env.dyn.positions:
            draw_sphere(ax3d, px, py, pz, radius=0.55, color="royalblue")

        # Draw recharge stations in 3D
        for rs in env.recharge_positions_3d:
            draw_recharge_station(ax3d, rs[0], rs[1], rs[2])
            ax3d.text(rs[0], rs[1], rs[2] + 2.0, "R", color="magenta", ha="center")

        # Append the current drone position to path
        drone_path.append(env.pos.copy())

        # 3D path line (red)
        if len(drone_path) > 1:
            pts = np.array(drone_path)
            ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-r", linewidth=2)

        # Compute min XY distance to static trees (for red safety bubble condition)
        if static_xy.size > 0:
            dxy = np.linalg.norm(static_xy - env.pos[:2], axis=1)
            min_static_xy = float(np.min(dxy))
        else:
            min_static_xy = float("inf")

        R = env.SAFETY_RADIUS

        # Color logic: red only when near static trees
        if min_static_xy < R:
            rgba = (1.0, 0.0, 0.0, 0.25)
            drone_color = "red"
        elif env._min_distance(env.pos) > R * 1.5:
            rgba = (0.0, 1.0, 0.0, 0.15)
            drone_color = "yellow"
        else:
            rgba = (1.0, 0.65, 0.0, 0.18)
            drone_color = "orange"

        draw_translucent_sphere(ax3d, env.pos[0], env.pos[1], env.pos[2], R, rgba)
        draw_sphere(ax3d, env.pos[0], env.pos[1], env.pos[2], radius=0.6, color=drone_color)

        # Final goal marker in 3D
        gx, gy = env.goal_xy
        ix = int(gx - 0.5)
        iy = int(gy - 0.5)
        gz = terrain[ix, iy]
        draw_sphere(ax3d, gx, gy, gz + 1.2, radius=1.1, color="darkgreen")
        ax3d.text(gx, gy, gz + 2.2, "FINAL GOAL", color="green", ha="center")

        # ------- 2D top-down view -------
        ax2d.imshow(Zimg, cmap="gray", origin="lower", alpha=0.9)

        # Static trees 2D
        if static_xy.size > 0:
            ax2d.scatter(static_xy[:, 0], static_xy[:, 1], s=12, c="darkgreen", label="Static trees")

        # Dynamic obstacles 2D
        dpos = np.array(env.dyn.positions)
        if dpos.size > 0:
            ax2d.scatter(dpos[:, 0], dpos[:, 1], s=40, facecolors="none", edgecolors="royalblue", label="Dynamic obstacles")

        # Recharge stations 2D (magenta squares) -- add label only once
        rs_label_added = False
        for rs in env.recharge_positions_3d:
            if not rs_label_added:
                ax2d.scatter(rs[0], rs[1], s=160, marker="s", c="magenta", edgecolors="k", label="Recharge station")
                rs_label_added = True
            else:
                ax2d.scatter(rs[0], rs[1], s=160, marker="s", c="magenta", edgecolors="k")

        # Drone 2D marker
        ax2d.scatter(env.pos[0], env.pos[1], s=120, facecolors="yellow", edgecolors="k", label="Drone")

        # Path 2D
        path2 = np.array(drone_path)
        ax2d.plot(path2[:, 0], path2[:, 1], "-r", linewidth=2, label="Drone path")

        # Final goal 2D
        ax2d.scatter(gx, gy, s=120, c="green", marker="*", edgecolors="k", label="Final goal")

        # Legend: top-left
        ax2d.legend(loc="upper left", framealpha=0.95)

        # Axes, labels
        ax3d.set_xlim(0, grid[0])
        ax3d.set_ylim(0, grid[1])
        ax3d.set_zlim(0, grid[2])
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title(f"3D View | Frame {frame}")

        ax2d.set_xlim(0, grid[0])
        ax2d.set_ylim(0, grid[1])
        ax2d.set_xlabel("X")
        ax2d.set_ylabel("Y")
        ax2d.set_title("Top-Down Path View")

        battery_pct = (env.battery / env.battery_capacity) * 100
        fig.suptitle(f"3D RL Drone Navigation | Battery {battery_pct:.1f}% | Phase={env.phase}", fontsize=14)

        plt.pause(0.001)

        # -------------- RL STEP (VecEnv returns 4 values) --------------
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)

        # Normalize done/info for single-env visualizer
        if isinstance(done, (list, tuple, np.ndarray)):
            done0 = bool(done[0])
        else:
            done0 = bool(done)

        if isinstance(info, (list, tuple)) and len(info) > 0:
            info0 = info[0]
        elif isinstance(info, dict):
            info0 = info
        else:
            info0 = {}

        # If episode finished due to reaching goal, snap final pos and stop
        if done0 and info0.get("goal_reached", False):
            drone_path.append(env.pos.copy())
            print("[visualizer] Goal reached; stopping.")
            break

        # Normal termination/truncation
        if done0:
            print("[visualizer] Episode finished:", info0)
            break

    plt.show()


if __name__ == "__main__":
    visualize()
