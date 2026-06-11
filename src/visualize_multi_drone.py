# src/visualize_multi_drone.py
"""
Live 2D visualiser + video exporter for the integrated FA+RA+PPO multi-drone system.

Panels:
  Top-left  : 2D top-down coverage map (drones, paths, obstacles, recharge stations)
  Top-right : 3 multi-objective metrics over time (coverage, collisions, battery)
  Bottom-left : Per-drone battery bars (live)
  Bottom-right: Phase status + wind indicator per drone

Usage:
    python src/visualize_multi_drone.py              # live + save video
    python src/visualize_multi_drone.py --no-video   # live only
    python src/visualize_multi_drone.py --steps 1000
"""

import os
import sys
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; switched to TkAgg if display available
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from collections import deque

# -- Make local imports work regardless of CWD ----------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from multi_drone_coverage_env import MultiDroneCoverageEnv, DRONE_COLORS

# -- Constants ------------------------------------------------------------
REPLAN_INTERVAL   = 150
N_WAYPOINTS_FA    = 12
N_WAYPOINTS_RA    = 8
MODEL_PATH        = os.path.join(SRC_DIR, "..", "models", "ppo_multi_drone_final.zip")
VEC_PATH          = os.path.join(SRC_DIR, "..", "models", "multi_drone_vecnorm.pkl")
DEFAULT_VIDEO_OUT = os.path.join(SRC_DIR, "..", "models", "drone_coverage.mp4")
FA_WP_PATH        = os.path.join(SRC_DIR, "..", "dataset", "fa_waypoints.npy")


# ==========================================================================
def _try_load_ppo():
    """Try to load PPO model+vecenv. Returns (model, venv) or (None, None)."""
    if not os.path.exists(MODEL_PATH):
        print("[INFO] PPO model not found — using waypoint-following controller")
        return None, None
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        raw_env = DummyVecEnv([lambda: MultiDroneCoverageEnv(n_drones=6)])
        venv    = VecNormalize.load(VEC_PATH, raw_env) if os.path.exists(VEC_PATH) else raw_env
        model   = PPO.load(MODEL_PATH, env=venv)
        print("[INFO] PPO model loaded [OK]")
        return model, venv
    except Exception as e:
        print(f"[WARN] Could not load PPO: {e}  — using waypoint controller")
        return None, None


def _try_load_fa_waypoints(n_drones):
    """Try to load pre-computed FA waypoints."""
    if os.path.exists(FA_WP_PATH):
        wp = np.load(FA_WP_PATH)
        if wp.shape[0] == n_drones:
            print(f"[INFO] Loaded FA waypoints from {FA_WP_PATH}")
            return wp
    return None


def _waypoint_action(positions, waypoints, wp_idx, max_speed=0.9, n_drones=6):
    """Simple proportional waypoint-following action."""
    action = np.zeros((n_drones, 3), dtype=float)
    for i in range(n_drones):
        idx = min(wp_idx[i], waypoints.shape[1] - 1)
        target = waypoints[i, idx]
        diff   = target - positions[i]
        dist   = np.linalg.norm(diff) + 1e-9
        vel    = (diff / dist) * min(dist, max_speed)
        action[i] = np.clip(vel, -1.0, 1.0)
    return action.flatten()


# ==========================================================================
def run_and_capture(
    max_steps: int   = 2000,
    fa_iterations: int = 40,
    ra_iterations: int = 25,
    use_ppo: bool   = True,
):
    """
    Run the integrated FA+RA+PPO system and capture all frames for video export.
    Returns: (env, all_frames_data) where all_frames_data is a list of dicts.
    """
    # -- Build environment -------------------------------------------------
    env = MultiDroneCoverageEnv(
        n_drones=6,
        wind_enabled=True,
        thermal_enabled=True,
        sensor_noise_std=0.04,
    )
    obs, _ = env.reset()
    n_drones = env.n_drones

    # -- Load or generate FA waypoints ------------------------------------
    waypoints = _try_load_fa_waypoints(n_drones)
    if waypoints is None:
        print(f"[FA] Running Firefly Algorithm ({fa_iterations} iters)...")
        try:
            from fa_coverage import FireflyPlanner
            fa = FireflyPlanner(
                n_drones=n_drones,
                n_waypoints=N_WAYPOINTS_FA,
                n_fireflies=25,
                max_iter=fa_iterations,
            )
            fa.optimize(verbose=True)
            waypoints = fa.get_best_waypoints()
            stats = fa.get_coverage_stats()
            print(f"[FA] Done — estimated coverage: {stats['coverage_ratio']*100:.1f}%")
        except Exception as e:
            print(f"[FA] Failed ({e}) — using grid waypoints")
            waypoints = _grid_waypoints(n_drones, env.GX, env.GY, env.CRUISE_Z, N_WAYPOINTS_FA)

    # -- Optional: RA replanner --------------------------------------------
    try:
        from ra_coverage import RavenReplanner
        ra = RavenReplanner(n_drones=n_drones, max_iter=ra_iterations)
        ra_available = True
        print("[RA] Raven Replanner ready [OK]")
    except Exception as e:
        print(f"[RA] Not available ({e}) — skipping adaptive replanning")
        ra = None
        ra_available = False

    # -- Optional: PPO ----------------------------------------------------
    model, venv = (None, None)
    if use_ppo:
        model, venv = _try_load_ppo()

    ppo_obs = None
    if model is not None and venv is not None:
        ppo_obs = venv.reset()

    # -- Waypoint tracking -------------------------------------------------
    wp_idx      = np.zeros(n_drones, dtype=int)
    wp_advance_thresh = 3.5   # units; advance waypoint when this close

    # -- History containers ------------------------------------------------
    drone_paths  = [[] for _ in range(n_drones)]
    frames_data  = []

    coverage_hist   = []
    collision_hist  = []
    battery_hist    = []
    path_hist       = []
    wind_hist       = []

    print(f"\n[RUN] Starting integrated mission — {max_steps} steps, {n_drones} drones")
    print("=" * 60)

    for step in range(max_steps):

        # -- RA adaptive replanning ----------------------------------------
        if ra_available and step > 0 and step % REPLAN_INTERVAL == 0:
            print(f"  [RA] Replanning at step {step} "
                  f"(coverage={env.coverage_ratio()*100:.1f}%)")
            try:
                new_wp = ra.replan(
                    current_positions=env.positions.copy(),
                    coverage_grid=env.coverage_grid.copy(),
                    n_remaining_waypoints=N_WAYPOINTS_RA,
                )
                waypoints = np.concatenate(
                    [waypoints[:, :wp_idx.min()],   # keep completed part
                     new_wp], axis=1
                )[:, :N_WAYPOINTS_FA]
                wp_idx = np.clip(wp_idx, 0, waypoints.shape[1] - 1)
            except Exception as e:
                print(f"  [RA] Replan failed: {e}")

        # -- Determine action ----------------------------------------------
        if model is not None and ppo_obs is not None:
            action_raw, _ = model.predict(ppo_obs, deterministic=True)
            action = action_raw.flatten() if hasattr(action_raw, 'flatten') else action_raw
        else:
            action = _waypoint_action(
                env.positions, waypoints, wp_idx,
                max_speed=env.max_speed, n_drones=n_drones
            )

        # -- Step environment ----------------------------------------------
        obs, reward, terminated, truncated, info = env.step(action)

        if model is not None and venv is not None:
            try:
                ppo_obs, _, _, _ = venv.step(
                    action.reshape(1, -1)
                )
            except Exception:
                pass

        # -- Advance waypoint indices --------------------------------------
        for i in range(n_drones):
            idx = min(wp_idx[i], waypoints.shape[1] - 1)
            if np.linalg.norm(env.positions[i] - waypoints[i, idx]) < wp_advance_thresh:
                wp_idx[i] = min(wp_idx[i] + 1, waypoints.shape[1] - 1)

        # -- Record frame data ---------------------------------------------
        for i in range(n_drones):
            drone_paths[i].append(env.positions[i].copy())

        coverage_hist.append(info["coverage_pct"])
        collision_hist.append(info["total_collisions"])
        battery_hist.append(info["total_battery_used"])
        path_hist.append(info["total_path_length"])
        wind_hist.append(info["wind_speed"])

        # Snapshot for animation
        frames_data.append({
            "step":         step,
            "positions":    env.positions.copy(),
            "coverage":     env.coverage_grid.copy(),
            "batteries":    np.array(info["batteries"]),
            "phases":       list(info["phases"]),
            "motor_health": np.array(info["motor_health"]),
            "dyn_pos":      [p.copy() for p in env.dyn.positions] if env.dyn.positions else [],
            "coverage_pct": info["coverage_pct"],
            "collisions":   info["total_collisions"],
            "battery_used": info["total_battery_used"],
            "path_length":  info["total_path_length"],
            "wind_speed":   info["wind_speed"],
            "drone_paths":  [list(drone_paths[i]) for i in range(n_drones)],
            "waypoints":    waypoints.copy(),
            "wp_idx":       wp_idx.copy(),
        })

        if step % 100 == 0:
            print(f"  step={step:4d} | coverage={info['coverage_pct']:5.1f}% "
                  f"| collisions={info['total_collisions']:3d} "
                  f"| battery_used={info['total_battery_used']:6.1f} "
                  f"| wind={info['wind_speed']:.3f}")

        if terminated or truncated:
            print(f"\n[RUN] Episode ended at step {step}: "
                  f"coverage={info['coverage_pct']:.1f}%")
            break

    print("\n[FINAL STATS]")
    last = frames_data[-1]
    print(f"  Coverage      : {last['coverage_pct']:.2f}%")
    print(f"  Total collisions : {last['collisions']}")
    print(f"  Total battery used: {last['battery_used']:.1f}")
    print(f"  Total path length : {last['path_length']:.1f}")

    return env, frames_data, coverage_hist, collision_hist, battery_hist, path_hist


# ==========================================================================
def _grid_waypoints(n_drones, GX, GY, cruise_z, n_waypoints):
    """Fallback: divide map into strips, one per drone."""
    waypoints = np.zeros((n_drones, n_waypoints, 3))
    strip_w   = GX / n_drones
    for i in range(n_drones):
        x_center = strip_w * i + strip_w / 2
        for k in range(n_waypoints):
            frac = k / max(n_waypoints - 1, 1)
            y    = frac * (GY - 4) + 2
            x    = x_center + ((-1) ** k) * strip_w * 0.3
            waypoints[i, k] = [np.clip(x, 1, GX-1), y, cruise_z]
    return waypoints


# ==========================================================================
def save_video(
    env,
    frames_data,
    coverage_hist,
    collision_hist,
    battery_hist,
    path_hist,
    video_path: str = DEFAULT_VIDEO_OUT,
    fps: int = 12,
    frame_skip: int = 3,    # render every N-th frame to keep file size manageable
):
    """
    Render all captured frames into a 2D video (MP4 or GIF fallback).

    Layout (2x2):
      [0,0] Top-down coverage map   [0,1] Multi-objective metrics
      [1,0] Per-drone battery bars  [1,1] Drone status table
    """
    n_drones = env.n_drones
    GX, GY   = env.GX, env.GY

    # Sample frames
    render_frames = frames_data[::frame_skip]
    total_frames  = len(render_frames)
    print(f"[VIDEO] Rendering {total_frames} frames -> {video_path}")

    fig = plt.figure(figsize=(18, 11), facecolor="#0d0d1a")
    fig.suptitle("Multi-Drone Coverage  |  FA + RA + PPO  |  Integrated System",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                   left=0.06, right=0.97, top=0.93, bottom=0.06)
    ax_map  = fig.add_subplot(gs[0, 0])
    ax_met  = fig.add_subplot(gs[0, 1])
    ax_bat  = fig.add_subplot(gs[1, 0])
    ax_stat = fig.add_subplot(gs[1, 1])

    for ax in [ax_map, ax_met, ax_bat, ax_stat]:
        ax.set_facecolor("#131328")
        for spine in ax.spines.values():
            spine.set_color("#3a3a6a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Recharge station markers (static)
    from multi_drone_coverage_env import RECHARGE_STATIONS_XY
    rs_xy = np.array(RECHARGE_STATIONS_XY, dtype=float)

    def animate(frame_idx):
        fd = render_frames[frame_idx]

        # -- [0,0] Top-down coverage map -----------------------------------
        ax_map.clear()
        ax_map.set_facecolor("#131328")

        cov_img = fd["coverage"].astype(float)
        ax_map.imshow(
            cov_img.T, origin="lower", cmap="YlGn",
            vmin=0, vmax=1, alpha=0.85,
            extent=[0, GX, 0, GY], aspect="auto"
        )

        # Terrain contour (subtle)
        if hasattr(env, "terrain"):
            X_t = np.arange(GX); Y_t = np.arange(GY)
            ax_map.contour(
                X_t, Y_t, env.terrain.T,
                levels=4, colors=["#334455"], linewidths=0.4, alpha=0.5
            )

        # Recharge stations
        ax_map.scatter(rs_xy[:, 0], rs_xy[:, 1], s=120, marker="s",
                       c="magenta", edgecolors="white", linewidths=0.8,
                       zorder=5, label="Recharge")

        # Dynamic obstacles
        if fd["dyn_pos"]:
            dp = np.array(fd["dyn_pos"])
            ax_map.scatter(dp[:, 0], dp[:, 1], s=55, c="royalblue",
                           alpha=0.7, edgecolors="white", linewidths=0.5,
                           zorder=4, marker="o")

        # Drone paths + current positions
        for i in range(n_drones):
            col  = DRONE_COLORS[i]
            path = fd["drone_paths"][i]
            if len(path) > 1:
                pts  = np.array(path)
                ax_map.plot(pts[:, 0], pts[:, 1], color=col,
                            linewidth=1.0, alpha=0.55, zorder=3)
            pos = fd["positions"][i]
            ax_map.scatter(pos[0], pos[1], s=90, c=col,
                           edgecolors="white", linewidths=0.8,
                           zorder=6, marker="^")
            ax_map.annotate(
                f"D{i}", (pos[0], pos[1]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=6.5, color=col
            )

            # Next waypoint target
            wp_i  = fd["wp_idx"][i]
            wp_pt = fd["waypoints"][i, min(wp_i, fd["waypoints"].shape[1]-1)]
            ax_map.scatter(wp_pt[0], wp_pt[1], s=40, marker="x",
                           c=col, linewidths=1.2, alpha=0.7, zorder=5)

        ax_map.set_xlim(0, GX); ax_map.set_ylim(0, GY)
        ax_map.set_xlabel("X (m)", fontsize=8)
        ax_map.set_ylabel("Y (m)", fontsize=8)
        ax_map.set_title(
            f"Step {fd['step']:4d}  |  Coverage: {fd['coverage_pct']:.1f}%"
            f"  |  Wind: {fd['wind_speed']:.3f}",
            fontsize=9
        )

        # -- [0,1] Multi-objective metrics ---------------------------------
        ax_met.clear()
        ax_met.set_facecolor("#131328")
        t_end = fd["step"] + 1

        xs = range(min(t_end, len(coverage_hist)))

        # Coverage (primary axis, left)
        ax_met.plot(list(xs), coverage_hist[:t_end],
                    color="#00ff88", linewidth=1.4, label="Coverage %")
        ax_met.set_ylabel("Coverage %", color="#00ff88", fontsize=8)
        ax_met.tick_params(axis="y", labelcolor="#00ff88")
        ax_met.set_ylim(0, 105)

        ax2 = ax_met.twinx()
        ax2.set_facecolor("#131328")

        # Collisions (right axis, red dashed)
        ax2.plot(list(xs), collision_hist[:t_end],
                 color="#ff4455", linewidth=1.1, linestyle="--", label="Collisions")
        # Battery used (right axis, orange dotted)
        bat_norm = [b / max(battery_hist[-1], 1) * 100 for b in battery_hist[:t_end]]
        ax2.plot(list(xs), bat_norm,
                 color="#ffaa00", linewidth=1.1, linestyle=":", label="Battery %")
        # Path length (right axis, cyan dash-dot)
        path_norm = [p / max(path_hist[-1], 1) * 100 for p in path_hist[:t_end]]
        ax2.plot(list(xs), path_norm,
                 color="#44ccff", linewidth=1.0, linestyle="-.", label="Path %")

        ax2.set_ylabel("Normalised (0-100)", color="white", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="white")
        ax2.set_ylim(0, 110)

        lines1, labels1 = ax_met.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_met.legend(lines1 + lines2, labels1 + labels2,
                      loc="upper left", fontsize=6.5,
                      facecolor="#1a1a35", labelcolor="white", framealpha=0.8)
        ax_met.set_xlabel("Step", fontsize=8)
        ax_met.set_title("Multi-Objective Metrics", fontsize=9)
        for spine in ax2.spines.values():
            spine.set_color("#3a3a6a")
        ax2.tick_params(colors="white")

        # -- [1,0] Per-drone battery bars ----------------------------------
        ax_bat.clear()
        ax_bat.set_facecolor("#131328")

        bats = fd["batteries"]
        for i in range(n_drones):
            pct = bats[i] / env.battery_capacity * 100
            bar_color = (
                "#00ff88" if pct > 50 else
                "#ffaa00" if pct > 20 else
                "#ff4455"
            )
            ax_bat.barh(i, pct, color=bar_color, alpha=0.85, height=0.6,
                        edgecolor="white", linewidth=0.5)
            ax_bat.text(pct + 0.5, i, f"{pct:.0f}%",
                        va="center", fontsize=7, color="white")

        ax_bat.set_xlim(0, 110)
        ax_bat.set_yticks(range(n_drones))
        ax_bat.set_yticklabels([f"Drone {i}" for i in range(n_drones)],
                               fontsize=7.5, color="white")
        ax_bat.set_xlabel("Battery %", fontsize=8)
        ax_bat.set_title("Per-Drone Battery", fontsize=9)
        ax_bat.axvline(20, color="#ff4455", linestyle="--", linewidth=0.8, alpha=0.7)

        # -- [1,1] Status table --------------------------------------------
        ax_stat.clear()
        ax_stat.set_facecolor("#131328")
        ax_stat.axis("off")

        phase_icon = {"ascend": "^ ASCEND", "cruise": "-> CRUISE",
                      "recharge_descend": "v RECHARGE", "landed": "? LANDED"}

        col_labels = ["Drone", "Phase", "Battery", "Motor", "Path"]
        row_data   = []
        for i in range(n_drones):
            ph  = fd["phases"][i]
            mh  = fd["motor_health"][i]
            bat = fd["batteries"][i]
            row_data.append([
                f"D{i}",
                phase_icon.get(ph, ph),
                f"{bat:.0f}",
                f"{mh*100:.0f}%",
                f"{fd['drone_paths'][i] and len(fd['drone_paths'][i]) or 0}",
            ])

        tbl = ax_stat.table(
            cellText=row_data, colLabels=col_labels,
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.55)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#1a1a35" if r > 0 else "#2a2a55")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#3a3a6a")
            if r > 0:
                i = r - 1
                ph = fd["phases"][i]
                if ph == "ascend":
                    cell.set_facecolor("#1a2a1a")
                elif ph == "recharge_descend":
                    cell.set_facecolor("#2a1a1a")

        ax_stat.set_title(
            f"Obj 1: Collisions={fd['collisions']:3d} | "
            f"Obj 2: Path={fd['path_length']:.0f} | "
            f"Obj 3: Bat={fd['battery_used']:.0f}",
            fontsize=8
        )

    anim = FuncAnimation(
        fig, animate, frames=total_frames,
        interval=1000 // fps, blit=False
    )

    # -- Save video (MP4 with ffmpeg, fallback to GIF) ---------------------
    saved_path = video_path
    try:
        writer = FFMpegWriter(fps=fps, metadata={"title": "Drone Coverage"}, bitrate=1800)
        anim.save(video_path, writer=writer, dpi=110)
        print(f"\n[VIDEO] [OK] Saved MP4 -> {os.path.abspath(video_path)}")
    except Exception as e:
        print(f"[VIDEO] ffmpeg not available ({e}). Saving as GIF...")
        gif_path = video_path.replace(".mp4", ".gif")
        saved_path = gif_path
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=85)
        print(f"[VIDEO] [OK] Saved GIF  -> {os.path.abspath(gif_path)}")

    plt.close(fig)
    return saved_path


# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Drone FA+RA+PPO Visualiser")
    parser.add_argument("--steps",    type=int,  default=2000)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fa-iter",  type=int,  default=40)
    parser.add_argument("--ra-iter",  type=int,  default=25)
    parser.add_argument("--no-ppo",   action="store_true")
    parser.add_argument("--fps",      type=int,  default=12)
    parser.add_argument("--skip",     type=int,  default=3,
                        help="Render every N-th frame (reduces file size)")
    parser.add_argument("--output",   type=str,  default=DEFAULT_VIDEO_OUT)
    args = parser.parse_args()

    env, frames_data, cov_h, col_h, bat_h, path_h = run_and_capture(
        max_steps=args.steps,
        fa_iterations=args.fa_iter,
        ra_iterations=args.ra_iter,
        use_ppo=(not args.no_ppo),
    )

    if not args.no_video and frames_data:
        saved = save_video(
            env, frames_data, cov_h, col_h, bat_h, path_h,
            video_path=args.output,
            fps=args.fps,
            frame_skip=args.skip,
        )
        print(f"\n[DONE] Video saved to: {os.path.abspath(saved)}")
    else:
        print("\n[DONE] No video saved (--no-video flag set or no frames captured).")


if __name__ == "__main__":
    main()
