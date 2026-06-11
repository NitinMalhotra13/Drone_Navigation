"""
run_integrated.py
=================
MAIN integrated runner combining three planning layers into a unified
hierarchical control system for multi-drone area coverage:

    +-------------------------------------------------------------+
    |  FA  (Firefly Algorithm)  –  Global planner                 |
    |      Runs ONCE at episode start -> initial waypoints (nx12x3)|
    +-------------------------------------------------------------+
    |  RA  (Raven Replanner)    –  Adaptive mid-level planner     |
    |      Runs every REPLAN_INTERVAL steps -> updated waypoints   |
    +-------------------------------------------------------------+
    |  PPO (Proximal Policy Optimisation) – Low-level controller  |
    |      Runs every step -> action toward current target waypoint|
    +-------------------------------------------------------------+

Usage:
    python run_integrated.py --steps 2000
    python run_integrated.py --steps 500 --no-video --no-ppo
    python run_integrated.py --steps 3000 --fa-iter 60 --ra-iter 30
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------------------------------------------------------------------------
# Ensure all local src modules are importable regardless of working directory
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SRC_DIR, "..")
for _p in (_SRC_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multi_drone_coverage_env import MultiDroneCoverageEnv  # noqa: E402
from firefly_planner import FireflyPlanner                  # noqa: E402
from raven_replanner import RavenReplanner                  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPLAN_INTERVAL: int = 150          # RA replanning cadence (steps)
N_WAYPOINTS_FA: int = 12            # waypoints per drone from FA
N_WAYPOINTS_RA: int = 8             # waypoints per drone from RA
N_DRONES: int = 6

DRONE_COLORS: list[str] = ["red", "cyan", "lime", "orange", "magenta", "yellow"]

# Absolute paths derived from script location
MODEL_PATH: str = os.path.join(_SRC_DIR, "..", "models", "ppo_multi_drone_final.zip")
VEC_PATH:   str = os.path.join(_SRC_DIR, "..", "models", "multi_drone_vecnorm.pkl")
FA_WAYPOINTS_CACHE: str = os.path.join(_SRC_DIR, "..", "dataset", "fa_waypoints.npy")

# Normalise to OS-native paths
MODEL_PATH = os.path.normpath(MODEL_PATH)
VEC_PATH   = os.path.normpath(VEC_PATH)
FA_WAYPOINTS_CACHE = os.path.normpath(FA_WAYPOINTS_CACHE)


# ---------------------------------------------------------------------------
# PPO loader
# ---------------------------------------------------------------------------

def load_ppo() -> tuple[PPO, VecNormalize, MultiDroneCoverageEnv]:
    """
    Load the trained PPO model together with its ``VecNormalize`` wrapper
    and a raw (unwrapped) ``MultiDroneCoverageEnv`` for direct inspection.

    Returns
    -------
    model : PPO
    venv  : VecNormalize  (the normalised vector environment)
    raw_env : MultiDroneCoverageEnv  (inner unwrapped env for attribute access)

    Raises
    ------
    FileNotFoundError
        If the model zip or VecNormalize pickle cannot be located.
    """
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"[PPO] Trained model not found at:\n  {MODEL_PATH}\n"
            "Run `python train_multi_drone_ppo.py` first, or pass --no-ppo."
        )
    if not os.path.isfile(VEC_PATH):
        raise FileNotFoundError(
            f"[PPO] VecNormalize stats not found at:\n  {VEC_PATH}\n"
            "Run `python train_multi_drone_ppo.py` first, or pass --no-ppo."
        )

    raw_env = MultiDroneCoverageEnv(n_drones=N_DRONES, randomize_static=False)
    venv = DummyVecEnv([lambda: raw_env])
    venv = VecNormalize.load(VEC_PATH, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(MODEL_PATH, env=venv)
    print(f"[PPO] Model loaded from      : {MODEL_PATH}")
    print(f"[PPO] VecNormalize loaded from: {VEC_PATH}")
    return model, venv, raw_env


# ---------------------------------------------------------------------------
# Main integrated runner
# ---------------------------------------------------------------------------

def run_integrated(
    max_steps: int = 2000,
    save_video: bool = True,
    video_path: str = os.path.normpath(
        os.path.join(_SRC_DIR, "..", "models", "drone_coverage.mp4")
    ),
    fa_iterations: int = 40,
    ra_iterations: int = 25,
    use_ppo: bool = True,
) -> None:
    """
    Execute one full episode of the integrated FA + RA + PPO pipeline and
    optionally render an animation of the coverage run.

    Parameters
    ----------
    max_steps    : int   – maximum environment steps before forced termination
    save_video   : bool  – whether to save an animation to ``video_path``
    video_path   : str   – output path for the mp4/gif artefact
    fa_iterations: int   – FA optimisation iterations
    ra_iterations: int   – RA optimisation iterations per replan call
    use_ppo      : bool  – whether to use the PPO low-level controller;
                           if False or model missing, falls back to pure
                           waypoint-following with proportional control
    """

    # ---------------------------------------------------------------
    # 1. Setup environment
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("  Integrated FA + RA + PPO Runner")
    print("="*60)

    model:   Optional[PPO]              = None
    venv:    Optional[VecNormalize]     = None
    raw_env: Optional[MultiDroneCoverageEnv] = None

    if use_ppo:
        try:
            model, venv, raw_env = load_ppo()
            env = venv          # use normalised env for PPO predictions
            obs, info = env.reset()
        except FileNotFoundError as exc:
            print(f"[Warning] {exc}")
            print("[Warning] Falling back to waypoint-following without PPO.\n")
            use_ppo = False

    if not use_ppo:
        raw_env = MultiDroneCoverageEnv(n_drones=N_DRONES, randomize_static=False)
        env     = raw_env
        obs, info = env.reset()

    n_drones: int = raw_env.n_drones  # type: ignore[union-attr]

    # ---------------------------------------------------------------
    # 2. FA – initial global waypoint plan
    # ---------------------------------------------------------------
    fa_waypoints: np.ndarray  # shape (n_drones, N_WAYPOINTS_FA, 3)

    if os.path.isfile(FA_WAYPOINTS_CACHE):
        answer = input(
            f"\n[FA] Cached waypoints found at:\n  {FA_WAYPOINTS_CACHE}\n"
            "Load them? [y/N]: "
        ).strip().lower()
        if answer == "y":
            fa_waypoints = np.load(FA_WAYPOINTS_CACHE)
            print(f"[FA] Loaded cached waypoints, shape: {fa_waypoints.shape}")
        else:
            fa_waypoints = _run_fa(raw_env, fa_iterations)
    else:
        fa_waypoints = _run_fa(raw_env, fa_iterations)

    # ---------------------------------------------------------------
    # 3. Initialise per-drone state
    # ---------------------------------------------------------------
    # Active waypoint plan (updated by RA); starts as FA output
    active_waypoints: np.ndarray = fa_waypoints.copy()  # (n, n_wp, 3)

    current_waypoint_idx: list[int] = [0] * n_drones
    drone_paths: list[list[np.ndarray]] = [[] for _ in range(n_drones)]

    # Metric histories (one value per step)
    coverage_history:     list[float] = []
    collision_history:    list[int]   = []
    battery_history:      list[float] = []
    path_length_history:  list[float] = []

    # ---------------------------------------------------------------
    # 4. Main loop
    # ---------------------------------------------------------------
    print(f"\n[Run] Starting episode | max_steps={max_steps}\n")

    for step in range(max_steps):

        # -- 4a. RA replanning --------------------------------------
        if step > 0 and step % REPLAN_INTERVAL == 0:
            print(f"  [RA] Replanning at step {step} ...")
            active_waypoints = _run_ra(
                raw_env,
                current_waypoint_idx,
                active_waypoints,
                ra_iterations,
            )

        # -- 4b. Determine current target waypoint per drone --------
        target_waypoints = _get_targets(
            active_waypoints, current_waypoint_idx, n_drones
        )

        # -- 4c. Compute or predict action -------------------------
        if use_ppo and model is not None:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = _waypoint_follow_action(raw_env, target_waypoints)

        # -- 4d. Environment step -----------------------------------
        obs, reward, terminated, truncated, info = env.step(action)

        # -- 4e. Advance waypoint index when drone reaches target ---
        _advance_waypoints(
            raw_env,
            current_waypoint_idx,
            active_waypoints,
            arrival_radius=3.0,
        )

        # -- 4f. Record drone paths ---------------------------------
        positions = _get_drone_positions(raw_env)
        for i, pos in enumerate(positions):
            drone_paths[i].append(pos.copy())

        # -- 4g. Record metrics ------------------------------------
        _append_metrics(
            info if not use_ppo else info[0],
            coverage_history,
            collision_history,
            battery_history,
            path_length_history,
            step,
        )

        # -- 4h. Terminal check ------------------------------------
        done = bool(terminated) if not use_ppo else bool(terminated[0])
        trunc = bool(truncated) if not use_ppo else bool(truncated[0])

        if (step + 1) % 200 == 0 or done or trunc:
            cov = coverage_history[-1] if coverage_history else 0.0
            print(f"  Step {step+1:>5} | Coverage: {cov:.1f}% | Done: {done or trunc}")

        if done or trunc:
            print(f"\n[Run] Episode ended at step {step+1}.")
            break

    else:
        print(f"\n[Run] Reached max_steps={max_steps}.")

    # ---------------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------------
    final_cov = coverage_history[-1] if coverage_history else 0.0
    final_col = collision_history[-1] if collision_history else 0
    final_bat = battery_history[-1] if battery_history else 0.0
    print(f"\n{'='*60}")
    print(f"  Final Coverage      : {final_cov:.2f}%")
    print(f"  Total Collisions    : {final_col}")
    print(f"  Total Battery Used  : {final_bat:.2f}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------
    # 6. Video / animation
    # ---------------------------------------------------------------
    if save_video:
        _save_animation(
            raw_env=raw_env,
            drone_paths=drone_paths,
            coverage_history=coverage_history,
            collision_history=collision_history,
            battery_history=battery_history,
            video_path=video_path,
        )

    plt.show()

    # Tidy up
    env.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run_fa(
    raw_env: MultiDroneCoverageEnv,
    iterations: int,
) -> np.ndarray:
    """Run the Firefly Algorithm planner and return waypoints (n, n_wp, 3)."""
    print(f"[FA] Running Firefly Algorithm ({iterations} iterations) ...")
    t0 = time.time()
    fa = FireflyPlanner(
        env=raw_env,
        n_waypoints=N_WAYPOINTS_FA,
        max_iter=iterations,
    )
    waypoints = fa.plan()   # expected shape: (n_drones, N_WAYPOINTS_FA, 3)
    print(f"[FA] Done in {time.time()-t0:.1f}s | waypoints shape: {waypoints.shape}")
    return waypoints


def _run_ra(
    raw_env: MultiDroneCoverageEnv,
    current_wp_idx: list[int],
    current_waypoints: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Run the Raven Replanner and return updated waypoints (n, n_wp, 3)."""
    ra = RavenReplanner(
        env=raw_env,
        n_waypoints=N_WAYPOINTS_RA,
        max_iter=iterations,
    )
    new_waypoints = ra.replan(
        current_waypoints=current_waypoints,
        current_wp_idx=current_wp_idx,
    )
    # Pad or trim to maintain a consistent shape
    n_drones = raw_env.n_drones
    padded = np.zeros((n_drones, N_WAYPOINTS_FA, 3), dtype=np.float32)
    take = min(new_waypoints.shape[1], N_WAYPOINTS_FA)
    padded[:, :take, :] = new_waypoints[:, :take, :]
    return padded


def _get_targets(
    waypoints: np.ndarray,
    wp_idx: list[int],
    n_drones: int,
) -> np.ndarray:
    """Return the current target waypoint for each drone (n_drones, 3)."""
    max_wp = waypoints.shape[1] - 1
    targets = np.array(
        [waypoints[i][min(wp_idx[i], max_wp)] for i in range(n_drones)],
        dtype=np.float32,
    )
    return targets


def _advance_waypoints(
    raw_env: MultiDroneCoverageEnv,
    wp_idx: list[int],
    waypoints: np.ndarray,
    arrival_radius: float = 3.0,
) -> None:
    """
    Increment a drone's waypoint index when it is within ``arrival_radius``
    units of its current target.
    """
    positions = _get_drone_positions(raw_env)
    max_wp = waypoints.shape[1] - 1
    for i, pos in enumerate(positions):
        if wp_idx[i] >= max_wp:
            continue  # already at last waypoint
        target = waypoints[i][wp_idx[i]]
        dist = float(np.linalg.norm(pos[:3] - target[:3]))
        if dist < arrival_radius:
            wp_idx[i] += 1


def _get_drone_positions(raw_env: MultiDroneCoverageEnv) -> list[np.ndarray]:
    """
    Retrieve current 3-D positions for all drones from the raw environment.
    Falls back gracefully if the attribute name differs.
    """
    # Try common attribute names
    for attr in ("drone_positions", "positions", "state"):
        if hasattr(raw_env, attr):
            pos = getattr(raw_env, attr)
            if isinstance(pos, np.ndarray) and pos.ndim == 2:
                return [pos[i] for i in range(pos.shape[0])]
    # Ultimate fallback – return zeros so the rest of the pipeline doesn't crash
    return [np.zeros(3) for _ in range(raw_env.n_drones)]


def _waypoint_follow_action(
    raw_env: MultiDroneCoverageEnv,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Compute a proportional waypoint-following action without RL.

    Each drone's action is the unit-normalised vector from its current position
    to its target waypoint, clipped to the action space range [-1, 1].
    """
    positions = _get_drone_positions(raw_env)
    max_speed = getattr(raw_env, "max_speed", 1.0)
    actions = []
    for i, pos in enumerate(positions):
        delta = targets[i][:3] - pos[:3]
        norm  = np.linalg.norm(delta)
        if norm > 1e-6:
            direction = delta / norm
        else:
            direction = np.zeros(3, dtype=np.float32)
        action_i = np.clip(direction * max_speed, -1.0, 1.0)
        actions.append(action_i)
    return np.array(actions, dtype=np.float32).flatten()


def _append_metrics(
    info: dict,
    coverage_history:    list[float],
    collision_history:   list[int],
    battery_history:     list[float],
    path_length_history: list[float],
    step: int,
) -> None:
    """Extract metric values from ``info`` dict and append to history lists."""
    coverage_history.append(float(info.get("coverage_pct", 0.0)))
    collision_history.append(int(info.get("total_collisions", 0)))
    battery_history.append(float(info.get("total_battery_used", 0.0)))
    path_length_history.append(float(info.get("total_path_length", 0.0)))


# ---------------------------------------------------------------------------
# Animation / video
# ---------------------------------------------------------------------------

def _save_animation(
    raw_env:          MultiDroneCoverageEnv,
    drone_paths:      list[list[np.ndarray]],
    coverage_history: list[float],
    collision_history: list[int],
    battery_history:  list[float],
    video_path:       str,
) -> None:
    """
    Build and save a two-panel matplotlib animation:

    Left panel  (ax1) – 2-D top-down heatmap of the coverage grid with
                        drone paths, obstacles, and recharge stations.
    Right panel (ax2) – Multi-objective metrics time-series:
                        coverage %, cumulative collisions, battery used.

    Attempts FFMpegWriter (mp4) first; falls back to PillowWriter (gif).
    """
    n_steps   = len(coverage_history)
    n_drones  = len(drone_paths)
    steps_arr = np.arange(n_steps)

    # Pre-compute normalised collision and battery curves for ax2
    col_arr = np.array(collision_history, dtype=float)
    bat_arr = np.array(battery_history,   dtype=float)
    col_norm = col_arr / (col_arr.max() + 1e-8) * 100.0
    bat_norm = bat_arr / (bat_arr.max() + 1e-8) * 100.0

    # Coverage grid for heatmap (try common attribute names)
    cov_grid = None
    for attr in ("coverage_grid", "grid", "map"):
        if hasattr(raw_env, attr):
            cov_grid = getattr(raw_env, attr)
            break
    if cov_grid is None:
        cov_grid = np.zeros((50, 50), dtype=float)

    # Recharge stations and obstacles
    recharge_stations = getattr(raw_env, "recharge_stations", [])
    dynamic_obstacles = getattr(raw_env, "dynamic_obstacles", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    def animate(frame: int):
        ax1.cla()
        ax2.cla()

        # Style resets
        ax1.set_facecolor("#16213e")
        ax2.set_facecolor("#16213e")
        for ax in (ax1, ax2):
            ax.tick_params(colors="white")

        step_idx = min(frame, n_steps - 1)

        # -- Left panel: 2-D coverage heatmap ----------------------
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "cov", ["#16213e", "#00b894"]
        )
        ax1.imshow(
            cov_grid,
            cmap=cmap,
            origin="lower",
            aspect="auto",
            alpha=0.85,
        )

        # Drone paths up to current frame
        for i in range(n_drones):
            path = drone_paths[i][: step_idx + 1]
            if len(path) < 2:
                continue
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax1.plot(xs, ys, color=DRONE_COLORS[i % len(DRONE_COLORS)],
                     linewidth=1.2, alpha=0.7)
            # Current drone position
            ax1.plot(
                xs[-1], ys[-1],
                "o",
                color=DRONE_COLORS[i % len(DRONE_COLORS)],
                markersize=7,
                label=f"D{i+1}",
                markeredgecolor="white",
                markeredgewidth=0.6,
            )

        # Recharge stations
        for rs in recharge_stations:
            ax1.plot(
                rs[0], rs[1],
                "s",
                color="magenta",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label="Recharge",
            )

        # Dynamic obstacles
        for obs in dynamic_obstacles:
            circle = plt.Circle(
                (obs[0], obs[1]),
                radius=getattr(obs, "radius", 2.0),
                color="dodgerblue",
                alpha=0.4,
            )
            ax1.add_patch(circle)

        cov_pct = coverage_history[step_idx] if coverage_history else 0.0
        ax1.set_title(
            f"Step {step_idx} | Coverage: {cov_pct:.1f}%",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("X", color="white")
        ax1.set_ylabel("Y", color="white")

        handles, labels = ax1.get_legend_handles_labels()
        # Deduplicate legend entries
        by_label = dict(zip(labels, handles))
        ax1.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            fontsize=7,
            facecolor="#0f3460",
            labelcolor="white",
            framealpha=0.7,
        )

        # -- Right panel: metrics time-series ----------------------
        t = steps_arr[: step_idx + 1]
        cov_slice = coverage_history[: step_idx + 1]
        col_slice = col_norm[: step_idx + 1]
        bat_slice = bat_norm[: step_idx + 1]

        ax2.plot(t, cov_slice,  color="#00b894", linewidth=1.8, label="Coverage %")
        ax2.plot(t, col_slice,  color="#e17055", linewidth=1.4,
                 linestyle="--", label="Collisions (norm)")
        ax2.plot(t, bat_slice,  color="#fdcb6e", linewidth=1.4,
                 linestyle=":",  label="Battery (norm)")

        ax2.set_xlim(0, max(n_steps, 1))
        ax2.set_ylim(-5, 105)
        ax2.set_xlabel("Step", color="white")
        ax2.set_ylabel("Value (%)", color="white")
        ax2.set_title("Multi-Objective Metrics", color="white",
                      fontsize=11, fontweight="bold")
        ax2.legend(
            fontsize=8,
            facecolor="#0f3460",
            labelcolor="white",
            framealpha=0.7,
        )

        fig.tight_layout()

    frame_count = n_steps
    interval_ms = max(1, int(1000 / 30))  # target ~30 fps

    ani = FuncAnimation(
        fig, animate, frames=frame_count, interval=interval_ms, blit=False
    )

    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

    # Try mp4 via FFMpeg first
    try:
        writer = FFMpegWriter(fps=30, metadata={"title": "Multi-Drone Coverage"})
        ani.save(video_path, writer=writer)
        print(f"\n[Video] Saved mp4 -> {os.path.abspath(video_path)}")
    except (RuntimeError, FileNotFoundError, Exception) as exc:
        print(f"[Video] FFMpeg unavailable ({type(exc).__name__}: {exc}).")
        gif_path = os.path.splitext(video_path)[0] + ".gif"
        try:
            writer_gif = PillowWriter(fps=15)
            ani.save(gif_path, writer=writer_gif)
            print(f"[Video] Saved gif  -> {os.path.abspath(gif_path)}")
        except Exception as gif_exc:
            print(f"[Video] Could not save animation: {gif_exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated FA + RA + PPO multi-drone coverage runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum episode steps.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        default=False,
        help="Disable video/animation saving.",
    )
    parser.add_argument(
        "--fa-iter",
        type=int,
        default=40,
        dest="fa_iter",
        help="Firefly Algorithm optimisation iterations.",
    )
    parser.add_argument(
        "--ra-iter",
        type=int,
        default=25,
        dest="ra_iter",
        help="Raven Replanner optimisation iterations per replan.",
    )
    parser.add_argument(
        "--no-ppo",
        action="store_true",
        default=False,
        help="Disable PPO controller; use pure waypoint-following instead.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_integrated(
        max_steps=args.steps,
        save_video=not args.no_video,
        video_path=os.path.normpath(
            os.path.join(_SRC_DIR, "..", "models", "drone_coverage.mp4")
        ),
        fa_iterations=args.fa_iter,
        ra_iterations=args.ra_iter,
        use_ppo=not args.no_ppo,
    )
