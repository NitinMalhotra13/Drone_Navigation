# src/drone_env_3d.py
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from dynamic_obstacles import DynamicObstacle
from generate_static_obstacles import generate_static_obstacles  # domain randomization for trees

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
TERRAIN_PATH = os.path.join(BASE_DIR, "terrain.npy")
STATIC_PATH = os.path.join(BASE_DIR, "static_obstacles.npy")

START_XY = (0, 0)
GOAL_XY = (45, 45)  # goal cell; env uses +0.5 offset internally


class Drone3DEnv(gym.Env):
    """
    RL-first drone environment with:
    - Cruise altitude
    - Descent + landing at final goal
    - Recharge stations (touch-and-go when battery low)
    - Dynamic obstacles
    - Optional random static trees per episode
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        demo_mode=False,
        grid_size=(50, 50, 10),
        safety_radius=1.5,
        max_speed=1.0,
        battery_capacity=100.0,
        battery_drain_per_step=0.12,
        wind_enabled=False,
        sensor_noise_std=0.0,
        dyn_count=6,
        move_mode="above_terrain",
        randomize_static=False,   # domain randomization flag
    ):
        super().__init__()

        # grid & terrain
        self.grid = np.array(grid_size, dtype=int)
        self.GX, self.GY, self.GZ = self.grid

        if not os.path.exists(TERRAIN_PATH):
            raise FileNotFoundError("Missing terrain.npy — run generate_terrain.py")
        self.terrain = np.load(TERRAIN_PATH)

        # static obstacles (trees)
        self.randomize_static = bool(randomize_static)
        if os.path.exists(STATIC_PATH):
            self.static = np.load(STATIC_PATH)
        else:
            self.static = np.zeros((self.GX, self.GY, self.GZ), dtype=bool)

        # dynamic obstacles
        self.dyn = DynamicObstacle(grid_size=self.grid, count=dyn_count, move_mode=move_mode)

        # basic params
        self.SAFETY_RADIUS = float(safety_radius)
        self.max_speed = float(max_speed)
        self.battery_capacity = float(battery_capacity)
        self.battery = float(battery_capacity)
        self.battery_drain_per_step = float(battery_drain_per_step)
        self.wind_enabled = bool(wind_enabled)
        self.demo_mode = bool(demo_mode)
        self.sensor_noise_std = float(sensor_noise_std)

        # start and goal XY (center of cell)
        sx, sy = START_XY
        gx, gy = GOAL_XY
        self.start = np.array(
            [
                float(sx) + 0.5,
                float(sy) + 0.5,
                min(self.terrain[sx, sy] + 0.8, self.GZ - 2),
            ],
            dtype=float,
        )
        self.goal_xy = np.array([float(gx) + 0.5, float(gy) + 0.5], dtype=float)

        # flight constants
        self.CRUISE_Z = min(6.0, float(self.GZ - 1.0))
        self.HORIZONTAL_STEP_CAP = 0.5
        self.VERTICAL_CAP = 0.2
        self.VERTICAL_RANGE = 1.0
        self.ASCENT_TOL = 1e-6
        self.APPROACH_XY_THRESH = 1.0

        # smoothing parameter (for action smoothing)
        self.smooth_alpha = 0.35  # blend factor: new = alpha*desired + (1-alpha)*prev_vel

        # recharge stations (fixed z = 3.5)
        self.recharge_stations = [(10, 10), (25, 25), (40, 10)]
        self._compute_recharge_positions_3d_fixed()

        # phases & goals
        self.phase = "ascend"
        self.current_goal = np.array(
            [self.goal_xy[0], self.goal_xy[1], self.CRUISE_Z],
            dtype=float,
        )
        self.final_goal_3d = None  # when descending to either final goal or recharge

        # autopilot (RL-first)
        self.autopilot_enabled = True
        self.autopilot_trigger_multiplier = 0.7

        # anti-stuck
        self.progress_window = deque(maxlen=12)
        self.stuck_progress_threshold = 0.01
        self.stuck_boost = 0.18
        self.stuck_check_min_steps = 8

        # action & observation spaces (pos3 + vel3 + goal_rel3 + batt1 + lidar8 = 18)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_low = np.array([-np.inf] * 18, dtype=np.float32)
        obs_high = np.array([np.inf] * 18, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # state
        self.pos = self.start.copy()
        self.vel = np.zeros(3, dtype=float)
        self.step_count = 0
        self.max_steps = 2000

        self._ensure_dynamic_initialized()

    # -------------------------
    # Helpers
    # -------------------------
    def _compute_recharge_positions_3d_fixed(self):
        out = []
        for (rx, ry) in self.recharge_stations:
            ix = int(np.clip(rx, 0, self.GX - 1))
            iy = int(np.clip(ry, 0, self.GY - 1))
            out.append(np.array([ix + 0.5, iy + 0.5, 3.5], dtype=float))
        self.recharge_positions_3d = out

    def _ensure_dynamic_initialized(self):
        if not getattr(self.dyn, "initialized", False):
            self.dyn.initialize(static=self.static, terrain=self.terrain)

    def _clip_pos(self, p):
        p = p.copy()
        p[0] = np.clip(p[0], 0.0, self.GX - 1.0)
        p[1] = np.clip(p[1], 0.0, self.GY - 1.0)
        p[2] = np.clip(p[2], 0.0, self.GZ - 1.0)
        return p

    def _min_distance(self, pos):
        # distance to static obstacles
        idx = np.argwhere(self.static)
        if idx.size > 0:
            centers = idx.astype(float) + 0.5
            dists = np.linalg.norm(centers - pos.reshape(1, 3), axis=1)
            min_static = float(np.min(dists))
        else:
            min_static = float(np.inf)

        # distance to dynamic obstacles
        dpos = np.array(self.dyn.positions) if hasattr(self.dyn, "positions") else np.zeros((0, 3))
        if dpos.size > 0:
            dists2 = np.linalg.norm(dpos - pos.reshape(1, 3), axis=1)
            min_dyn = float(np.min(dists2))
        else:
            min_dyn = float(np.inf)

        return min(min_static, min_dyn)

    def _closest_obstacle_distances(self, pos, bearings=8, max_range=12.0):
        angles = np.linspace(0, 2 * np.pi, bearings, endpoint=False)
        dists = np.full(bearings, max_range, dtype=float)

        # static
        idx = np.argwhere(self.static)
        centers = (idx.astype(float) + 0.5) if idx.size > 0 else np.zeros((0, 3))

        if centers.size > 0:
            rel = centers - pos.reshape(1, 3)
            for i, a in enumerate(angles):
                dir2d = np.array([math.cos(a), math.sin(a)])
                proj = rel[:, :2].dot(dir2d)
                lateral = np.linalg.norm(rel[:, :2] - np.outer(proj, dir2d), axis=1)
                mask = (proj > 0) & (proj < max_range) & (lateral < 0.8)
                if np.any(mask):
                    dists[i] = float(np.min(proj[mask]))

        # dynamic
        if hasattr(self.dyn, "positions"):
            for p in self.dyn.positions:
                rel = np.array(p, dtype=float) - pos
                for i, a in enumerate(angles):
                    dir2d = np.array([math.cos(a), math.sin(a)])
                    proj = rel[:2].dot(dir2d)
                    lateral = np.linalg.norm(rel[:2] - proj * dir2d)
                    if (proj > 0) and (proj < max_range) and (lateral < 0.8):
                        dists[i] = min(dists[i], float(proj))

        return dists

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # randomize trees each episode if enabled
        if self.randomize_static:
            try:
                self.static = generate_static_obstacles()
            except Exception as e:
                print("[WARN] generate_static_obstacles failed, keeping previous trees:", e)

        self._ensure_dynamic_initialized()
        self.dyn.initialize(static=self.static, terrain=self.terrain)
        self._compute_recharge_positions_3d_fixed()

        self.pos = self.start.copy()
        self.vel = np.zeros(3, dtype=float)
        self.step_count = 0
        self.battery = float(self.battery_capacity)
        self.phase = "ascend"
        self.final_goal_3d = None
        self.current_goal = np.array(
            [self.goal_xy[0], self.goal_xy[1], self.CRUISE_Z],
            dtype=float,
        )

        self.progress_window.clear()

        # small XY jitter at start
        jitter = np.random.uniform(-0.3, 0.3, size=2)
        self.pos[0] = np.clip(self.pos[0] + jitter[0], 0, self.GX - 1.0)
        self.pos[1] = np.clip(self.pos[1] + jitter[1], 0, self.GY - 1.0)

        obs = self._get_observation()
        return obs, {}

    def _apply_wind(self):
        if not self.wind_enabled:
            return np.zeros(3, dtype=float)
        return np.random.normal(scale=0.04, size=3)

    def _apply_autopilot(self, pos, vel):
        to_goal = (self.current_goal - pos)
        dist = np.linalg.norm(to_goal) + 1e-9
        desired = (to_goal / dist) * (self.max_speed * 0.9)
        return desired

    def step(self, action):
        self.step_count += 1
        # move dynamic obstacles first
        self.dyn.move(static=self.static, terrain=self.terrain)

        act = np.array(action, dtype=float).flatten()
        act = np.clip(act, -1.0, 1.0)
        desired_vel = act * self.max_speed

        # previous distance for progress/reward
        prev_goal_dist = np.linalg.norm(self.pos - self.current_goal)

        # ----------------- PHASE LOGIC -----------------
        if self.phase == "ascend":
            climb_amount = min(self.max_speed, self.CRUISE_Z - self.pos[2])
            climb_amount = max(climb_amount, 0.0)
            desired_vel = np.array([0.0, 0.0, climb_amount], dtype=float)

        elif self.phase == "cruise":
            dx, dy, dz = desired_vel
            dx = np.clip(dx, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
            dy = np.clip(dy, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
            dz = np.clip(dz, -self.VERTICAL_CAP, self.VERTICAL_CAP)
            next_z = self.pos[2] + dz
            if next_z < (self.CRUISE_Z - self.VERTICAL_RANGE) or next_z > (
                self.CRUISE_Z + self.VERTICAL_RANGE
            ):
                dz = 0.0

            goal_vec = self.goal_xy - self.pos[:2]
            goal_dist = np.linalg.norm(goal_vec) + 1e-9
            goal_dir = goal_vec / goal_dist

            # forward bias toward goal
            dx += 0.20 * goal_dir[0]
            dy += 0.20 * goal_dir[1]

            # sideways suppression
            fwd = dx * goal_dir[0] + dy * goal_dir[1]
            if fwd > 0:
                side_x = dx - fwd * goal_dir[0]
                side_y = dy - fwd * goal_dir[1]
                dx -= 0.35 * side_x
                dy -= 0.35 * side_y

            dx = np.clip(dx, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
            dy = np.clip(dy, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
            desired_vel = np.array([dx, dy, dz], dtype=float)

        elif self.phase == "descent":
            # small XY correction while descending to final_goal_3d
            if self.final_goal_3d is not None:
                vec_xy = self.final_goal_3d[:2] - self.pos[:2]
                dist_xy = np.linalg.norm(vec_xy) + 1e-9
                dir_xy = vec_xy / dist_xy
                corr_xy = dir_xy * 0.12  # ~12 cm per step
                corr_xy[np.isnan(corr_xy)] = 0.0
            else:
                corr_xy = np.zeros(2, dtype=float)

            target_z = float(self.final_goal_3d[2]) if self.final_goal_3d is not None else self.pos[2]
            dz = np.clip(target_z - self.pos[2], -self.max_speed, self.max_speed)
            desired_vel = np.array([corr_xy[0], corr_xy[1], dz], dtype=float)

        # sensor noise
        if self.sensor_noise_std > 0.0:
            desired_vel = desired_vel + np.random.normal(scale=self.sensor_noise_std, size=3)

        # RL-first autopilot trigger (predict next step)
        wind = self._apply_wind()
        # tentative pos before smoothing (used for autopilot trigger)
        tentative_vel = desired_vel + wind
        tentative_pos = self._clip_pos(self.pos + tentative_vel)
        min_d_next = self._min_distance(tentative_pos)

        use_autopilot = False
        if self.autopilot_enabled and (min_d_next < self.SAFETY_RADIUS * self.autopilot_trigger_multiplier):
            use_autopilot = True

        if use_autopilot and not self.demo_mode:
            ap = self._apply_autopilot(self.pos, self.vel)
            if self.phase == "cruise":
                ax, ay, az = ap
                ax = np.clip(ax, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
                ay = np.clip(ay, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
                az = np.clip(az, -self.VERTICAL_CAP, self.VERTICAL_CAP)
                desired_vel = np.array([ax, ay, az], dtype=float)
            elif self.phase == "descent":
                desired_vel = np.array(
                    [desired_vel[0], desired_vel[1], np.clip(ap[2], -self.max_speed, self.max_speed)],
                    dtype=float,
                )

        # --- ACTION SMOOTHING: low-pass filter to reduce zig-zag ---
        alpha = float(self.smooth_alpha)
        desired_vel = alpha * desired_vel + (1.0 - alpha) * self.vel

        # integrate (with wind)
        self.vel = desired_vel + wind
        self.pos = self._clip_pos(self.pos + self.vel)

        # ----------------- PHASE TRANSITIONS -----------------
        # ascend -> cruise
        if self.phase == "ascend" and self.pos[2] >= self.CRUISE_Z - self.ASCENT_TOL:
            self.pos[2] = float(self.CRUISE_Z)
            self.vel[2] = 0.0
            self.phase = "cruise"
            self.current_goal = np.array(
                [self.goal_xy[0], self.goal_xy[1], self.CRUISE_Z],
                dtype=float,
            )

        # cruise -> descent for final goal (XY close enough)
        if self.phase == "cruise":
            xy_dist_to_goal = np.linalg.norm(self.pos[:2] - self.goal_xy)
            if xy_dist_to_goal <= self.APPROACH_XY_THRESH:
                gi = int(np.clip(self.goal_xy[0] - 0.5, 0, self.GX - 1))
                gj = int(np.clip(self.goal_xy[1] - 0.5, 0, self.GY - 1))
                terrain_z = float(self.terrain[gi, gj])
                self.final_goal_3d = np.array(
                    [self.goal_xy[0], self.goal_xy[1], terrain_z],
                    dtype=float,
                )
                self.phase = "descent"

        # cruise -> descent for recharge ONLY when battery is low
        if self.phase == "cruise" and self.battery < 20.0:
            for rs in self.recharge_positions_3d:
                if np.linalg.norm(self.pos[:2] - rs[:2]) <= self.APPROACH_XY_THRESH:
                    self.final_goal_3d = rs.copy()
                    self.phase = "descent"
                    break

        # descent -> landed / recharge touch-and-go
        if self.phase == "descent" and (self.final_goal_3d is not None):
            horiz = np.linalg.norm(self.pos[:2] - self.final_goal_3d[:2])
            vert = abs(self.pos[2] - self.final_goal_3d[2])

            if horiz < 0.3 and vert < 0.3:
                # Is this the main final goal or a recharge station?
                is_main_goal = np.allclose(self.final_goal_3d[:2], self.goal_xy, atol=1e-3)

                if is_main_goal:
                    # Snap exactly to final_goal_3d so visualisers see exact goal reached
                    self.pos = self.final_goal_3d.copy()
                    self.vel[:] = 0.0
                    self.phase = "landed"
                else:
                    # This is a recharge station → touch-and-go:
                    # refill and jump back to cruise toward main goal.
                    self.battery = self.battery_capacity

                    # Snap to recharge XY at cruise altitude
                    self.pos = np.array(
                        [self.final_goal_3d[0], self.final_goal_3d[1], self.CRUISE_Z],
                        dtype=float,
                    )
                    self.vel[:] = 0.0

                    # Back to cruise, target = main goal at cruise Z
                    self.phase = "cruise"
                    self.current_goal = np.array(
                        [self.goal_xy[0], self.goal_xy[1], self.CRUISE_Z],
                        dtype=float,
                    )
                    self.final_goal_3d = None

                    print("[INFO] Recharged and returning to cruise")

        # ensure current_goal pointer is consistent
        if self.phase == "cruise":
            self.current_goal = np.array(
                [self.goal_xy[0], self.goal_xy[1], self.CRUISE_Z],
                dtype=float,
            )
        elif self.phase in ("descent", "landed") and (self.final_goal_3d is not None):
            self.current_goal = self.final_goal_3d.copy()

        # battery drain
        self.battery = max(
            0.0,
            self.battery - self.battery_drain_per_step - 0.001 * np.linalg.norm(self.vel),
        )

        # progress & anti-stuck
        curr_goal_dist = np.linalg.norm(self.pos - self.current_goal)
        progress = max(0.0, prev_goal_dist - curr_goal_dist)
        self.progress_window.append(progress)

        stuck = False
        if (len(self.progress_window) >= self.stuck_check_min_steps) and (
            np.mean(self.progress_window) < self.stuck_progress_threshold
        ):
            stuck = True

        if stuck and self.phase == "cruise":
            goal_vec = self.goal_xy - self.pos[:2]
            goal_dist = np.linalg.norm(goal_vec) + 1e-9
            goal_dir = goal_vec / goal_dist
            boost_x = self.stuck_boost * goal_dir[0]
            boost_y = self.stuck_boost * goal_dir[1]
            nudge = np.array([boost_x, boost_y, 0.0], dtype=float)
            proposed = self._clip_pos(self.pos + nudge)
            if self._min_distance(proposed) > 0.35:
                self.pos = proposed
                self.progress_window.clear()
                print(f"[unstuck] mild boost applied at step={self.step_count}")

        # collision detection
        min_d = self._min_distance(self.pos)
        collided = (min_d < 0.35)

        # determine if final goal reached
        goal_reached = False
        if self.phase == "landed" and (self.final_goal_3d is not None):
            if np.allclose(self.final_goal_3d[:2], self.goal_xy, atol=1e-3):
                goal_reached = True

        # low battery (global) → set temporary cruise-level goal toward nearest recharge
        if self.battery < 20.0 and (self.phase != "descent"):
            rs_xy = np.array(self.recharge_stations, dtype=float)
            dists_xy = np.linalg.norm(rs_xy - self.pos[:2], axis=1)
            idx = int(np.argmin(dists_xy))
            rs3 = self.recharge_positions_3d[idx]
            self.current_goal = np.array(
                [rs3[0], rs3[1], self.CRUISE_Z],
                dtype=float,
            )

        # ----------------- REWARD SHAPING -----------------
        reward = 0.0
        reward -= 0.01  # step penalty

        prev_goal_dist_est = np.linalg.norm((self.pos - self.vel) - self.current_goal)
        reward += (prev_goal_dist_est - curr_goal_dist) * 0.8

        # encourage heading alignment with goal in XY for smoother paths
        goal_vec_xy = self.goal_xy - self.pos[:2]
        norm_goal_xy = np.linalg.norm(goal_vec_xy)
        norm_vel_xy = np.linalg.norm(self.vel[:2])
        if norm_goal_xy > 1e-6 and norm_vel_xy > 1e-6:
            heading_cos = float(
                np.dot(goal_vec_xy / norm_goal_xy, self.vel[:2] / norm_vel_xy)
            )
            reward += 0.03 * heading_cos

        # light penalty for energy usage
        reward -= (0.002 * (self.battery_capacity - self.battery))

        # keep drone roughly near map center (soft)
        center = np.array([self.GX / 2.0, self.GY / 2.0])
        dist_to_center = np.linalg.norm(self.pos[:2] - center)
        max_center_dist = math.sqrt((self.GX / 2.0) ** 2 + (self.GY / 2.0) ** 2)
        center_penalty = dist_to_center / (max_center_dist + 1e-9)
        reward -= 0.12 * center_penalty

        # Extra penalty for passing too close to static tree centers (top-down safety)
        # compute minimal XY distance to static tree centers (if any)
        idx = np.argwhere(self.static)
        if idx.size > 0:
            centers_xy = (idx.astype(float)[:, :2] + 0.5)
            rel_xy = centers_xy - self.pos[:2].reshape(1, 2)
            dists_xy = np.linalg.norm(rel_xy, axis=1)
            min_static_xy = float(np.min(dists_xy))
            # Penalize strongly when within 1.2 units in XY (discourage near-misses)
            if min_static_xy < 1.2:
                reward -= 4.0 * (1.2 - min_static_xy)

        # ----------------- TERMINATION / INFO --------------
        terminated = False
        truncated = False
        info = {
            "collided": False,
            "recharge": False,
            "goal_reached": False,
            "battery_dead": False,
            "live_pos": self.pos.copy(),
            "message": "",
            "phase": self.phase,
        }

        print(
            f"[live] step={self.step_count} "
            f"pos={self.pos[0]:.3f},{self.pos[1]:.3f},{self.pos[2]:.3f} "
            f"battery={self.battery:.2f} phase={self.phase}"
        )

        if collided:
            reward -= 50.0
            terminated = True
            info["collided"] = True
            info["message"] = "COLLIDED"
            print(f"[ALERT] COLLISION at {self.pos}")

        if self.battery <= 0.0:
            reward -= 30.0
            terminated = True
            info["battery_dead"] = True
            info["message"] = "BATTERY_DEPLETED"
            print("[ALERT] Battery depleted")

        if goal_reached:
            reward += 100.0
            terminated = True
            info["goal_reached"] = True
            info["message"] = "GOAL_REACHED"
            print("[INFO] Final goal reached after landing")
            # reset current goal to cruise-level final for consistency
            self.current_goal = np.array(
                [self.goal_xy[0], self.goal_xy[1], self.CRUISE_Z],
                dtype=float,
            )

        # (old recharge detection kept for info; usually not triggered now
        # because we "touch-and-go" at cruise altitude)
        for idx, rs_pos in enumerate(self.recharge_positions_3d):
            if np.linalg.norm(self.pos - rs_pos) < 1.0:
                self.battery = self.battery_capacity
                info["recharge"] = True
                info["message"] = "RECHARGED"
                print(f"[INFO] Recharged at station idx={idx} pos={rs_pos}")

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_observation()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -------------------------
    # observations
    # -------------------------
    def _get_observation(self):
        pos = self.pos.astype(np.float32)
        vel = self.vel.astype(np.float32)
        goal_rel = (self.current_goal - self.pos).astype(np.float32)
        batt = np.array([self.battery / self.battery_capacity], dtype=np.float32)
        dists = self._closest_obstacle_distances(
            self.pos,
            bearings=8,
            max_range=12.0,
        ).astype(np.float32)
        obs = np.concatenate([pos, vel, goal_rel, batt, dists]).astype(np.float32)
        if self.sensor_noise_std > 0.0:
            obs = obs + np.random.normal(
                scale=self.sensor_noise_std,
                size=obs.shape,
            ).astype(np.float32)
        return obs

    def render(self):
        pass

    def close(self):
        pass
