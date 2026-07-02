# src/multi_drone_coverage_env.py
"""
Multi-Drone Maximum Area Coverage Environment  (100x100x15 grid, 6 drones)

Real-world physics modelled:
  ? Spatially-varying wind field   — gusts change direction/magnitude over time
  ? Atmospheric turbulence         — high-freq Gaussian noise on top of wind
  ? Aerodynamic drag               — v2 drag slows drones; more energy to fight it
  ? Air-density / altitude effect  — thinner air at high Z -> less lift efficiency
  ? Ground effect                  — cushion of air near terrain reduces drag
  ? Thermal updrafts               — sun-warmed terrain creates localised lift/sink
  ? Wake turbulence                — drones disturb each other's airflow when close
  ? GPS / sensor noise             — position & velocity observations are noisy
  ? Battery degradation            — drain prop to thrust2 (fighting wind costs more)
  ? Motor health                   — slow random degradation; very low probability
      of partial motor failure reducing max-speed

Flight phases (identical to original Drone3DEnv):
  "ascend"           -> climb to CRUISE_Z = 6.0 before any horizontal movement
  "cruise"           -> cover the map at cruise altitude
  "recharge_descend" -> descend to nearest station when battery < 20 %
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from dynamic_obstacles import DynamicObstacle
from generate_static_obstacles import generate_static_obstacles

# -- Paths -----------------------------------------------------------------
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..", "dataset")
TERRAIN_PATH = os.path.join(BASE_DIR, "terrain.npy")
STATIC_PATH  = os.path.join(BASE_DIR, "static_obstacles.npy")

# -- Grid ------------------------------------------------------------------
GRID_SIZE = (100, 100, 15)
CRUISE_Z  = 6.0
COVERAGE_SENSOR_RADIUS = 3.5

# -- 6 drone start positions (clustered dispatch zone at bottom-left) -------
DRONE_STARTS_XY = [
    (4, 4), (4, 6), (6, 4),
    (6, 6), (5, 4), (5, 6),
]

# -- Recharge stations -----------------------------------------------------
RECHARGE_STATIONS_XY = [
    (20, 20), (20, 80), (50, 50), (80, 20), (80, 80),
]

DRONE_COLORS = ["red", "cyan", "lime", "orange", "magenta", "yellow"]


# ==========================================================================
class WindField:
    """
    Spatially and temporally varying 3D wind field over the 100x100 map.

    Models:
      - Dominant wind direction that slowly rotates over time
      - Spatial variation: wind is stronger over open areas, weaker near edges
      - Gusts: sudden short-duration wind speed spikes
      - Turbulence: per-step high-frequency Gaussian noise
    """

    def __init__(self, grid_x: int, grid_y: int, seed: int = 42):
        self.GX = grid_x
        self.GY = grid_y
        self.rng = np.random.default_rng(seed)

        # Base wind state
        self.base_dir   = self.rng.uniform(0, 2 * math.pi)  # radians
        self.base_speed = self.rng.uniform(0.05, 0.20)       # m/step units
        self.gust_remaining = 0
        self.gust_vec       = np.zeros(3)

        # Rotation rate of dominant wind direction (rad/step)
        self.rotation_rate = self.rng.uniform(0.002, 0.008)

    def step(self):
        """Advance the wind field one timestep."""
        # Slowly rotate dominant direction
        self.base_dir += self.rotation_rate + self.rng.normal(0, 0.001)

        # Randomly trigger gusts
        if self.gust_remaining <= 0 and self.rng.random() < 0.03:
            gust_speed   = self.rng.uniform(0.2, 0.5)
            gust_angle   = self.rng.uniform(0, 2 * math.pi)
            gust_vz      = self.rng.uniform(-0.08, 0.08)
            self.gust_vec = np.array([
                math.cos(gust_angle) * gust_speed,
                math.sin(gust_angle) * gust_speed,
                gust_vz,
            ])
            self.gust_remaining = int(self.rng.integers(5, 25))
        elif self.gust_remaining > 0:
            self.gust_remaining -= 1
            self.gust_vec *= 0.92  # decay
        else:
            self.gust_vec = np.zeros(3)

    def at(self, pos: np.ndarray) -> np.ndarray:
        """
        Return the wind vector at a given 3D position (pos = [x, y, z]).
        Includes spatial variation and turbulence.
        """
        # Base wind in dominant direction
        base = np.array([
            math.cos(self.base_dir) * self.base_speed,
            math.sin(self.base_dir) * self.base_speed,
            0.0,
        ])

        # Altitude effect: wind is stronger at higher altitudes
        alt_factor = 0.5 + 0.5 * (pos[2] / 15.0)
        base *= alt_factor

        # Add gust
        wind = base + self.gust_vec

        # High-frequency turbulence (per-call noise)
        turb_std = 0.03 + 0.02 * alt_factor
        turbulence = np.random.normal(0.0, turb_std, size=3)
        turbulence[2] *= 0.5   # vertical turbulence is smaller

        return wind + turbulence


# ==========================================================================
class ThermalField:
    """
    Terrain-driven thermal updrafts / downdrafts.

    Sun-heated flat terrain creates upward columns (thermals).
    Shaded valleys and ridges create downdrafts.
    Intensity varies slowly over time.
    """

    def __init__(self, terrain: np.ndarray, seed: int = 7):
        self.terrain = terrain
        self.rng = np.random.default_rng(seed)
        GX, GY = terrain.shape

        # Pre-compute thermal map: flat low-lying areas -> updraft (+)
        #                           steep ridges         -> downdraft (-)
        from scipy.ndimage import gaussian_filter
        low_flat = (terrain < 1.5).astype(float)
        self.thermal_map = gaussian_filter(low_flat, sigma=4.0) * 2.0 - 1.0
        self.thermal_map = np.clip(self.thermal_map, -1.0, 1.0)

        # Time-varying intensity
        self.intensity    = self.rng.uniform(0.02, 0.06)
        self.phase_offset = self.rng.uniform(0, math.pi)
        self.t = 0

    def step(self):
        self.t += 1

    def at(self, pos: np.ndarray) -> float:
        """
        Returns vertical drift (m/step) due to thermals at pos.
        Positive = updraft, Negative = downdraft.
        """
        xi = int(np.clip(pos[0], 0, self.thermal_map.shape[0] - 1))
        yi = int(np.clip(pos[1], 0, self.thermal_map.shape[1] - 1))
        base_strength = float(self.thermal_map[xi, yi])

        # Thermals weaken with altitude (peak at 3–8 m, fade out above)
        alt_envelope = math.exp(-0.5 * ((pos[2] - 5.0) / 4.0) ** 2)

        # Time oscillation (thermals pulse with solar heating)
        time_factor = 0.6 + 0.4 * math.sin(self.t * 0.02 + self.phase_offset)

        return base_strength * self.intensity * alt_envelope * time_factor


# ==========================================================================
class MultiDroneCoverageEnv(gym.Env):
    """
    6-drone area coverage environment with real-world physics.

    Observation per drone (14 values):
        pos_norm(3)  + vel_norm(3)  + battery_norm(1) + coverage_ratio(1)
        + rel_centroid_norm(3) + wind_at_drone_norm(3)
    -> total obs dim = 14 x N_DRONES (flattened)

    Action (flattened, shape = N_DRONES x 3):
        Desired XY velocity + vertical adjustment. Clipped to [-1, 1].
        During "ascend" phase, horizontal action is ignored.
    """

    metadata = {"render_modes": ["human"]}

    # ----------------------------------------------------------------------
    def __init__(
        self,
        n_drones: int          = 6,
        grid_size: tuple       = GRID_SIZE,
        safety_radius: float   = 1.8,
        max_speed: float       = 1.8,
        battery_capacity: float= 200.0,
        battery_drain_base: float = 0.08,   # base drain per step
        max_steps: int         = 2000,
        dyn_count: int         = 12,
        randomize_static: bool = False,
        sensor_radius: float   = COVERAGE_SENSOR_RADIUS,
        wind_enabled: bool     = True,
        thermal_enabled: bool  = True,
        sensor_noise_std: float= 0.04,      # GPS noise std (normalised units)
        drag_coeff: float      = 0.12,      # aerodynamic drag coefficient
        motor_fail_prob: float = 0.0002,    # per-drone per-step probability
    ):
        super().__init__()

        self.n_drones          = n_drones
        self.grid              = np.array(grid_size, dtype=int)
        self.GX, self.GY, self.GZ = self.grid
        self.safety_radius     = float(safety_radius)
        self.max_speed         = float(max_speed)
        self.battery_capacity  = float(battery_capacity)
        self.battery_drain_base= float(battery_drain_base)
        self.max_steps         = int(max_steps)
        self.sensor_radius     = float(sensor_radius)
        self.randomize_static  = randomize_static
        self.wind_enabled      = wind_enabled
        self.thermal_enabled   = thermal_enabled
        self.sensor_noise_std  = float(sensor_noise_std)
        self.drag_coeff        = float(drag_coeff)
        self.motor_fail_prob   = float(motor_fail_prob)

        # -- Multi-objective reward weights --------------------------------
        # w_collision : penalise any collision event
        # w_path      : penalise total distance flown (shorter path = better)
        # w_battery   : penalise battery consumed (conservation = better)
        # All three are combined into the shared team reward each step.
        self.w_collision = 15.0    # per-collision penalty multiplier
        self.w_path      = 0.005   # per-unit-distance penalty
        self.w_battery   = 0.008   # per-unit-battery-consumed penalty

        # -- Flight phase constants (same as original Drone3DEnv) ----------
        self.CRUISE_Z            = CRUISE_Z
        self.HORIZONTAL_STEP_CAP = 1.20
        self.VERTICAL_CAP        = 0.60
        self.VERTICAL_RANGE      = 1.20
        self.ASCENT_TOL          = 1e-6
        self.smooth_alpha        = 0.40
        self.WAKE_RADIUS         = 4.0    # m — wake turbulence effect range

        # -- Terrain -------------------------------------------------------
        if not os.path.exists(TERRAIN_PATH):
            raise FileNotFoundError("Missing terrain.npy — run generate_terrain.py")
        self.terrain = np.load(TERRAIN_PATH)

        # -- Static obstacles ----------------------------------------------
        if os.path.exists(STATIC_PATH):
            self.static = np.load(STATIC_PATH)
        else:
            self.static = np.zeros((self.GX, self.GY, self.GZ), dtype=bool)

        # -- Dynamic obstacles (full-3D mode) ------------------------------
        self.dyn = DynamicObstacle(
            grid_size=self.grid, count=dyn_count, move_mode="full_3d"
        )

        # -- Physics subsystems --------------------------------------------
        self.wind_field   = WindField(self.GX, self.GY, seed=42)
        self.thermal_field= ThermalField(self.terrain, seed=7)

        # -- Recharge stations ---------------------------------------------
        self.recharge_xy = RECHARGE_STATIONS_XY
        self._build_recharge_3d()

        # -- Drone start positions -----------------------------------------
        starts_xy = list(DRONE_STARTS_XY[:n_drones])
        while len(starts_xy) < n_drones:
            starts_xy.append((
                np.random.randint(5, self.GX - 5),
                np.random.randint(5, self.GY - 5),
            ))
        self.starts_xy = starts_xy
        self._build_start_3d()

        # -- Coverage grid -------------------------------------------------
        self.total_cells   = self.GX * self.GY
        self.coverage_grid = np.zeros((self.GX, self.GY), dtype=bool)

        xs = np.arange(self.GX) + 0.5
        ys = np.arange(self.GY) + 0.5
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        self.cell_centers = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (GX*GY, 2)

        # -- Gym spaces ----------------------------------------------------
        obs_dim = 14 * self.n_drones   # +3 for wind obs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_drones * 3,), dtype=np.float32
        )

        # -- Per-drone runtime state ---------------------------------------
        self.positions      = None   # (N, 3)
        self.velocities     = None   # (N, 3)
        self.batteries      = None   # (N,)
        self.motor_health   = None   # (N,) ? [0.5, 1.0]  — 1.0 = perfect
        self.phases         = None   # list[str]
        self.progress_wins  = None   # deques for anti-stuck
        self.step_count     = 0

        # Anti-stuck params
        self.stuck_window = 14
        self.stuck_thresh = 0.006
        self.stuck_boost  = 0.28

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _build_start_3d(self):
        self._start_3d = []
        for (sx, sy) in self.starts_xy:
            sx = int(np.clip(sx, 0, self.GX - 1))
            sy = int(np.clip(sy, 0, self.GY - 1))
            z  = min(float(self.terrain[sx, sy]) + 0.8, self.CRUISE_Z - 0.5)
            self._start_3d.append(
                np.array([sx + 0.5, sy + 0.5, z], dtype=float)
            )

    def _build_recharge_3d(self):
        self.recharge_3d = []
        for (rx, ry) in self.recharge_xy:
            rx = int(np.clip(rx, 0, self.GX - 1))
            ry = int(np.clip(ry, 0, self.GY - 1))
            self.recharge_3d.append(
                np.array([rx + 0.5, ry + 0.5, 3.5], dtype=float)
            )

    def _clip_pos(self, p):
        p = p.copy()
        p[0] = np.clip(p[0], 0.1, self.GX - 0.1)
        p[1] = np.clip(p[1], 0.1, self.GY - 0.1)
        p[2] = np.clip(p[2], 0.3, self.GZ - 0.3)
        return p

    def _min_dist_obstacles(self, pos) -> float:
        min_d = float("inf")
        idx = np.argwhere(self.static)
        if idx.size > 0:
            centers = idx.astype(float) + 0.5
            dists   = np.linalg.norm(centers - pos.reshape(1, 3), axis=1)
            min_d   = min(min_d, float(np.min(dists)))
        if hasattr(self.dyn, "positions") and self.dyn.positions:
            dpos  = np.array(self.dyn.positions)
            dists = np.linalg.norm(dpos - pos.reshape(1, 3), axis=1)
            min_d = min(min_d, float(np.min(dists)))
        return min_d

    def _repulsion_vector(self, pos) -> np.ndarray:
        """Compute repulsion vector away from the nearest obstacle."""
        repulse = np.zeros(3, dtype=float)
        idx_s = np.argwhere(self.static)
        if idx_s.size > 0:
            centers = idx_s.astype(float) + 0.5
            diffs   = pos - centers
            dists   = np.linalg.norm(diffs, axis=1)
            n       = np.argmin(dists)
            repulse += diffs[n] / (dists[n] + 1e-9) * 0.35
        if hasattr(self.dyn, "positions") and self.dyn.positions:
            dpos  = np.array(self.dyn.positions)
            ddiff = pos - dpos
            dd    = np.linalg.norm(ddiff, axis=1)
            n     = np.argmin(dd)
            repulse += ddiff[n] / (dd[n] + 1e-9) * 0.35
        return repulse

    def _update_coverage(self, positions_2d: np.ndarray) -> int:
        r2  = self.sensor_radius ** 2
        new = 0
        for pos2 in positions_2d:
            diffs  = self.cell_centers - pos2.reshape(1, 2)
            dists2 = (diffs ** 2).sum(axis=1)
            hits   = np.where(dists2 <= r2)[0]
            for h in hits:
                xi = h // self.GY
                yi = h  % self.GY
                if not self.coverage_grid[xi, yi]:
                    self.coverage_grid[xi, yi] = True
                    new += 1
        return new

    def _nearest_recharge(self, pos) -> np.ndarray:
        dists = [np.linalg.norm(pos[:2] - rs[:2]) for rs in self.recharge_3d]
        return self.recharge_3d[int(np.argmin(dists))]

    def coverage_ratio(self) -> float:
        return float(np.sum(self.coverage_grid)) / self.total_cells

    def _ground_effect_factor(self, pos) -> float:
        """
        Ground effect: near terrain the drone gets extra lift (reduced drag).
        Returns a factor in [0.5, 1.0] — lower near ground (less drag).
        """
        xi = int(np.clip(pos[0], 0, self.GX - 1))
        yi = int(np.clip(pos[1], 0, self.GY - 1))
        ground_z = float(self.terrain[xi, yi])
        height_agl = pos[2] - ground_z   # height above ground level
        # Ground effect kicks in below ~2 m AGL
        if height_agl < 2.0:
            return max(0.5, height_agl / 2.0)
        return 1.0

    def _air_density_factor(self, z: float) -> float:
        """
        Air density decreases with altitude. Simplified barometric model.
        Returns fraction of sea-level density (~= 1.0 at z=0, ~0.80 at z=15).
        """
        return math.exp(-z / 80.0)   # gentle decay over our 15 m envelope

    def _compute_wake_turbulence(self, drone_idx: int) -> np.ndarray:
        """
        Other drones nearby disturb airflow -> random perturbation.
        Effect decays with distance.
        """
        wake = np.zeros(3, dtype=float)
        pos_i = self.positions[drone_idx]
        for j in range(self.n_drones):
            if j == drone_idx:
                continue
            d = np.linalg.norm(pos_i - self.positions[j])
            if d < self.WAKE_RADIUS and d > 0.1:
                strength = 0.06 * (1.0 - d / self.WAKE_RADIUS)
                wake += np.random.normal(0.0, strength, size=3)
        return wake

    # ======================================================================
    # Gym API
    # ======================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_static:
            try:
                self.static = generate_static_obstacles()
            except Exception as e:
                print(f"[WARN] generate_static_obstacles failed: {e}")

        if not getattr(self.dyn, "initialized", False):
            self.dyn.initialize(static=self.static, terrain=self.terrain)
        else:
            self.dyn.initialize(static=self.static, terrain=self.terrain)

        self.coverage_grid = np.zeros((self.GX, self.GY), dtype=bool)
        self.step_count    = 0

        # Reset physics subsystems
        self.wind_field    = WindField(self.GX, self.GY,
                                       seed=int(np.random.randint(0, 9999)))
        self.thermal_field = ThermalField(self.terrain,
                                           seed=int(np.random.randint(0, 9999)))

        # Drone state
        self.positions    = []
        for sp in self._start_3d:
            j = np.random.uniform(-0.4, 0.4, size=3); j[2] = 0.0
            self.positions.append(self._clip_pos(sp + j))
        self.positions    = np.array(self.positions, dtype=float)
        self.velocities   = np.zeros((self.n_drones, 3), dtype=float)
        self.batteries    = np.full(self.n_drones, self.battery_capacity, dtype=float)
        self.motor_health = np.ones(self.n_drones, dtype=float)  # perfect at start
        self.phases        = ["ascend"] * self.n_drones
        self.pre_recharge_phase = ["cruise"] * self.n_drones
        self.progress_wins = [deque(maxlen=self.stuck_window) for _ in range(self.n_drones)]

        # -- Cumulative multi-objective trackers (reset each episode) ------
        self.total_path_length   = np.zeros(self.n_drones, dtype=float)
        self.total_battery_used  = np.zeros(self.n_drones, dtype=float)
        self.total_collisions    = np.zeros(self.n_drones, dtype=int)

        self._update_coverage(self.positions[:, :2])
        return self._get_obs(), {}

    # ----------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action = np.clip(
            np.array(action, dtype=float).reshape(self.n_drones, 3),
            -1.0, 1.0
        )

        # -- Advance global physics -------------------------------------
        if self.wind_enabled:
            self.wind_field.step()
        if self.thermal_enabled:
            self.thermal_field.step()

        # Move dynamic obstacles (full 3D)
        self.dyn.move(static=self.static, terrain=self.terrain)

        reward = -0.004 * self.n_drones  # tiny step cost
        step_path_total    = 0.0
        step_battery_total = 0.0
        step_collision_total = 0

        for i in range(self.n_drones):
            if self.batteries[i] <= 0.0:
                continue

            prev_pos = self.positions[i].copy()
            phase    = self.phases[i]
            mh       = self.motor_health[i]   # 1.0 = healthy; <1 = degraded

            # -- Phase-based desired velocity (same logic as Drone3DEnv) --
            if phase == "ascend":
                climb = min(self.max_speed * mh,
                            self.CRUISE_Z - self.positions[i, 2])
                climb = max(climb, 0.0)
                desired_vel = np.array([0.0, 0.0, climb], dtype=float)

            elif phase == "cruise":
                dx, dy, dz = action[i] * self.max_speed * mh
                dx = np.clip(dx, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
                dy = np.clip(dy, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
                dz = np.clip(dz, -self.VERTICAL_CAP,        self.VERTICAL_CAP)

                # Soft altitude lock
                next_z = self.positions[i, 2] + dz
                if next_z < self.CRUISE_Z - self.VERTICAL_RANGE or \
                   next_z > self.CRUISE_Z + self.VERTICAL_RANGE:
                    dz = 0.0

                desired_vel = np.array([dx, dy, dz], dtype=float)

                # Anti-stuck -> steer toward nearest uncovered cell
                if (len(self.progress_wins[i]) >= self.stuck_window and
                        float(np.mean(self.progress_wins[i])) < self.stuck_thresh):
                    uncov = np.argwhere(~self.coverage_grid)
                    if uncov.size > 0:
                        t = uncov[np.random.randint(len(uncov))].astype(float) + 0.5
                        d = t - self.positions[i, :2]
                        d /= (np.linalg.norm(d) + 1e-9)
                        nudge = np.array([d[0], d[1], 0.0]) * self.stuck_boost
                        prop  = self._clip_pos(self.positions[i] + nudge)
                        if self._min_dist_obstacles(prop) > 0.4:
                            self.positions[i] = prop
                            self.progress_wins[i].clear()

            elif phase == "recharge_descend":
                rs      = self._nearest_recharge(self.positions[i])
                vec_xy  = rs[:2] - self.positions[i, :2]
                dist_xy = np.linalg.norm(vec_xy) + 1e-9
                corr_xy = (vec_xy / dist_xy) * min(dist_xy, self.max_speed * mh)
                dz      = np.clip(rs[2] - self.positions[i, 2],
                                   -self.max_speed, self.max_speed)
                desired_vel = np.array([corr_xy[0], corr_xy[1], dz], dtype=float)

            elif phase == "return_home":
                dx, dy, dz = action[i] * self.max_speed * mh
                dx = np.clip(dx, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
                dy = np.clip(dy, -self.HORIZONTAL_STEP_CAP, self.HORIZONTAL_STEP_CAP)
                dz = np.clip(dz, -self.VERTICAL_CAP,        self.VERTICAL_CAP)

                # Soft altitude lock
                next_z = self.positions[i, 2] + dz
                if next_z < self.CRUISE_Z - self.VERTICAL_RANGE or \
                   next_z > self.CRUISE_Z + self.VERTICAL_RANGE:
                    dz = 0.0

                desired_vel = np.array([dx, dy, dz], dtype=float)

            elif phase == "descend_land":
                start_xy = self.starts_xy[i]
                target_xy = np.array([start_xy[0] + 0.5, start_xy[1] + 0.5])
                diff_xy = target_xy - self.positions[i, :2]
                dist_xy = np.linalg.norm(diff_xy) + 1e-9
                corr_xy = (diff_xy / dist_xy) * min(dist_xy, self.max_speed * mh)

                ground_z = float(self.terrain[int(np.clip(self.positions[i, 0], 0, self.GX - 1)),
                                             int(np.clip(self.positions[i, 1], 0, self.GY - 1))])
                land_target_z = ground_z + 0.5
                dz = np.clip(land_target_z - self.positions[i, 2], -0.15, 0.15)
                desired_vel = np.array([corr_xy[0], corr_xy[1], dz], dtype=float)
            else:
                desired_vel = np.zeros(3, dtype=float)

            # -- Real-world physics perturbations --------------------------

            # ? Wind force
            wind_vec = np.zeros(3)
            if self.wind_enabled:
                wind_vec = self.wind_field.at(self.positions[i])

            # ? Aerodynamic drag (opposes velocity, proportional to v2)
            #    drag = drag_coeff * air_density * speed * velocity
            speed = np.linalg.norm(desired_vel) + 1e-9
            rho   = self._air_density_factor(self.positions[i, 2])
            ge    = self._ground_effect_factor(self.positions[i])
            drag  = -self.drag_coeff * rho * ge * speed * desired_vel

            # ? Thermal updraft / downdraft (vertical only)
            thermal_dz = 0.0
            if self.thermal_enabled:
                thermal_dz = self.thermal_field.at(self.positions[i])
            thermal_vec = np.array([0.0, 0.0, thermal_dz])

            # ? Wake turbulence from neighbouring drones
            wake = self._compute_wake_turbulence(i)

            # -- Combine: control + wind + drag + thermal + wake -----------
            effective_vel = desired_vel + wind_vec + drag + thermal_vec + wake

            # -- Action smoothing (low-pass, same as original) -------------
            effective_vel = (
                self.smooth_alpha * effective_vel
                + (1.0 - self.smooth_alpha) * self.velocities[i]
            )

            # -- Obstacle proximity -> autopilot repulsion override ---------
            tentative = self._clip_pos(self.positions[i] + effective_vel)
            if self._min_dist_obstacles(tentative) < self.safety_radius * 0.65:
                effective_vel = np.clip(
                    self._repulsion_vector(self.positions[i]),
                    -self.max_speed, self.max_speed
                )

            # -- Integrate position ----------------------------------------
            self.velocities[i] = effective_vel
            new_pos = self._clip_pos(self.positions[i] + effective_vel)

            # -- Collision check (Objective 1: collision avoidance) --------
            collided = False
            if self._min_dist_obstacles(new_pos) < 0.40:
                reward  -= self.w_collision
                new_pos  = self.positions[i].copy()
                collided = True
                self.total_collisions[i] += 1
                step_collision_total     += 1

            # -- Inter-drone collision (disabled near designated Home landing pad or during return flight to allow safe returns) --
            in_home_zone = (np.linalg.norm(new_pos[:2] - np.array([5.0, 5.0])) < 12.0)
            is_returning = (phase in ["return_home", "descend_land", "landed"])
            if not in_home_zone and not is_returning:
                for j in range(self.n_drones):
                    if j != i:
                        other_returning = (self.phases[j] in ["return_home", "descend_land", "landed"])
                        if not other_returning:
                            sep = np.linalg.norm(new_pos - self.positions[j])
                            if sep < self.safety_radius:
                                reward -= 4.0
                                if not collided:
                                    self.total_collisions[i] += 1
                                    step_collision_total     += 1

            # If drone is already landed safely, lock position and zero out velocity
            if phase == "descend_land":
                ground_z = float(self.terrain[int(np.clip(self.positions[i, 0], 0, self.GX - 1)),
                                             int(np.clip(self.positions[i, 1], 0, self.GY - 1))])
                if abs(self.positions[i, 2] - (ground_z + 0.5)) < 0.2:
                    self.positions[i, 2] = ground_z + 0.5
                    self.velocities[i] = np.zeros(3)
                    new_pos = self.positions[i].copy()

            self.positions[i] = new_pos

            # -- Path length tracking (Objective 2: shortest path) ---------
            step_dist = float(np.linalg.norm(new_pos - prev_pos))
            if phase == "descend_land" and abs(self.positions[i, 2] - (ground_z + 0.5)) < 0.2:
                step_dist = 0.0
            self.total_path_length[i] += step_dist
            step_path_total           += step_dist
            reward -= self.w_path * step_dist   # penalise unnecessary travel

            # -- Battery drain (Objective 3: battery conservation) ---------
            #    Drain prop to thrust2 + wind resistance load
            wind_load      = 0.5 * np.linalg.norm(wind_vec)
            speed_load     = 0.002 * np.linalg.norm(effective_vel) ** 2
            drain_this_step = self.battery_drain_base + wind_load + speed_load
            if phase == "descend_land" and abs(self.positions[i, 2] - (ground_z + 0.5)) < 0.2:
                drain_this_step = 0.0
            self.batteries[i] = max(0.0, self.batteries[i] - drain_this_step)
            self.total_battery_used[i] += drain_this_step
            step_battery_total         += drain_this_step
            reward -= self.w_battery * drain_this_step  # penalise battery use

            # -- Motor health degradation (very slow random wear) ----------
            if np.random.random() < self.motor_fail_prob:
                self.motor_health[i] = max(
                    0.5, self.motor_health[i] - np.random.uniform(0.02, 0.08)
                )

            # Progress tracking
            self.progress_wins[i].append(np.linalg.norm(new_pos - prev_pos))

            # -- Phase transitions -----------------------------------------
            if phase == "ascend":
                if self.positions[i, 2] >= self.CRUISE_Z - self.ASCENT_TOL:
                    self.positions[i, 2] = self.CRUISE_Z
                    self.velocities[i, 2] = 0.0
                    self.phases[i] = "cruise"

            elif phase == "cruise" or phase == "return_home":
                if self.batteries[i] < 30.0:
                    self.pre_recharge_phase[i] = phase
                    self.phases[i] = "recharge_descend"

            elif phase == "recharge_descend":
                rs    = self._nearest_recharge(self.positions[i])
                horiz = np.linalg.norm(self.positions[i, :2] - rs[:2])
                vert  = abs(self.positions[i, 2] - rs[2])
                if horiz < 0.5 and vert < 0.5:
                    self.batteries[i]        = self.battery_capacity
                    self.positions[i]        = np.array([rs[0], rs[1], self.CRUISE_Z])
                    self.velocities[i, :]    = 0.0
                    self.phases[i]           = self.pre_recharge_phase[i]
                    print(f"[INFO] Drone {i} recharged. Resuming phase: {self.phases[i]}")

            elif phase == "return_home":
                # Check if arrived at home coords
                start_xy = self.starts_xy[i]
                diff = np.array([start_xy[0] + 0.5, start_xy[1] + 0.5]) - self.positions[i, :2]
                if np.linalg.norm(diff) < 3.5:
                    self.phases[i] = "descend_land"
                    print(f"[INFO] Drone {i} returned home. Starting landing sequence...")

            elif phase == "descend_land":
                pass  # Landing handled in the integration overrides above

        # -- Team coverage reward -------------------------------------------
        new_cells = self._update_coverage(self.positions[:, :2])
        reward   += new_cells * 2.0

        # -- Spread bonus ---------------------------------------------------
        if self.n_drones > 1:
            centroid = self.positions.mean(axis=0)
            spread   = float(np.mean(np.linalg.norm(self.positions - centroid, axis=1)))
            reward  += 0.02 * spread

        # Terminate immediately when all active/available drones have successfully returned home AND touched down on the ground
        all_landed = True
        for i in range(self.n_drones):
            # Exclude dead/depleted drones from blocking early shutdown
            if self.batteries[i] <= 0.5:
                continue
            if self.phases[i] != "descend_land":
                all_landed = False
                break
            # Check if drone is close to ground level (terrain height + 0.5m safety landing height)
            ground_z = float(self.terrain[int(np.clip(self.positions[i, 0], 0, self.GX - 1)),
                                         int(np.clip(self.positions[i, 1], 0, self.GY - 1))])
            if abs(self.positions[i, 2] - (ground_z + 0.5)) > 0.25:
                all_landed = False
                break

        coverage   = self.coverage_ratio()
        truncated  = self.step_count >= self.max_steps
        terminated = bool(coverage >= 0.995 or all_landed)

        # Current wind speed for logging
        wind_speed = float(np.linalg.norm(
            self.wind_field.at(np.array([50.0, 50.0, CRUISE_Z]))
        )) if self.wind_enabled else 0.0

        # Multi-objective step penalties (already in reward above)
        reward -= self.w_path      * step_path_total      * 0.0   # already applied per drone
        # (no double-counting; the per-drone penalties above are the authoritative ones)

        info = {
            # -- Primary objective --------------------------------------
            "coverage_ratio":      coverage,
            "coverage_pct":        coverage * 100.0,
            "new_cells":           new_cells,
            "step":                self.step_count,
            # -- Multi-objective metrics --------------------------------
            "total_collisions":    int(self.total_collisions.sum()),
            "total_path_length":   float(self.total_path_length.sum()),
            "total_battery_used":  float(self.total_battery_used.sum()),
            "per_drone_collisions":self.total_collisions.tolist(),
            "per_drone_path":      self.total_path_length.tolist(),
            "per_drone_battery":   self.total_battery_used.tolist(),
            # -- Step-level metrics -------------------------------------
            "step_collisions":     step_collision_total,
            "step_path":           step_path_total,
            "step_battery":        step_battery_total,
            # -- Environment state --------------------------------------
            "phases":              list(self.phases),
            "batteries":           self.batteries.tolist(),
            "motor_health":        self.motor_health.tolist(),
            "wind_speed":          wind_speed,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        if self.positions is None:
            return np.zeros(14 * self.n_drones, dtype=np.float32)

        centroid  = self.positions.mean(axis=0)
        cov_ratio = self.coverage_ratio()
        parts = []

        for i in range(self.n_drones):
            pos_n  = self.positions[i]  / np.array([self.GX, self.GY, self.GZ], dtype=float)
            vel_n  = self.velocities[i] / (self.max_speed + 1e-9)
            bat_n  = np.array([self.batteries[i] / self.battery_capacity])
            cov_n  = np.array([cov_ratio])
            rel_c  = (self.positions[i] - centroid) / (self.GX + 1e-9)

            # Wind observation (normalised, so agent can anticipate gusts)
            if self.wind_enabled:
                w_raw  = self.wind_field.at(self.positions[i])
                wind_n = w_raw / 0.6   # normalise by max plausible wind
            else:
                wind_n = np.zeros(3)

            part = np.concatenate([pos_n, vel_n, bat_n, cov_n, rel_c, wind_n])

            # ? GPS / sensor noise on observation (not on true state)
            if self.sensor_noise_std > 0.0:
                part = part + np.random.normal(0.0, self.sensor_noise_std, size=part.shape)

            parts.append(part)

        return np.concatenate(parts).astype(np.float32)

    # ----------------------------------------------------------------------
    def render(self):
        pass

    def close(self):
        pass


# ==========================================================================
# Smoke test
# ==========================================================================
if __name__ == "__main__":
    print("[smoke] MultiDroneCoverageEnv — 6 drones, 100x100x15, full physics")
    env = MultiDroneCoverageEnv(n_drones=6, wind_enabled=True, thermal_enabled=True)
    obs, _ = env.reset()
    print(f"  obs shape  : {obs.shape}")
    print(f"  action dim : {env.action_space.shape}")

    total_rew = 0.0
    for _ in range(400):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)
        total_rew += rew
        if term or trunc:
            break

    print(f"  steps        : {env.step_count}")
    print(f"  coverage     : {info['coverage_pct']:.2f}%")
    print(f"  wind speed   : {info['wind_speed']:.4f}")
    print(f"  motor health : {[f'{h:.3f}' for h in info['motor_health']]}")
    print(f"  batteries    : {[f'{b:.1f}' for b in info['batteries']]}")
    print(f"  phases       : {info['phases']}")
    print(f"  total_rew    : {total_rew:.2f}")
    print("[smoke] PASSED [OK]")
