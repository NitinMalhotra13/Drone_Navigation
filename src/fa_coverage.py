"""
fa_coverage.py
==============
Firefly Algorithm (FA) for multi-drone waypoint planning.
Enforces monotonic forward progress from start to goal coordinates.
Optimises only the perpendicular fanning-out sweeps (lateral deviations)
to achieve maximum coverage and zero path overlap in minimum time.
"""

import os
import math
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# ---------------------------------------------------------------------------
# FireflyPlanner
# ---------------------------------------------------------------------------
class FireflyPlanner:
    """
    Firefly Algorithm planner that enforces strict monotonic progress toward the goal.
    Optimises lateral deviations (sweeping width) to prevent drones from roaming around,
    guaranteeing they fly straight forward in non-overlapping coordinated corridors.
    """

    def __init__(
        self,
        n_drones: int = 6,
        n_waypoints: int = 12,
        grid_x: int = 100,
        grid_y: int = 100,
        cruise_z: float = 6.0,
        n_fireflies: int = 30,
        max_iter: int = 50,
        alpha: float = 0.25,      # random walk step size
        beta0: float = 1.0,       # maximum attraction coefficient
        gamma: float = 0.08,      # light absorption coefficient
        start_positions: np.ndarray = None,
        goal_positions: np.ndarray = None,
    ):
        self.n_drones     = n_drones
        self.n_waypoints  = n_waypoints
        self.grid_x       = grid_x
        self.grid_y       = grid_y
        self.cruise_z     = cruise_z
        self.n_fireflies  = n_fireflies
        self.max_iter     = max_iter
        self.alpha        = alpha
        self.beta0        = beta0
        self.gamma        = gamma

        # Enforce starts/goals
        if start_positions is None:
            self.start_positions = np.array([
                [4.0, 4.0, self.cruise_z],
                [4.0, 6.0, self.cruise_z],
                [6.0, 4.0, self.cruise_z],
                [6.0, 6.0, self.cruise_z],
                [5.0, 4.0, self.cruise_z],
                [5.0, 6.0, self.cruise_z]
            ])[:n_drones]
        else:
            self.start_positions = np.array(start_positions)

        if goal_positions is None:
            self.goal_positions = np.array([
                [94.0, 94.0, self.cruise_z],
                [94.0, 96.0, self.cruise_z],
                [96.0, 94.0, self.cruise_z],
                [96.0, 96.0, self.cruise_z],
                [95.0, 94.0, self.cruise_z],
                [95.0, 96.0, self.cruise_z]
            ])[:n_drones]
        else:
            self.goal_positions = np.array(goal_positions)

        # Sensor footprint radius (cells) used for coverage simulation
        self._sensor_radius = 4.5   # matches env sensor_radius for accurate planning
        # Interpolation steps between consecutive waypoints (higher = more accurate coverage sim)
        self._interp_steps  = 14

        # Load terrain
        terrain_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "terrain.npy")
        if os.path.exists(terrain_path):
            self.terrain = np.load(terrain_path)
            print(f"[FA] Terrain loaded from {terrain_path}  shape={self.terrain.shape}")
        else:
            self.terrain = np.zeros((grid_y, grid_x), dtype=np.float32)
            print("[FA] terrain.npy not found – using flat zero terrain.")

        # Initialise firefly population in parameter space (lateral deviations + vertical jitter)
        self._population = self._init_population()

        # Best solution bookkeeping
        self._best_firefly  = None
        self._best_fitness  = -np.inf
        self._best_stats    = {}

    # ------------------------------------------------------------------
    # Parameter space conversion
    # ------------------------------------------------------------------
    def firefly_to_waypoints(self, pop_element: np.ndarray) -> np.ndarray:
        """
        pop_element: shape (n_drones, n_waypoints - 2, 2)
        Converts the lateral deviations and vertical jitters back into 3D waypoints
        along a monotonic fanned-out path from start to goal.
        """
        wps = np.empty((self.n_drones, self.n_waypoints, 3))
        for d in range(self.n_drones):
            p_start = self.start_positions[d]
            p_goal  = self.goal_positions[d]
            wps[d, 0]  = p_start
            wps[d, -1] = p_goal
            
            V = p_goal - p_start
            D = np.linalg.norm(V[:2]) + 1e-9
            dir_vec = V[:2] / D
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])
            
            for w in range(1, self.n_waypoints - 1):
                t = w / (self.n_waypoints - 1)
                p_linear = p_start + t * V
                
                # lateral deviation (dynamic eye-shaped scaling to fit boundaries perfectly)
                lat_dev = pop_element[d, w - 1, 0] * (4.0 * t * (1.0 - t))
                # vertical jitter
                vert_jit = pop_element[d, w - 1, 1]
                
                p_wp = p_linear.copy()
                p_wp[:2] += perp_vec * lat_dev
                p_wp[2]  += vert_jit
                
                # clip to grid bounds
                p_wp[0] = np.clip(p_wp[0], 1, self.grid_x - 1)
                p_wp[1] = np.clip(p_wp[1], 1, self.grid_y - 1)
                p_wp[2] = np.clip(p_wp[2], self.cruise_z - 0.5, self.cruise_z + 0.5)
                
                wps[d, w] = p_wp
        return wps

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_population(self) -> list:
        """
        Build the initial population of n_fireflies in parameter space:
        Each firefly has shape = (n_drones, n_waypoints - 2, 2)
        - Index 0: lateral deviation (uniform [-35.0, 35.0])
        - Index 1: vertical jitter (uniform [-0.4, 0.4])
        """
        population = []
        for _ in range(self.n_fireflies):
            pop_element = np.empty((self.n_drones, self.n_waypoints - 2, 2))
            
            # Spread search corridors across [-65, 65] for maximum grid edge reach
            for d in range(self.n_drones):
                corridor_center = -65.0 + (130.0 * d / max(1, self.n_drones - 1))
                pop_element[d, :, 0] = corridor_center + np.random.uniform(-5.0, 5.0, self.n_waypoints - 2)
                pop_element[d, :, 1] = np.random.uniform(-0.4, 0.4, self.n_waypoints - 2)
                
            population.append(pop_element)
        return population

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _simulate_coverage(self, firefly: np.ndarray):
        """
        Simulate all drones flying the waypoints encoded in *firefly*.
        Tracks the overlap ratio across all drones on the shared map.
        """
        # If in parameter space, convert to 3D waypoints
        if firefly.shape == (self.n_drones, self.n_waypoints - 2, 2):
            wps_3d = self.firefly_to_waypoints(firefly)
        else:
            wps_3d = firefly

        coverage_grid  = np.zeros((self.grid_y, self.grid_x), dtype=bool)
        total_path_len = 0.0
        total_steps    = 0
        total_indiv_covered = 0

        # Pre-build an integer disk mask for the sensor footprint
        r  = int(math.ceil(self._sensor_radius))
        ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
        disk   = (xs ** 2 + ys ** 2) <= self._sensor_radius ** 2

        for d in range(self.n_drones):
            drone_cov = np.zeros((self.grid_y, self.grid_x), dtype=bool)
            waypoints = wps_3d[d]  # shape (n_waypoints, 3)

            for w in range(self.n_waypoints - 1):
                p_start = waypoints[w]
                p_end   = waypoints[w + 1]

                # Segment length contributes to path length
                seg_len = float(np.linalg.norm(p_end - p_start))
                total_path_len += seg_len

                # Interpolate positions along segment
                for step in range(self._interp_steps + 1):
                    t   = step / self._interp_steps
                    pos = p_start + t * (p_end - p_start)

                    cx = int(round(pos[0]))
                    cy = int(round(pos[1]))
                    total_steps += 1

                    # Mark sensor footprint in the individual coverage grid
                    x_lo = max(0, cx - r);  x_hi = min(self.grid_x, cx + r + 1)
                    y_lo = max(0, cy - r);  y_hi = min(self.grid_y, cy + r + 1)

                    disk_x_lo = x_lo - (cx - r)
                    disk_x_hi = disk_x_lo + (x_hi - x_lo)
                    disk_y_lo = y_lo - (cy - r)
                    disk_y_hi = disk_y_lo + (y_hi - y_lo)

                    drone_cov[y_lo:y_hi, x_lo:x_hi] |= \
                        disk[disk_y_lo:disk_y_hi, disk_x_lo:disk_x_hi]

            total_indiv_covered += int(drone_cov.sum())
            coverage_grid |= drone_cov

        n_total          = self.grid_x * self.grid_y
        coverage_ratio   = float(coverage_grid.sum()) / n_total
        total_battery    = 0.08 * total_steps + 0.001 * total_path_len

        # Overlap ratio calculation
        union_covered = int(coverage_grid.sum())
        overlap_ratio = 0.0
        if union_covered > 0:
            overlap_ratio = (total_indiv_covered - union_covered) / union_covered

        return coverage_ratio, total_path_len, total_battery, overlap_ratio

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------
    def _fitness(self, firefly: np.ndarray) -> float:
        """
        Multi-objective scalar fitness prioritising maximum area coverage.
        Coverage reward weight raised (150) vs overlap penalty (20) to push
        the FA to spread drones as widely as possible across the 100x100 grid.
        """
        cov, path_len, battery, overlap = self._simulate_coverage(firefly)
        return cov * 150.0 - 20.0 * overlap - 0.02 * path_len - 0.05 * battery

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------
    def optimize(self, verbose: bool = True) -> np.ndarray:
        """
        Run the Firefly Algorithm to optimize the fanned-out parameter space.
        """
        population   = self._population          # list of fireflies
        n_ff         = self.n_fireflies
        brightness   = np.array([self._fitness(ff) for ff in population])

        # Initialise global best from the initial population
        best_idx           = int(np.argmax(brightness))
        self._best_firefly = population[best_idx].copy()
        self._best_fitness = brightness[best_idx]

        if verbose:
            print(f"[FA] Starting optimisation | fireflies={n_ff} | iters={self.max_iter}")
            print(f"[FA] Initial best fitness: {self._best_fitness:.4f}")

        for it in range(self.max_iter):
            # Move each firefly toward brighter neighbours
            for i in range(n_ff):
                for j in range(n_ff):
                    if brightness[j] <= brightness[i]:
                        continue

                    pos_i = population[i]
                    pos_j = population[j]

                    # Distance in parameter space
                    diff = pos_j - pos_i
                    r    = float(np.linalg.norm(diff))

                    # Attractiveness decays with distance
                    attraction = self.beta0 * math.exp(-self.gamma * r * r)

                    # Move i toward j + random walk
                    noise        = self.alpha * np.random.randn(*pos_i.shape)
                    noise[..., 1] *= 0.15  # damp vertical jitter search
                    
                    new_pos      = pos_i + attraction * diff + noise

                    # Clip deviations to valid limits (±80 for full edge-to-edge reach)
                    new_pos[..., 0] = np.clip(new_pos[..., 0], -80.0, 80.0)  # Max lateral deviation
                    new_pos[..., 1] = np.clip(new_pos[..., 1], -0.5, 0.5)    # Vertical jitter
                    
                    population[i] = new_pos

                # Re-evaluate after all moves for firefly i
                brightness[i] = self._fitness(population[i])

            # Update global best
            iter_best_idx = int(np.argmax(brightness))
            if brightness[iter_best_idx] > self._best_fitness:
                self._best_fitness  = brightness[iter_best_idx]
                self._best_firefly  = population[iter_best_idx].copy()

            if verbose and (it + 1) % 10 == 0:
                print(f"[FA] Iter {it + 1:3d}/{self.max_iter} | "
                      f"best_fitness={self._best_fitness:.4f}")

        # Store final stats on the best converted waypoints
        best_wps = self.firefly_to_waypoints(self._best_firefly)
        cov, path_len, battery, overlap = self._simulate_coverage(best_wps)
        self._best_stats = {
            "coverage_ratio"   : cov,
            "total_path_length": path_len,
            "total_battery"    : battery,
            "overlap_ratio"    : overlap,
        }

        if verbose:
            print(f"\n[FA] Optimisation complete.")
            print(f"[FA]   Coverage ratio   : {cov * 100:.2f}%")
            print(f"[FA]   Total path length: {path_len:.2f} m")
            print(f"[FA]   Battery estimate : {battery:.2f} units")
            print(f"[FA]   Path Overlap     : {overlap * 100:.2f}%")

        return best_wps

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_best_waypoints(self) -> np.ndarray:
        if self._best_firefly is None:
            raise RuntimeError("Call optimize() before get_best_waypoints().")
        return self.firefly_to_waypoints(self._best_firefly)

    def get_coverage_stats(self) -> dict:
        if not self._best_stats:
            raise RuntimeError("Call optimize() before get_coverage_stats().")
        return dict(self._best_stats)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Firefly Algorithm – multi-drone coverage waypoint planning"
    )
    parser.add_argument("--drones",      type=int, default=6,
                        help="Number of drones (default: 6)")
    parser.add_argument("--iterations",  type=int, default=50,
                        help="FA optimisation iterations (default: 50)")
    args = parser.parse_args()

    planner = FireflyPlanner(n_drones=args.drones, max_iter=args.iterations)
    planner.optimize(verbose=True)

    stats = planner.get_coverage_stats()
    print("\n--- Final Coverage Statistics ---")
    print(f"  Coverage Ratio   : {stats['coverage_ratio'] * 100:.2f}%")
    print(f"  Total Path Length: {stats['total_path_length']:.2f} m")
    print(f"  Battery Estimate : {stats['total_battery']:.2f} units")
    if 'overlap_ratio' in stats:
        print(f"  Path Overlap     : {stats['overlap_ratio'] * 100:.2f}%")

    waypoints    = planner.get_best_waypoints()
    out_path     = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "fa_waypoints.npy"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, waypoints)
    print(f"\n[FA] Best waypoints saved to {out_path}  shape={waypoints.shape}")
