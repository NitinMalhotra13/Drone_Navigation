"""
ra_coverage.py
==============
Raven Roosting Algorithm (RRA) for adaptive mid-mission waypoint replanning.
Enforces strict forward progress toward the final goals. Optimises lateral
sweeping widths (corridor deviations) dynamically based on remaining uncovered areas,
preventing the drones from roaming or circling back during updates.
"""

import os
import math
import numpy as np

# ---------------------------------------------------------------------------
# RavenReplanner
# ---------------------------------------------------------------------------
class RavenReplanner:
    """
    Raven Roosting Algorithm replanner that dynamically adjusts fanned-out corridors
    during flight while guaranteeing strict monotonic progress toward destination goal posts.
    """

    def __init__(
        self,
        n_drones: int   = 6,
        grid_x:   int   = 100,
        grid_y:   int   = 100,
        cruise_z: float = 6.0,
        n_ravens: int   = 20,
        max_iter: int   = 30,
        pa_init:  float = 0.9,
        pa_final: float = 0.1,
        goal_positions: np.ndarray = None,
    ):
        self.n_drones  = n_drones
        self.grid_x    = grid_x
        self.grid_y    = grid_y
        self.cruise_z  = cruise_z
        self.n_ravens  = n_ravens
        self.max_iter  = max_iter
        self.pa_init   = pa_init
        self.pa_final  = pa_final

        if goal_positions is None:
            # Default goals: clustered top-right
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

        # Simulation parameters (matches FA planner)
        self._sensor_radius = 3.5
        self._interp_steps  = 8

    # ------------------------------------------------------------------
    # Parameter space conversion
    # ------------------------------------------------------------------
    def raven_to_waypoints(self, pop_element: np.ndarray, current_positions: np.ndarray) -> np.ndarray:
        """
        pop_element: shape (n_drones, n_remaining_wps - 2, 2)
        Converts lateral deviations and vertical jitters into 3D waypoints
        where waypoint 0 is current_positions and waypoint -1 is goal_positions.
        """
        n_wps = pop_element.shape[1] + 2
        wps = np.empty((self.n_drones, n_wps, 3))
        for d in range(self.n_drones):
            p_start = current_positions[d]
            p_goal  = self.goal_positions[d]
            wps[d, 0]  = p_start
            wps[d, -1] = p_goal
            
            V = p_goal - p_start
            D = np.linalg.norm(V[:2]) + 1e-9
            dir_vec = V[:2] / D
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])
            
            for w in range(1, n_wps - 1):
                t = w / (n_wps - 1)
                p_linear = p_start + t * V
                
                # lateral deviation (dynamic eye-shaped scaling to fit boundaries perfectly)
                lat_dev = pop_element[d, w - 1, 0] * (4.0 * t * (1.0 - t))
                # vertical jitter
                vert_jit = pop_element[d, w - 1, 1]
                
                p_wp = p_linear.copy()
                p_wp[:2] += perp_vec * lat_dev
                p_wp[2]  += vert_jit
                
                # clip to valid boundaries
                p_wp[0] = np.clip(p_wp[0], 1, self.grid_x - 1)
                p_wp[1] = np.clip(p_wp[1], 1, self.grid_y - 1)
                p_wp[2] = np.clip(p_wp[2], self.cruise_z - 0.5, self.cruise_z + 0.5)
                
                wps[d, w] = p_wp
        return wps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _awareness_prob(self, iteration: int) -> float:
        return self.pa_init - (self.pa_init - self.pa_final) * iteration / self.max_iter

    def _simulate_new_coverage(
        self,
        wps_3d:        np.ndarray,
        coverage_grid: np.ndarray,
    ) -> tuple:
        """
        Simulate coverage on 3D waypoints. Only new cells count.
        Tracks the overlap ratio among drones for the newly planned path segments.
        """
        new_covered    = np.zeros((self.grid_y, self.grid_x), dtype=bool)
        total_path_len = 0.0
        total_indiv_covered = 0

        r  = int(math.ceil(self._sensor_radius))
        ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
        disk   = (xs ** 2 + ys ** 2) <= self._sensor_radius ** 2

        for d in range(self.n_drones):
            drone_cov = np.zeros((self.grid_y, self.grid_x), dtype=bool)
            waypoints    = wps_3d[d]
            n_wp         = len(waypoints)

            for w in range(n_wp - 1):
                p_start  = waypoints[w]
                p_end    = waypoints[w + 1]
                seg_len  = float(np.linalg.norm(p_end - p_start))
                total_path_len += seg_len

                for step in range(self._interp_steps + 1):
                    t   = step / self._interp_steps
                    pos = p_start + t * (p_end - p_start)

                    cx = int(round(float(pos[0])))
                    cy = int(round(float(pos[1])))

                    x_lo = max(0, cx - r);  x_hi = min(self.grid_x, cx + r + 1)
                    y_lo = max(0, cy - r);  y_hi = min(self.grid_y, cy + r + 1)

                    dx_lo = x_lo - (cx - r)
                    dx_hi = dx_lo + (x_hi - x_lo)
                    dy_lo = y_lo - (cy - r)
                    dy_hi = dy_lo + (y_hi - y_lo)

                    drone_cov[y_lo:y_hi, x_lo:x_hi] |= \
                        disk[dy_lo:dy_hi, dx_lo:dx_hi]

            total_indiv_covered += int(drone_cov.sum())
            new_covered |= drone_cov

        new_cells = int(np.sum(new_covered & ~coverage_grid))

        # Overlap ratio for planned paths
        union_covered = int(new_covered.sum())
        overlap_ratio = 0.0
        if union_covered > 0:
            overlap_ratio = (total_indiv_covered - union_covered) / union_covered

        return new_cells, total_path_len, overlap_ratio

    def _fitness(
        self,
        pop_element:       np.ndarray,
        current_positions: np.ndarray,
        coverage_grid:     np.ndarray,
    ) -> float:
        wps_3d = self.raven_to_waypoints(pop_element, current_positions)
        new_cells, path_len, overlap = self._simulate_new_coverage(wps_3d, coverage_grid)
        # Multiply overlap by 2500.0 to match the scale of cell count reward (grid max cells = 10000)
        return float(new_cells) - 2500.0 * overlap - 0.05 * path_len

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------
    def _init_population(
        self,
        current_positions: np.ndarray,
        coverage_grid:     np.ndarray,
        n_remaining_wps:   int,
    ) -> list:
        """
        Build initial population of ravens in parameter space.
        Each raven shape = (n_drones, n_remaining_wps - 2, 2)
        - Index 0: lateral deviation (uniform [-35.0, 35.0])
        - Index 1: vertical jitter (uniform [-0.4, 0.4])
        """
        population = []
        for _ in range(self.n_ravens):
            pop_element = np.empty((self.n_drones, n_remaining_wps - 2, 2))
            
            for d in range(self.n_drones):
                # Corridor bias spreads from negative to positive [-40, 40] to fit cleanly within boundaries
                corridor_center = -40.0 + (80.0 * d / max(1, self.n_drones - 1))
                pop_element[d, :, 0] = corridor_center + np.random.uniform(-3.0, 3.0, n_remaining_wps - 2)
                pop_element[d, :, 1] = np.random.uniform(-0.4, 0.4, n_remaining_wps - 2)
                
            population.append(pop_element)
        return population

    # ------------------------------------------------------------------
    # Main replanning method
    # ------------------------------------------------------------------
    def replan(
        self,
        current_positions:   np.ndarray,
        coverage_grid:       np.ndarray,
        n_remaining_waypoints: int = 8,
    ) -> np.ndarray:
        """
        Replan remaining intermediate deviations using RRA.
        Returns final waypoints: (n_drones, n_remaining_waypoints, 3)
        """
        if n_remaining_waypoints <= 2:
            # No intermediate waypoints to optimize, return direct start-to-goal line
            dummy_element = np.zeros((self.n_drones, max(1, n_remaining_waypoints - 2), 2))
            return self.raven_to_waypoints(dummy_element, current_positions)[:, :n_remaining_waypoints, :]

        # Initialise population in parameter space
        population = self._init_population(
            current_positions, coverage_grid, n_remaining_waypoints
        )
        n_rp = self.n_ravens

        # Evaluate initial fitness
        fitness = np.array([self._fitness(rv, current_positions, coverage_grid) for rv in population])

        # Personal bests
        personal_best         = [rv.copy() for rv in population]
        personal_best_fitness = fitness.copy()

        # Global best
        gb_idx    = int(np.argmax(fitness))
        glob_best = population[gb_idx].copy()
        glob_fit  = fitness[gb_idx]

        # Raven Roosting iterations
        for it in range(self.max_iter):
            pa = self._awareness_prob(it)

            # Top-30% indices by fitness (foraging behavior)
            top_k      = max(1, int(0.3 * n_rp))
            top_indices = np.argsort(fitness)[-top_k:]

            for i in range(n_rp):
                raven_i = population[i].copy()
                r_rand  = np.random.rand()

                if r_rand < pa:
                    # Roosting: move toward global best
                    diff       = glob_best - raven_i
                    raven_i   += np.random.rand() * 0.6 * diff
                else:
                    # Foraging: move toward a top-30% successful raven
                    chosen_idx = int(np.random.choice(top_indices))
                    chosen_rv  = population[chosen_idx]
                    diff       = chosen_rv - raven_i
                    noise      = np.random.randn(*raven_i.shape) * 0.15
                    noise[..., 1] *= 0.15  # damp vertical search noise
                    raven_i   += np.random.rand() * 0.8 * diff + noise

                # Clip deviations (±50 to prevent squashing/bunching at boundaries)
                raven_i[..., 0] = np.clip(raven_i[..., 0], -50.0, 50.0)  # Max lateral deviation
                raven_i[..., 1] = np.clip(raven_i[..., 1], -0.5, 0.5)    # Vertical jitter

                population[i] = raven_i

                # Evaluate
                f_new = self._fitness(raven_i, current_positions, coverage_grid)
                fitness[i] = f_new

                # Update personal best
                if f_new > personal_best_fitness[i]:
                    personal_best_fitness[i] = f_new
                    personal_best[i]         = raven_i.copy()

                # Update global best
                if f_new > glob_fit:
                    glob_fit  = f_new
                    glob_best = raven_i.copy()

        # Return best converted 3D waypoints
        return self.raven_to_waypoints(glob_best, current_positions)

# ---------------------------------------------------------------------------
# Main entry point – quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Raven Roosting Replanner – Self-Test ===\n")
    N_DRONES = 6
    GRID_X   = 100
    GRID_Y   = 100

    coverage_grid = np.random.rand(GRID_Y, GRID_X) < 0.4
    current_positions = np.column_stack([
        np.random.uniform(10, 40, N_DRONES),
        np.random.uniform(10, 40, N_DRONES),
        np.full(N_DRONES, 6.0),
    ])

    replanner = RavenReplanner(n_drones=N_DRONES, grid_x=GRID_X, grid_y=GRID_Y)
    updated_waypoints = replanner.replan(
        current_positions=current_positions,
        coverage_grid=coverage_grid,
        n_remaining_waypoints=8,
    )

    print(f"Replanned waypoints shape: {updated_waypoints.shape}")
    print(f"  Expected            : ({N_DRONES}, 8, 3)")
    print(f"\nDrone 0 waypoints:\n{updated_waypoints[0]}")
    print("\n[RA] Replanning test completed successfully.")
