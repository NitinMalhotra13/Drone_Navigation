"""
dynamic_obstacles.py  —  UPGRADED
Supports full 3D movement (not just above-terrain) for richer avoidance challenge.
move_mode options:
  "above_terrain"  – classic ground-skimming mode
  "cruise"         – mid-altitude roaming (5-9 units)
  "full_3d"        – random 3D walk throughout the entire grid volume
"""

import numpy as np


class DynamicObstacle:
    def __init__(self, grid_size=(100, 100, 15), count=12, move_mode="full_3d"):
        self.grid = np.array(grid_size, int)
        self.count = int(count)
        self.move_mode = move_mode
        self.positions = []
        self.velocities = []       # NEW: persistent 3D velocities for smoother motion
        self.initialized = False

    # ------------------------------------------------------------------
    def _blocked(self, static, x, y, z):
        try:
            return bool(static[int(x), int(y), int(z)])
        except Exception:
            return False

    # ------------------------------------------------------------------
    def initialize(self, static=None, terrain=None):
        self.positions = []
        self.velocities = []
        X, Y, Z = self.grid

        for _ in range(self.count):
            placed = False
            for _ in range(500):
                x = np.random.randint(0, X)
                y = np.random.randint(0, Y)

                if self.move_mode == "above_terrain":
                    h = int(terrain[x, y]) if terrain is not None else 0
                    z = h + np.random.uniform(1.2, 2.5)
                elif self.move_mode == "cruise":
                    z = np.random.uniform(5.0, min(9.0, Z - 1))
                else:  # full_3d  — spread through the whole volume
                    h = int(terrain[x, y]) if terrain is not None else 0
                    z = np.random.uniform(h + 1.0, Z - 1.0)

                z = float(np.clip(z, 0.5, Z - 0.5))
                zi = int(z)

                if static is not None and self._blocked(static, x, y, zi):
                    continue

                p = np.array([float(x) + 0.5, float(y) + 0.5, z])
                # Initial random velocity — capped at 0.4 per axis
                v = np.random.uniform(-0.35, 0.35, size=3)
                v[2] *= 0.6  # slightly dampen vertical initially
                self.positions.append(p)
                self.velocities.append(v)
                placed = True
                break

            if not placed:
                p = np.array([float(X) / 2, float(Y) / 2, float(Z) / 2])
                self.positions.append(p)
                self.velocities.append(np.zeros(3))

        self.initialized = True

    # ------------------------------------------------------------------
    def move(self, static=None, terrain=None):
        if not self.initialized:
            self.initialize(static=static, terrain=terrain)
            return

        X, Y, Z = self.grid
        updated_pos = []
        updated_vel = []

        for i, (pos, vel) in enumerate(zip(self.positions, self.velocities)):
            # Perturb velocity (random walk in velocity space)
            dv = np.random.uniform(-0.12, 0.12, size=3)
            vel = vel + dv
            # Speed cap
            speed = np.linalg.norm(vel)
            if speed > 0.45:
                vel = vel / speed * 0.45

            new_pos = pos + vel

            # Boundary reflection
            for axis, limit in enumerate([X, Y, Z]):
                if new_pos[axis] < 0.5:
                    new_pos[axis] = 0.5
                    vel[axis] = abs(vel[axis])  # reflect
                elif new_pos[axis] > limit - 0.5:
                    new_pos[axis] = limit - 0.5
                    vel[axis] = -abs(vel[axis])

            # Terrain floor: stay at least 1 m above ground
            if terrain is not None:
                xi = int(np.clip(new_pos[0], 0, X - 1))
                yi = int(np.clip(new_pos[1], 0, Y - 1))
                ground = float(terrain[xi, yi])
                if new_pos[2] < ground + 1.0:
                    new_pos[2] = ground + 1.0
                    vel[2] = abs(vel[2])  # bounce up

            # Static obstacle avoidance: if next cell blocked, reflect
            zi = int(np.clip(new_pos[2], 0, Z - 1))
            if static is not None and self._blocked(static, int(new_pos[0]), int(new_pos[1]), zi):
                vel = -vel * 0.5
                new_pos = pos.copy()

            updated_pos.append(new_pos)
            updated_vel.append(vel)

        self.positions = updated_pos
        self.velocities = updated_vel
