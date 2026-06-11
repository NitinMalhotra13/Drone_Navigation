import os
import numpy as np
from noise import pnoise2
from scipy.ndimage import gaussian_filter

# -- UPGRADED: 100x100x15 grid ----------------------------------------------
GRID_X = 100
GRID_Y = 100
MAX_HEIGHT = 4.0
PERLIN_SCALE = 0.08          # lower = smoother, larger features
SMOOTH_SIGMA = 1.8
SEED = 1234
np.random.seed(SEED)


def generate_perlin_terrain(grid_x=GRID_X, grid_y=GRID_Y):
    terrain = np.zeros((grid_x, grid_y), dtype=float)
    for x in range(grid_x):
        for y in range(grid_y):
            nx = x * PERLIN_SCALE
            ny = y * PERLIN_SCALE
            n = pnoise2(
                nx, ny,
                octaves=4,
                persistence=0.50,
                lacunarity=2.0,
                repeatx=2048,
                repeaty=2048,
                base=SEED,
            )
            terrain[x, y] = n
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-9)
    terrain *= MAX_HEIGHT
    terrain = gaussian_filter(terrain, sigma=SMOOTH_SIGMA)
    terrain = np.clip(terrain, 0.0, MAX_HEIGHT)
    return terrain


def save_terrain():
    base = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(base, exist_ok=True)
    T = generate_perlin_terrain()
    np.save(os.path.join(base, "terrain.npy"), T)
    print(f"[OK] Terrain generated ({GRID_X}x{GRID_Y}) -> dataset/terrain.npy")
    print(f"  Height range: {float(T.min()):.3f}  ->  {float(T.max()):.3f}")


if __name__ == "__main__":
    save_terrain()
