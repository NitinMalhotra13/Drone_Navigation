# src/generate_static_obstacles.py
import os
import numpy as np

# -- UPGRADED: 100x100x15 grid ----------------------------------------------
GRID_X = 100
GRID_Y = 100
GRID_Z = 15

# Moderate densities for a clean, spacious obstacle field (no buildings)
TREE_DENSITY   = 0.02    # ~200 trees (was 0.085)
ROCK_COUNT     = 20      # boulder/rock clusters scattered across the map (was 75)
BUILDING_COUNT = 0       # no buildings generated (was 3)

TREE_MIN_HEIGHT = 5
TREE_MAX_HEIGHT = 10

ROCK_MIN_HEIGHT = 3      # shorter but very wide footprint
ROCK_MAX_HEIGHT = 6
ROCK_RADIUS     = 2      # cells around the centre that are also blocked

BLDG_MIN_HEIGHT = 6      # taller buildings
BLDG_MAX_HEIGHT = 12
BLDG_HALF_W    = 2       # half-width in cells

# Multi-drone start corners + recharge stations must stay clear
STARTS = [(2, 2), (2, 97), (97, 2), (97, 97), (2, 50), (97, 50)]
RECHARGE_STATIONS = [(20, 20), (50, 50), (80, 20), (20, 80), (80, 80)]

FORBIDDEN_RADIUS = 7   # slightly larger clear zone (was 5)


def is_forbidden(x, y):
    """Return True if cell (x,y) must remain obstacle-free."""
    for (sx, sy) in STARTS:
        if abs(x - sx) <= FORBIDDEN_RADIUS and abs(y - sy) <= FORBIDDEN_RADIUS:
            return True
    for (rx, ry) in RECHARGE_STATIONS:
        if abs(x - rx) <= FORBIDDEN_RADIUS and abs(y - ry) <= FORBIDDEN_RADIUS:
            return True
    return False


def generate_static_obstacles(grid_x=GRID_X, grid_y=GRID_Y, grid_z=GRID_Z):
    """
    Generate a rich 3D boolean obstacle array with three obstacle types:
      1. Dense forest trees  (~650 individual trunks)
      2. Rock / boulder clusters (55 clusters, radius up to 2 cells)
      3. Building clusters      (10 clusters, up to 2x2 cell footprint)
    All obstacle types respect the FORBIDDEN zones around drone starts and
    recharge stations so those areas remain navigable.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(base_dir, exist_ok=True)

    terrain_path = os.path.join(base_dir, "terrain.npy")
    if not os.path.exists(terrain_path):
        raise FileNotFoundError("Missing terrain.npy — run generate_terrain.py first.")

    terrain = np.load(terrain_path)
    tx, ty = terrain.shape
    static = np.zeros((tx, ty, grid_z), dtype=bool)

    # ------------------------------------------------------------------
    # 1. Forest trees — random per-cell with density filter
    # ------------------------------------------------------------------
    tree_count = 0
    for x in range(tx):
        for y in range(ty):
            if is_forbidden(x, y):
                continue
            if np.random.rand() > TREE_DENSITY:
                continue
            ground_height = int(np.floor(terrain[x, y]))
            if ground_height > 3:      # no trees on steep terrain
                continue
            th  = np.random.randint(TREE_MIN_HEIGHT, TREE_MAX_HEIGHT + 1)
            top = min(ground_height + th, grid_z - 1)
            static[x, y, ground_height:top] = True
            tree_count += 1
    print(f"[static] Trees placed: {tree_count}")

    # ------------------------------------------------------------------
    # 2. Rock / boulder clusters — wider footprint, shorter height
    # ------------------------------------------------------------------
    rock_placed = 0
    attempts    = 0
    while rock_placed < ROCK_COUNT and attempts < ROCK_COUNT * 20:
        attempts += 1
        cx = np.random.randint(ROCK_RADIUS + 1, tx - ROCK_RADIUS - 1)
        cy = np.random.randint(ROCK_RADIUS + 1, ty - ROCK_RADIUS - 1)
        if is_forbidden(cx, cy):
            continue
        ground_height = int(np.floor(terrain[cx, cy]))
        rh  = np.random.randint(ROCK_MIN_HEIGHT, ROCK_MAX_HEIGHT + 1)
        top = min(ground_height + rh, grid_z - 1)
        # Circular footprint within ROCK_RADIUS
        for dx in range(-ROCK_RADIUS, ROCK_RADIUS + 1):
            for dy in range(-ROCK_RADIUS, ROCK_RADIUS + 1):
                if dx*dx + dy*dy <= ROCK_RADIUS*ROCK_RADIUS:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < tx and 0 <= ny < ty and not is_forbidden(nx, ny):
                        static[nx, ny, ground_height:top] = True
        rock_placed += 1
    print(f"[static] Rock clusters placed: {rock_placed}")

    # ------------------------------------------------------------------
    # 3. Building clusters — tall, rectangular footprint
    # ------------------------------------------------------------------
    bldg_placed = 0
    attempts    = 0
    while bldg_placed < BUILDING_COUNT and attempts < BUILDING_COUNT * 20:
        attempts += 1
        cx = np.random.randint(BLDG_HALF_W + 2, tx - BLDG_HALF_W - 2)
        cy = np.random.randint(BLDG_HALF_W + 2, ty - BLDG_HALF_W - 2)
        if is_forbidden(cx, cy):
            continue
        ground_height = int(np.floor(terrain[cx, cy]))
        bh  = np.random.randint(BLDG_MIN_HEIGHT, BLDG_MAX_HEIGHT + 1)
        top = min(ground_height + bh, grid_z - 1)
        blocked = False
        for dx in range(-BLDG_HALF_W, BLDG_HALF_W + 1):
            for dy in range(-BLDG_HALF_W, BLDG_HALF_W + 1):
                if is_forbidden(cx + dx, cy + dy):
                    blocked = True
        if blocked:
            continue
        for dx in range(-BLDG_HALF_W, BLDG_HALF_W + 1):
            for dy in range(-BLDG_HALF_W, BLDG_HALF_W + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < tx and 0 <= ny < ty:
                    static[nx, ny, ground_height:top] = True
        bldg_placed += 1
    print(f"[static] Building clusters placed: {bldg_placed}")

    out_path = os.path.join(base_dir, "static_obstacles.npy")
    np.save(out_path, static)
    total_blocked = int(static.any(axis=2).sum())
    print(f"[static] Saved obstacles ({tx}x{ty}x{grid_z}) -> {out_path}  "
          f"| {total_blocked} XY cells blocked ({100*total_blocked/(tx*ty):.1f}%)")
    return static


def preview(static):
    import matplotlib.pyplot as plt
    img = static.any(axis=2).T
    plt.figure(figsize=(9, 9))
    plt.imshow(img, cmap="Greens", origin="lower")
    plt.title("Obstacle Map (Trees + Rocks + Buildings)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    s = generate_static_obstacles()
    preview(s)
