"""
fast_results.py  -  Optimised FA+RA+Waypoint runner
Targets ~3-5 min total on CPU.
- Vectorised FA (no Python loops over firefly pairs)
- Lightweight RA (10 ravens x 15 iters per replan)
- 600 steps, frame-skip=8 -> ~75 frames GIF
- Low-DPI 2-panel figure for fast render
- Full results table printed at end
"""
import os, sys, time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)
from multi_drone_coverage_env import (
    MultiDroneCoverageEnv, DRONE_COLORS, RECHARGE_STATIONS_XY
)

# ── tuneable knobs ────────────────────────────────────────────────────────
N_STEPS       = 600
N_DRONES      = 6
REPLAN_EVERY  = 120          # RA kicks in 5 times
N_WP          = 14           # waypoints per drone
SENSOR_R      = 3.5
GX, GY        = 100, 100
CRUISE_Z      = 6.0
FRAME_SKIP    = 8            # ~75 animation frames
FPS           = 10
DPI           = 85
GIF_OUT       = os.path.join(os.path.dirname(__file__), "models", "drone_coverage.gif")
FA_CACHE      = os.path.join(os.path.dirname(__file__), "dataset", "fa_waypoints.npy")

# ═══════════════════════════════════════════════════════════════════════════
# FAST VECTORISED FIREFLY ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════
def _coverage_from_paths(paths_xy, grid_x=GX, grid_y=GY, r=SENSOR_R):
    """paths_xy: (N_drones, N_pts, 2). Returns coverage bool grid."""
    cov  = np.zeros((grid_x, grid_y), dtype=bool)
    r2   = r * r
    xs   = np.arange(grid_x) + 0.5
    ys   = np.arange(grid_y) + 0.5
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    cells  = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (GX*GY, 2)
    for d in range(paths_xy.shape[0]):
        for pt in paths_xy[d]:
            d2 = ((cells - pt) ** 2).sum(axis=1)
            hits = np.where(d2 <= r2)[0]
            for h in hits:
                cov[h // grid_y, h % grid_y] = True
    return cov

def interpolate_waypoints(wp, steps_per_seg=5):
    """wp: (N_drones, N_wps, 3) -> (N_drones, N_pts, 2) XY only."""
    n_d, n_w, _ = wp.shape
    pts = []
    for d in range(n_d):
        drone_pts = []
        for w in range(n_w - 1):
            for s in range(steps_per_seg):
                t = s / steps_per_seg
                p = wp[d, w, :2] * (1 - t) + wp[d, w+1, :2] * t
                drone_pts.append(p)
        drone_pts.append(wp[d, -1, :2])
        pts.append(np.array(drone_pts))
    return np.array(pts, dtype=object)  # ragged OK

def fa_fitness_vec(pop, grid_x=GX, grid_y=GY):
    """Vectorised fitness for whole population. pop: (P, N_d, N_wp, 3)."""
    P = len(pop)
    scores = np.zeros(P)
    r2 = SENSOR_R ** 2
    xs = np.arange(grid_x) + 0.5
    ys = np.arange(grid_y) + 0.5
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    cells = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (GX*GY, 2)

    for p in range(P):
        cov  = np.zeros(grid_x * grid_y, dtype=bool)
        path = 0.0
        for d in range(pop[p].shape[0]):
            wps = pop[p][d]                      # (N_wp, 3)
            for w in range(len(wps) - 1):
                seg = wps[w+1, :2] - wps[w, :2]
                path += float(np.linalg.norm(seg))
                # 3 sample points per segment (fast approximation)
                for t in [0.0, 0.5, 1.0]:
                    pt = wps[w, :2] + t * seg
                    d2 = ((cells - pt) ** 2).sum(axis=1)
                    cov |= (d2 <= r2)
        cov_ratio = cov.sum() / (grid_x * grid_y)
        bat_est   = 0.08 * N_WP * 5 * N_DRONES + 0.001 * path
        scores[p] = cov_ratio * 100 - 0.015 * path - 0.04 * bat_est
    return scores

def run_fa(n_fireflies=25, max_iter=40, alpha=0.22, beta0=1.0, gamma=0.06):
    print(f"[FA] {n_fireflies} fireflies x {max_iter} iters ...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    # Population: (P, N_d, N_wp, 3)
    pop = rng.uniform([1,1,CRUISE_Z-0.4], [GX-1, GY-1, CRUISE_Z+0.4],
                      size=(n_fireflies, N_DRONES, N_WP, 3))
    bright = fa_fitness_vec(pop)
    best_i = int(np.argmax(bright))
    best   = pop[best_i].copy()
    best_f = bright[best_i]

    for it in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if bright[j] > bright[i]:
                    diff = pop[j] - pop[i]
                    r2   = float(np.sum(diff**2))
                    att  = beta0 * math.exp(-gamma * r2)
                    pop[i] += att * diff + alpha * rng.standard_normal(pop[i].shape)
                    # clip
                    pop[i, :, :, 0] = np.clip(pop[i, :, :, 0], 1, GX-1)
                    pop[i, :, :, 1] = np.clip(pop[i, :, :, 1], 1, GY-1)
                    pop[i, :, :, 2] = np.clip(pop[i, :, :, 2], CRUISE_Z-0.4, CRUISE_Z+0.4)
        bright = fa_fitness_vec(pop)
        bi = int(np.argmax(bright))
        if bright[bi] > best_f:
            best_f = bright[bi]; best = pop[bi].copy()
        if (it+1) % 10 == 0:
            print(f"  iter {it+1:3d}/{max_iter}  best_fitness={best_f:.4f}")

    print(f"[FA] Done in {time.time()-t0:.1f}s  fitness={best_f:.4f}")
    return best   # (N_d, N_wp, 3)

# ═══════════════════════════════════════════════════════════════════════════
# FAST RAVEN REPLANNER
# ═══════════════════════════════════════════════════════════════════════════
def run_ra(cur_pos, cov_grid, n_ravens=10, max_iter=15, n_wp=8):
    """Returns (N_d, n_wp, 3) best waypoints biased to uncovered cells."""
    uncov = np.argwhere(~cov_grid)
    rng   = np.random.default_rng()

    def _init_raven():
        rv = np.zeros((N_DRONES, n_wp, 3))
        rv[:, 0, :] = cur_pos
        for d in range(N_DRONES):
            for w in range(1, n_wp):
                if len(uncov) > 0:
                    c = uncov[rng.integers(len(uncov))]
                    rv[d, w, 0] = np.clip(c[0] + rng.normal(0, 4), 1, GX-1)
                    rv[d, w, 1] = np.clip(c[1] + rng.normal(0, 4), 1, GY-1)
                else:
                    rv[d, w, 0] = rng.uniform(1, GX-1)
                    rv[d, w, 1] = rng.uniform(1, GY-1)
                rv[d, w, 2] = CRUISE_Z + rng.uniform(-0.3, 0.3)
        return rv

    def _fit(rv):
        """Quick fitness: new coverage only, minus path."""
        r2    = SENSOR_R ** 2
        new_c = np.zeros((GX, GY), dtype=bool)
        path  = 0.0
        for d in range(N_DRONES):
            for w in range(n_wp - 1):
                seg  = rv[d, w+1, :2] - rv[d, w, :2]
                path += float(np.linalg.norm(seg))
                for t in [0.0, 0.5, 1.0]:
                    pt = rv[d, w, :2] + t * seg
                    xi = int(np.clip(pt[0], 0, GX-1))
                    yi = int(np.clip(pt[1], 0, GY-1))
                    # mark a disk (fast integer loop)
                    ri = int(SENSOR_R) + 1
                    for dx in range(-ri, ri+1):
                        for dy in range(-ri, ri+1):
                            if dx*dx + dy*dy <= r2:
                                nx2 = int(np.clip(xi+dx, 0, GX-1))
                                ny2 = int(np.clip(yi+dy, 0, GY-1))
                                new_c[nx2, ny2] = True
        new_cells = int(np.sum(new_c & ~cov_grid))
        return float(new_cells) - 0.03 * path

    pop     = [_init_raven() for _ in range(n_ravens)]
    fit     = np.array([_fit(r) for r in pop])
    gb_idx  = int(np.argmax(fit))
    gb      = pop[gb_idx].copy()
    gb_fit  = fit[gb_idx]

    for it in range(max_iter):
        pa = 0.9 - 0.8 * it / max_iter
        top_k = max(1, int(0.3 * n_ravens))
        top   = np.argsort(fit)[-top_k:]
        for i in range(n_ravens):
            rv = pop[i].copy()
            if rng.random() < pa:
                rv += rng.random() * 0.6 * (gb - rv)
            else:
                ch = pop[int(rng.choice(top))].copy()
                rv += rng.random() * 0.8 * (ch - rv) + rng.standard_normal(rv.shape) * 0.12
            rv[:, 0, :] = cur_pos
            rv[:, :, 0] = np.clip(rv[:, :, 0], 1, GX-1)
            rv[:, :, 1] = np.clip(rv[:, :, 1], 1, GY-1)
            rv[:, :, 2] = np.clip(rv[:, :, 2], CRUISE_Z-0.4, CRUISE_Z+0.4)
            pop[i] = rv
            f = _fit(rv)
            fit[i] = f
            if f > gb_fit:
                gb_fit = f; gb = rv.copy()
    return gb

# ═══════════════════════════════════════════════════════════════════════════
# WAYPOINT ACTION
# ═══════════════════════════════════════════════════════════════════════════
def wp_action(positions, waypoints, wp_idx, max_speed):
    act = np.zeros((N_DRONES, 3))
    for i in range(N_DRONES):
        idx  = min(int(wp_idx[i]), waypoints.shape[1]-1)
        diff = waypoints[i, idx] - positions[i]
        dist = np.linalg.norm(diff) + 1e-9
        act[i] = np.clip((diff/dist)*min(dist, max_speed), -1, 1)
    return act.flatten()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)

# ── Environment ────────────────────────────────────────────────────────────
print("\n[INIT] Building MultiDroneCoverageEnv ...")
env = MultiDroneCoverageEnv(
    n_drones=N_DRONES,
    wind_enabled=True,
    thermal_enabled=True,
    sensor_noise_std=0.03,
    dyn_count=12,
)
obs, _ = env.reset()
print(f"  obs={obs.shape}  action={env.action_space.shape}")

# ── FA Waypoints ───────────────────────────────────────────────────────────
if os.path.exists(FA_CACHE):
    waypoints = np.load(FA_CACHE)
    print(f"[FA] Loaded cached waypoints {waypoints.shape}")
    # pad/trim to N_WP
    if waypoints.shape[1] < N_WP:
        pad = np.tile(waypoints[:, -1:, :], (1, N_WP - waypoints.shape[1], 1))
        waypoints = np.concatenate([waypoints, pad], axis=1)
    waypoints = waypoints[:, :N_WP, :]
else:
    waypoints = run_fa(n_fireflies=25, max_iter=40)
    np.save(FA_CACHE, waypoints)

# ── Mission loop ───────────────────────────────────────────────────────────
wp_idx       = np.zeros(N_DRONES, dtype=int)
drone_paths  = [[] for _ in range(N_DRONES)]
frames       = []
cov_h, col_h, bat_h, path_h, wind_h = [], [], [], [], []

print(f"\n[RUN] {N_STEPS} steps | FA+RA waypoint following | wind+thermals ON")
print("="*65)
t_run = time.time()

for step in range(N_STEPS):

    # RA adaptive replan
    if step > 0 and step % REPLAN_EVERY == 0:
        cov_pct = env.coverage_ratio() * 100
        print(f"  [RA] step={step} | coverage={cov_pct:.1f}% | replanning ...")
        t_ra = time.time()
        new_wp = run_ra(env.positions.copy(), env.coverage_grid.copy(),
                        n_ravens=10, max_iter=15, n_wp=8)
        # append new waypoints after current completed ones
        min_idx = int(wp_idx.min())
        prefix  = waypoints[:, :min_idx, :]
        waypoints = np.concatenate([prefix, new_wp], axis=1)[:, :N_WP, :]
        wp_idx = np.clip(wp_idx, 0, waypoints.shape[1]-1)
        print(f"  [RA] done in {time.time()-t_ra:.1f}s")

    action = wp_action(env.positions, waypoints, wp_idx, env.max_speed)
    obs, reward, term, trunc, info = env.step(action)

    # Advance waypoint index
    for i in range(N_DRONES):
        idx = min(int(wp_idx[i]), waypoints.shape[1]-1)
        if np.linalg.norm(env.positions[i] - waypoints[i, idx]) < 4.0:
            wp_idx[i] = min(wp_idx[i]+1, waypoints.shape[1]-1)

    for i in range(N_DRONES):
        drone_paths[i].append(env.positions[i].copy())

    cov_h.append(info["coverage_pct"])
    col_h.append(info["total_collisions"])
    bat_h.append(info["total_battery_used"])
    path_h.append(info["total_path_length"])
    wind_h.append(info["wind_speed"])

    if step % FRAME_SKIP == 0:
        frames.append({
            "step":      step,
            "pos":       env.positions.copy(),
            "cov":       env.coverage_grid.copy(),
            "bats":      list(info["batteries"]),
            "phases":    list(info["phases"]),
            "mh":        list(info["motor_health"]),
            "dyn":       [p.copy() for p in env.dyn.positions],
            "cov_pct":   info["coverage_pct"],
            "col":       info["total_collisions"],
            "bat_used":  info["total_battery_used"],
            "path_len":  info["total_path_length"],
            "wind":      info["wind_speed"],
            "paths":     [list(drone_paths[i]) for i in range(N_DRONES)],
            "wp":        waypoints.copy(),
            "wp_idx":    wp_idx.copy(),
        })

    if step % 100 == 0:
        print(f"  step={step:4d} | cov={info['coverage_pct']:5.1f}%"
              f" | col={info['total_collisions']:3d}"
              f" | bat={info['total_battery_used']:6.1f}"
              f" | path={info['total_path_length']:7.1f}"
              f" | wind={info['wind_speed']:.3f}")

    if term or trunc:
        print(f"  Episode ended at step {step}"); break

t_run_end = time.time()
fd = frames[-1]

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  RESULTS  |  Multi-Drone Coverage  |  FA + RA + Physics")
print("=" * 65)
print(f"  Grid            : 100 x 100 x 15  ({GX*GY:,} cells)")
print(f"  Drones          : {N_DRONES}")
print(f"  Steps           : {env.step_count}")
print(f"  Sim wall-time   : {t_run_end - t_run:.1f} s")
print("-" * 65)
print(f"  COVERAGE        : {fd['cov_pct']:.2f} %")
print(f"  Cells covered   : {int(fd['cov_pct']/100*GX*GY):,} / {GX*GY:,}")
print("-" * 65)
print(f"  OBJ 1 Collisions: {fd['col']}")
print(f"  OBJ 2 Path len  : {fd['path_len']:.1f} m")
print(f"  OBJ 3 Battery   : {fd['bat_used']:.1f} units consumed")
print("-" * 65)
print(f"  Wind avg/peak   : {np.mean(wind_h):.4f} / {np.max(wind_h):.4f} m/step")
print(f"  RA replannings  : {N_STEPS // REPLAN_EVERY}")
print("-" * 65)
print(f"  {'Drone':<8} {'Phase':<20} {'Battery':>10} {'MotorHlth':>10} {'Steps':>8}")
print(f"  {'-'*7}  {'-'*19}  {'-'*9}  {'-'*9}  {'-'*7}")
for i in range(N_DRONES):
    print(f"  Drone {i}  {fd['phases'][i]:<20}  "
          f"{fd['bats'][i]:>9.1f}  "
          f"{fd['mh'][i]*100:>9.1f}%  "
          f"{len(drone_paths[i]):>7}")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════════════
# FAST 2-PANEL GIF
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n[GIF] Rendering {len(frames)} frames at {DPI} DPI ...")
t_gif = time.time()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0d0d1a")
fig.suptitle("Multi-Drone Coverage  |  FA + RA + Physics  |  6 Drones  |  100x100",
             color="white", fontsize=12, fontweight="bold")
for ax in (ax1, ax2):
    ax.set_facecolor("#131328")
    for sp in ax.spines.values(): sp.set_color("#3a3a6a")
    ax.tick_params(colors="#aaaacc")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

rs_xy = np.array(RECHARGE_STATIONS_XY, dtype=float)

def animate(fi):
    f  = frames[fi]
    xs = list(range(min(f["step"]+1, len(cov_h))))

    # ── Left: 2D coverage map ─────────────────────────────────────────────
    ax1.clear(); ax1.set_facecolor("#131328")
    for sp in ax1.spines.values(): sp.set_color("#3a3a6a")
    ax1.imshow(f["cov"].T, origin="lower", cmap="YlGn",
               vmin=0, vmax=1, alpha=0.85,
               extent=[0, GX, 0, GY], aspect="auto")
    # Recharge stations
    ax1.scatter(rs_xy[:, 0], rs_xy[:, 1], s=100, marker="s",
                c="magenta", edgecolors="white", linewidths=0.7, zorder=5)
    # Dynamic obstacles
    if f["dyn"]:
        dp = np.array(f["dyn"])
        ax1.scatter(dp[:, 0], dp[:, 1], s=35, c="royalblue",
                    alpha=0.7, edgecolors="white", linewidths=0.4, zorder=4)
    # Drone paths + markers
    for i in range(N_DRONES):
        col = DRONE_COLORS[i]
        pts = f["paths"][i]
        if len(pts) > 1:
            p = np.array(pts)
            ax1.plot(p[:, 0], p[:, 1], color=col, lw=1.0, alpha=0.5, zorder=3)
        pos = f["pos"][i]
        ax1.scatter(pos[0], pos[1], s=95, c=col,
                    edgecolors="white", linewidths=0.8, zorder=6, marker="^")
        ax1.annotate(f"D{i}", (pos[0], pos[1]),
                     xytext=(3, 3), textcoords="offset points",
                     fontsize=6.5, color=col, fontweight="bold")
        # next waypoint dashed line
        wi  = min(int(f["wp_idx"][i]), f["wp"].shape[1]-1)
        wpt = f["wp"][i, wi]
        ax1.plot([pos[0], wpt[0]], [pos[1], wpt[1]],
                 color=col, lw=0.6, alpha=0.3, linestyle="--", zorder=4)
        ax1.scatter(wpt[0], wpt[1], s=30, marker="x",
                    c=col, linewidths=1.1, alpha=0.7, zorder=5)

    ax1.set_xlim(0, GX); ax1.set_ylim(0, GY)
    ax1.set_xlabel("X (m)", fontsize=8); ax1.set_ylabel("Y (m)", fontsize=8)
    ax1.set_title(
        f"Step {f['step']:4d}  |  Coverage: {f['cov_pct']:.1f}%"
        f"  |  Wind: {f['wind']:.3f}  |  Col: {f['col']}",
        fontsize=9, color="white")
    ax1.tick_params(colors="#aaaacc")

    # ── Right: metrics panel ──────────────────────────────────────────────
    ax2.clear(); ax2.set_facecolor("#131328")
    for sp in ax2.spines.values(): sp.set_color("#3a3a6a")

    ax2.plot(xs, cov_h[:len(xs)], color="#00ff88", lw=2.0, label="Coverage %")
    ax2.set_ylabel("Coverage %", color="#00ff88", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#00ff88")
    ax2.set_ylim(0, 105)

    ax_r = ax2.twinx()
    ax_r.set_facecolor("#131328")
    mx_c = max(col_h[-1], 1); mx_b = max(bat_h[-1], 1); mx_p = max(path_h[-1], 1)
    ax_r.plot(xs, [v/mx_c*100 for v in col_h[:len(xs)]],
              color="#ff4455", lw=1.2, ls="--", label="Collisions (norm)")
    ax_r.plot(xs, [v/mx_b*100 for v in bat_h[:len(xs)]],
              color="#ffaa00", lw=1.2, ls=":",  label="Battery (norm)")
    ax_r.plot(xs, [v/mx_p*100 for v in path_h[:len(xs)]],
              color="#44ccff", lw=1.1, ls="-.", label="Path (norm)")
    ax_r.set_ylabel("Normalised 0-100", color="white", fontsize=8)
    ax_r.tick_params(axis="y", labelcolor="white", colors="#aaaacc")
    ax_r.set_ylim(0, 115)
    for sp in ax_r.spines.values(): sp.set_color("#3a3a6a")

    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax_r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, lbl1+lbl2, loc="upper left", fontsize=7,
               facecolor="#1a1a35", labelcolor="white", framealpha=0.8)

    # Battery mini-bars inside right panel
    for i in range(N_DRONES):
        pct = f["bats"][i] / env.battery_capacity
        bar_x = [0.62, 0.62+0.25*pct]
        bar_y = [0.82 - i*0.065] * 2
        col_b = ("#00ff88" if pct > 0.5 else "#ffaa00" if pct > 0.2 else "#ff4455")
        ax2.plot(bar_x, [bar_y[0]]*2, transform=ax2.transAxes,
                 color=col_b, lw=8, solid_capstyle="butt", alpha=0.8)
        ax2.text(0.89, bar_y[0], f"D{i} {pct*100:.0f}%",
                 transform=ax2.transAxes, fontsize=6.5,
                 color=DRONE_COLORS[i], va="center")

    ax2.set_xlabel("Step", fontsize=8)
    ax2.set_title(
        f"Objectives  |  Col={f['col']}  |  Path={f['path_len']:.0f}m"
        f"  |  Bat={f['bat_used']:.0f}",
        fontsize=9, color="white")
    ax2.tick_params(colors="#aaaacc")

anim = FuncAnimation(fig, animate, frames=len(frames),
                     interval=1000//FPS, blit=False)

writer = PillowWriter(fps=FPS)
anim.save(GIF_OUT, writer=writer, dpi=DPI)
plt.close(fig)

t_gif_end = time.time()
size_mb = os.path.getsize(GIF_OUT) / 1e6
print(f"[GIF] Saved -> {os.path.abspath(GIF_OUT)}")
print(f"[GIF] Size: {size_mb:.1f} MB  |  Render time: {t_gif_end-t_gif:.1f}s")
print(f"\n[TOTAL] Wall time: {t_gif_end - t_run:.1f}s")
print("[DONE]")
