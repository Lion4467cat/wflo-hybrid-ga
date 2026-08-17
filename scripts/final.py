"""
Hybrid Two-Fidelity GA — Wind Farm Layout Optimisation
=======================================================
GPU-vectorised Jensen wake model as fast surrogate screener,
with selective FLORIS validation for elite candidates only.

FIXES APPLIED (all rounds):
  Fix 1 — repair_individual: matrix.T.flatten() restores block layout
  Fix 2 — Jensen dim=1: sum over upstream i, not receiving j
  Fix 3 — Unit normalisation: both paths return GWh/yr
  Fix 4 — Call rate denominator: generations_completed not hardcoded 200
  Fix 5 — save_best_solutions removed: caused IndexError in newer PyGAD;
           best fitness tracked manually in best_fitness_history list instead
  Fix 6 — BEST_APPROX = 0.0 (was -1e9: caused full pop FLORIS flood in gen 1)
  Fix 7 — geometric_mutation: replace=False (was replacement — same turbine repeated)
  Fix 8 — generate_valid_position: raises on failure (was silent bad-coord return)
  Fix 9 — custom_crossover: splits on turbine index (was gene index — severed x/y blocks)
  Fix 10 — FLORIS hard cap: MAX_FLORIS_PER_GEN=5 stops the relative gate degrading
            to "evaluate everything" once the population matures. FLORIS_CALLS_THIS_GEN
            declared global at the TOP of fitness_func (not inside the if-block) so
            Python's static scope resolution honours it at the condition check.
"""

import numpy as np
import pygad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import torch
from floris import FlorisModel, WindRose
import warnings
import time
import logging
import os

logging.getLogger("floris").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ====================== CUDA SETUP ======================
USE_CUDA = True
if USE_CUDA and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA detected — {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("Running on CPU (no CUDA found)")

# ====================== PROBLEM SETUP ======================
NUM_TURBINES = 150
AREA_SIZE = 10000.0
MIN_DISTANCE = 400.0
CT = 0.8
ALPHA = 0.1
ROTOR_R = 63.0
HOURS_PER_YEAR = 8760.0

FARM_POLYGON = Polygon(
    [
        (1250, 500),
        (8750, 500),
        (9500, 2500),
        (9500, 5000),
        (7500, 7000),
        (5000, 8750),
        (2000, 8000),
        (750, 6250),
        (500, 2500),
        (1250, 500),
    ]
)
POLYGON_BOUNDS = FARM_POLYGON.bounds

# ====================== WIND ROSE ======================
WIND_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
WIND_SPEEDS_MS = np.array(
    [7.5, 8.5, 10.0, 11.5, 12.5, 11.0, 9.0, 8.0], dtype=np.float32
)
WIND_FREQS = np.array(
    [0.08, 0.12, 0.15, 0.18, 0.22, 0.13, 0.07, 0.05], dtype=np.float32
)
WIND_FREQS = WIND_FREQS / WIND_FREQS.sum()

WAKE_CONE_DEG = 20.0

# ====================== FLORIS MODEL SETUP ======================
_speeds_grid = np.arange(7.5, 13.0, 0.5, dtype=np.float32)
_freq_table = np.zeros((len(WIND_DIRECTIONS_DEG), len(_speeds_grid)), dtype=np.float32)
for _i, (_spd, _frq) in enumerate(zip(WIND_SPEEDS_MS, WIND_FREQS)):
    _j = int(round((_spd - 7.5) / 0.5))
    _freq_table[_i, _j] = _frq

fmodel = FlorisModel(configuration="defaults")
fmodel.set(
    wind_data=WindRose(
        wind_directions=WIND_DIRECTIONS_DEG,
        wind_speeds=_speeds_grid,
        freq_table=_freq_table,
        ti_table=0.06,
    )
)

# ====================== COUNTERS & GLOBALS ======================
FLORIS_CALLS = 0

# FIX 10: hard per-generation cap — prevents the relative gate from
# degrading to "evaluate everything" once the population matures.
FLORIS_CALLS_THIS_GEN = 0
MAX_FLORIS_PER_GEN = 10  # was 5 — doubles FLORIS budget, still only 20% of pop
# FIX 6: initialise to 0.0, not -1e9.
BEST_APPROX = 0.0

# ====================== GPU TENSORS (built once) ======================
t_dirs = torch.tensor(WIND_DIRECTIONS_DEG, device=device)
t_speeds = torch.tensor(WIND_SPEEDS_MS, device=device)
t_freqs = torch.tensor(WIND_FREQS, device=device)
t_thetas = torch.deg2rad(t_dirs)
CONE_RAD = torch.deg2rad(torch.tensor(WAKE_CONE_DEG, device=device))

_ONE_MINUS_SQRT_1_MINUS_CT = 1.0 - (1.0 - CT) ** 0.5


# ====================== POWER CURVE ======================
def power_curve_vectorized(u: torch.Tensor) -> torch.Tensor:
    """Piecewise 5 MW power curve. Returns kW."""
    p = torch.zeros_like(u)
    rated = 12.0
    p[(u >= 3.0) & (u < rated)] = 5000.0 * (u[(u >= 3.0) & (u < rated)] / rated) ** 3
    p[(u >= rated) & (u <= 25.0)] = 5000.0
    return p


# ====================== HYBRID FITNESS ======================
def fitness_func(ga_instance, solution, solution_idx):
    """
    Hybrid two-fidelity fitness.
    Both return paths give GWh/yr — PyGAD SSS compares on a consistent scale.
    D=8 directions, N=150 turbines.

    FIX 10: ALL globals that are assigned inside this function must be declared
    at the TOP of the function body. Placing 'global FLORIS_CALLS_THIS_GEN'
    inside the if-block means Python's static analysis may resolve it as a
    local at the earlier condition check (FLORIS_CALLS_THIS_GEN < MAX_FLORIS_PER_GEN),
    causing an UnboundLocalError or silently reading stale data depending on
    CPython version. Declare all at the top — no exceptions.
    """
    # FIX 10: declare every assigned global here, unconditionally.
    global BEST_APPROX, FLORIS_CALLS, FLORIS_CALLS_THIS_GEN

    pos = torch.tensor(
        solution.reshape((2, NUM_TURBINES)).T, dtype=torch.float32, device=device
    )
    pos_np = pos.cpu().numpy()

    # --- Feasibility checks ---
    in_poly = np.array([FARM_POLYGON.contains(Point(x, y)) for x, y in pos_np])
    if not in_poly.all():
        return -1e9

    dist_mat = np.linalg.norm(pos_np[:, None] - pos_np[None, :], axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    if (dist_mat < MIN_DISTANCE).any():
        return -1e9

    # --- GPU Jensen wake (D × N × N, zero Python loops) ---
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 2): diff[i,j]=pos[j]-pos[i]
    dists = torch.hypot(diff[..., 0], diff[..., 1])  # (N, N)
    angle_ij = torch.atan2(diff[..., 1], diff[..., 0])  # (N, N)

    wind_to_dir = 1.5 * torch.pi - t_thetas  # (D,)
    delta = angle_ij.unsqueeze(0) - wind_to_dir.view(-1, 1, 1)
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))

    downwind_d = dists.unsqueeze(0) * torch.cos(delta)
    valid_wake = (
        (torch.abs(delta) <= CONE_RAD) & (downwind_d > 0) & (dists.unsqueeze(0) > 1e-6)
    )

    safe_dist = downwind_d.clamp(min=1e-6)
    deficit = (
        _ONE_MINUS_SQRT_1_MINUS_CT * (ROTOR_R / (ROTOR_R + ALPHA * safe_dist)) ** 2
    )
    deficit = deficit * valid_wake.float()

    # FIX 2: dim=1 → sum over upstream i, giving deficit received at j
    total_deficit = torch.sqrt((deficit**2).sum(dim=1))  # (D, N)

    u_eff = t_speeds.view(-1, 1) * (1.0 - total_deficit)  # (D, N)
    power_kw = power_curve_vectorized(u_eff)  # (D, N)

    # FIX 3: return GWh/yr so both paths are on the same scale
    expected_kw = float((power_kw.sum(dim=1) * t_freqs).sum().item())
    approx_gwh = expected_kw * HOURS_PER_YEAR / 1e6

    if approx_gwh > BEST_APPROX:
        BEST_APPROX = approx_gwh

    # --- FLORIS gate: relative quality filter AND hard per-generation cap ---
    #
    # FIX 10 — two-part guard:
    #   Part A (relative): approx_gwh > 0.85 * BEST_APPROX
    #       filters obvious losers, same as before.
    #   Part B (absolute): FLORIS_CALLS_THIS_GEN < MAX_FLORIS_PER_GEN
    #       hard ceiling so a mature population (where Part A passes nearly
    #       everyone) cannot flood FLORIS. Both must be True.
    #
    # Short-circuit evaluation means Part B is only tested when Part A
    # passes, keeping the cheap Jensen path fast for clear losers.
    if approx_gwh > 0.88 * BEST_APPROX and FLORIS_CALLS_THIS_GEN < MAX_FLORIS_PER_GEN:
        try:
            fmodel.set(
                layout_x=solution[:NUM_TURBINES],
                layout_y=solution[NUM_TURBINES:],
            )
            fmodel.run()
            aep_gwh = fmodel.get_farm_AEP() / 1e9  # FIX 3: Wh → GWh
            FLORIS_CALLS += 1
            FLORIS_CALLS_THIS_GEN += 1
            return float(aep_gwh)
        except Exception as e:
            if solution_idx == 0:
                print(f"[FLORIS ERROR] {type(e).__name__}: {e}")
            return approx_gwh

    return approx_gwh


# ====================== VALID PLACEMENT HELPERS ======================
def generate_valid_position(existing_positions):
    """
    Try up to 200 random placements inside polygon with MIN_DISTANCE spacing.
    FIX 8: Raise RuntimeError on failure instead of silently returning a bad coord.
    """
    for _ in range(200):
        x = np.random.uniform(POLYGON_BOUNDS[0], POLYGON_BOUNDS[2])
        y = np.random.uniform(POLYGON_BOUNDS[1], POLYGON_BOUNDS[3])
        if not FARM_POLYGON.contains(Point(x, y)):
            continue
        if all(
            np.hypot(x - ex, y - ey) >= MIN_DISTANCE for ex, ey in existing_positions
        ):
            return x, y
    raise RuntimeError(
        f"generate_valid_position: could not place turbine after 200 attempts "
        f"({len(existing_positions)} existing turbines). "
        "Farm may be over-saturated for MIN_DISTANCE constraint."
    )


def create_valid_individual():
    """Build one valid individual: [x0..x149, y0..y149]."""
    xs, ys, existing = [], [], []
    for _ in range(NUM_TURBINES):
        for attempt in range(10):
            try:
                x, y = generate_valid_position(existing)
                break
            except RuntimeError:
                if attempt == 9:
                    raise RuntimeError(
                        f"create_valid_individual: failed to place turbine "
                        f"{len(xs) + 1}/{NUM_TURBINES} after 10 × 200 attempts."
                    )
        xs.append(x)
        ys.append(y)
        existing.append((x, y))
    return np.concatenate((xs, ys))


# ====================== FULL REPAIR ======================
def repair_individual(solution):
    """
    FIX 1: matrix.T.flatten() restores block layout [x0..xN, y0..yN].
    """
    matrix = solution.reshape((2, NUM_TURBINES)).T.copy()  # (N, 2)
    pos_list = [(matrix[i, 0], matrix[i, 1]) for i in range(NUM_TURBINES)]

    order = list(range(NUM_TURBINES))
    np.random.shuffle(order)

    for idx in order:
        x, y = matrix[idx]
        others = [pos_list[j] for j in range(NUM_TURBINES) if j != idx]

        if FARM_POLYGON.contains(Point(x, y)) and all(
            np.hypot(x - ex, y - ey) >= MIN_DISTANCE for ex, ey in others
        ):
            continue

        try:
            new_x, new_y = generate_valid_position(others)
        except RuntimeError:
            continue

        matrix[idx, 0] = new_x
        matrix[idx, 1] = new_y
        pos_list[idx] = (new_x, new_y)

    return matrix.T.flatten()  # FIX 1: block layout restored


# ====================== CUSTOM CROSSOVER ======================
def custom_crossover(parents, offspring_size, ga_instance):
    """
    FIX 9: Split point is a turbine index (0..N), not a gene index (0..2N).
    """
    offspring = np.empty(offspring_size)
    num_parents = parents.shape[0]

    for i in range(offspring_size[0]):
        p1_idx = np.random.randint(0, num_parents)
        p2_idx = np.random.randint(0, num_parents - 1)
        if p2_idx >= p1_idx:
            p2_idx += 1
        p1, p2 = parents[p1_idx], parents[p2_idx]

        k = np.random.randint(1, NUM_TURBINES)

        offspring[i, :k] = p1[:k]
        offspring[i, k:NUM_TURBINES] = p2[k:NUM_TURBINES]
        offspring[i, NUM_TURBINES : NUM_TURBINES + k] = p1[
            NUM_TURBINES : NUM_TURBINES + k
        ]
        offspring[i, NUM_TURBINES + k :] = p2[NUM_TURBINES + k :]

    for i in range(offspring_size[0]):
        offspring[i] = repair_individual(offspring[i])

    return offspring


# ====================== GEOMETRIC MUTATION ======================
def geometric_mutation(offspring, ga_instance):
    """
    FIX 7: turbine indices sampled without replacement (replace=False).
    """
    for i in range(offspring.shape[0]):
        matrix = offspring[i].reshape((2, NUM_TURBINES)).copy()
        num_mutate = np.random.randint(15, 21)

        indices = np.random.choice(NUM_TURBINES, size=num_mutate, replace=False)

        for turbine_idx in indices:
            for _ in range(30):
                new_x = np.random.uniform(POLYGON_BOUNDS[0], POLYGON_BOUNDS[2])
                new_y = np.random.uniform(POLYGON_BOUNDS[1], POLYGON_BOUNDS[3])
                if not FARM_POLYGON.contains(Point(new_x, new_y)):
                    continue
                valid = True
                for other in range(NUM_TURBINES):
                    if other == turbine_idx:
                        continue
                    if (
                        np.hypot(new_x - matrix[0, other], new_y - matrix[1, other])
                        < MIN_DISTANCE
                    ):
                        valid = False
                        break
                if valid:
                    matrix[0, turbine_idx] = new_x
                    matrix[1, turbine_idx] = new_y
                    break

        offspring[i] = repair_individual(matrix.flatten())

    return offspring


# ====================== GENERATION CALLBACK ======================
best_fitness_history = []
_start_time = None


def on_generation(ga_instance):
    # FIX 10: reset the per-generation FLORIS counter BEFORE printing,
    # so the count shown reflects calls made IN this generation only.
    global FLORIS_CALLS_THIS_GEN
    calls_this_gen = FLORIS_CALLS_THIS_GEN  # snapshot for the print line
    FLORIS_CALLS_THIS_GEN = 0  # reset for next generation

    gen = ga_instance.generations_completed
    best = float(max(ga_instance.last_generation_fitness))
    best_fitness_history.append(best)
    elapsed = time.time() - _start_time
    # FIX 4: use generations_completed in denominator, not hardcoded 200
    total_evals = 50 + max(gen - 1, 0) * (50 - ga_instance.keep_parents)
    print(
        f"  Gen {gen:3d} | Best: {best:.4f} GWh/yr | "
        f"FLORIS this gen: {calls_this_gen:2d}/{MAX_FLORIS_PER_GEN} | "
        f"FLORIS total: {FLORIS_CALLS:4d} ({FLORIS_CALLS / max(total_evals, 1) * 100:.1f}%) | "
        f"Elapsed: {elapsed:.0f}s",
        flush=True,
    )


# ====================== INITIAL POPULATION ======================
print("\nGenerating valid initial population (seed=42, pop=50)...", flush=True)
np.random.seed(42)
torch.manual_seed(42)
initial_population = np.array([create_valid_individual() for _ in range(50)])
print("Done. Starting hybrid GA...\n", flush=True)

# ====================== PYGAD SETUP ======================
_start_time = time.time()

ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=NUM_TURBINES * 2,
    initial_population=initial_population,
    parent_selection_type="sss",
    crossover_type=custom_crossover,
    mutation_type=geometric_mutation,
    gene_type=float,
    keep_parents=10,
    stop_criteria=["saturate_30"],
    on_generation=on_generation,
)

ga_instance.run()
total_time = time.time() - _start_time

# ====================== RESULTS ======================
solution, best_aep_gwh, _ = ga_instance.best_solution()
positions = solution.reshape((2, NUM_TURBINES)).T  # (N, 2)

actual_evals = 50 + max(ga_instance.generations_completed - 1, 0) * (
    50 - ga_instance.keep_parents
)  # FIX 4
floris_rate = FLORIS_CALLS / max(actual_evals, 1) * 100

print(f"\n{'=' * 60}")
print("HYBRID GA RESULTS")
print(f"{'=' * 60}")
print(f"  Best AEP (fitness)   : {best_aep_gwh:.4f} GWh/yr")
print(f"  Generations run      : {ga_instance.generations_completed}")
print(f"  Total evaluations    : {actual_evals}")
print(f"  FLORIS calls         : {FLORIS_CALLS}")
print(f"  FLORIS call rate     : {floris_rate:.2f}%")
print(f"  Runtime              : {total_time / 60:.1f} min")
print(f"{'=' * 60}")

# ====================== SAVE LAYOUT ======================
os.makedirs("../results", exist_ok=True)
os.makedirs("../paper_artifacts/layouts", exist_ok=True)
np.save("../results/hybrid_best_layout.npy", positions)
np.save("../paper_artifacts/layouts/hybrid_best_layout.npy", positions)
print("Layout saved -> ../results/hybrid_best_layout.npy and ../paper_artifacts/layouts/hybrid_best_layout.npy")

# ====================== FLORIS RESCORE ======================
print("\nRescoring best layout with fresh FLORIS instance...")
fm_rescore = FlorisModel(configuration="defaults")
fm_rescore.set(
    wind_data=WindRose(
        wind_directions=WIND_DIRECTIONS_DEG,
        wind_speeds=_speeds_grid,
        freq_table=_freq_table,
        ti_table=0.06,
    )
)
fm_rescore.set(layout_x=positions[:, 0], layout_y=positions[:, 1])
fm_rescore.run()
aep_floris_gwh = fm_rescore.get_farm_AEP() / 1e9

print(f"  FLORIS rescored AEP  : {aep_floris_gwh:.4f} GWh/yr")
print("  (Use this number in Table II)")

with open("../results/hybrid_summary.txt", "w") as f:
    f.write("HYBRID GA RESULTS\n")
    f.write("=" * 40 + "\n")
    f.write(f"Best AEP (FLORIS)    : {aep_floris_gwh:.4f} GWh/yr\n")
    f.write(f"Best AEP (fitness)   : {best_aep_gwh:.4f} GWh/yr\n")
    f.write(f"Generations run      : {ga_instance.generations_completed}\n")
    f.write(f"Total evaluations    : {actual_evals}\n")
    f.write(f"FLORIS calls         : {FLORIS_CALLS}\n")
    f.write(f"FLORIS call rate     : {floris_rate:.2f}%\n")
    f.write(f"Runtime (min)        : {total_time / 60:.1f}\n")
    f.write("Hardware             : GPU (Jensen) + CPU (FLORIS)\n")
print("Summary saved -> ../results/hybrid_summary.txt")

# ====================== PLOT 1: LAYOUT ======================
fig, ax = plt.subplots(figsize=(7, 7))
poly_x, poly_y = FARM_POLYGON.exterior.xy
ax.fill(poly_x, poly_y, alpha=0.12, color="red")
ax.plot(poly_x, poly_y, "r-", linewidth=2.5, label="Farm Boundary")
ax.scatter(
    positions[:, 0],
    positions[:, 1],
    s=180,
    c="steelblue",
    edgecolors="navy",
    zorder=5,
    label="Turbines",
)
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(
    f"Hybrid GA - Optimized Layout\nAEP = {aep_floris_gwh:.2f} GWh/yr  (FLORIS-scored)"
)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("../results/hybrid_layout.png", dpi=150, bbox_inches="tight")
print("Layout plot -> ../results/hybrid_layout.png")

# ====================== PLOT 2: CONVERGENCE ======================
fig2, ax2 = plt.subplots(figsize=(8, 4))
gens = np.arange(1, len(best_fitness_history) + 1)
fitns = np.array(best_fitness_history)

ax2.plot(gens, fitns, color="steelblue", linewidth=2, label="Hybrid GA (Jensen+FLORIS)")
ax2.axhline(
    y=best_aep_gwh,
    color="steelblue",
    linestyle="--",
    alpha=0.5,
    label=f"Final: {best_aep_gwh:.2f} GWh/yr",
)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best AEP (GWh/yr)")
ax2.set_title("Hybrid GA - Convergence Curve")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/hybrid_convergence.png", dpi=150, bbox_inches="tight")
print("Convergence plot -> ../results/hybrid_convergence.png")
plt.close("all")
print("\nDone. Use FLORIS rescored AEP from hybrid_summary.txt for Table II.")
