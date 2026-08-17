"""
Ablation Study — Jensen-Only Fitness
=====================================
PURPOSE: Isolate the contribution of FLORIS validation by removing it
entirely. This is the direct ablation of final.py: identical GA config,
identical operators, identical farm — only the fitness signal differs.

CHANGES vs final.py:
  - FLORIS completely removed from fitness function
  - Jensen GPU estimate is the sole fitness signal (returned in GWh)
  - No hybrid gate, no FLORIS calls
  - save_best_solutions=True  → convergence curve saved
  - All other GA params and operators match final.py (pop=50, gen=200, seed=42)

HOW TO RUN:
    python ablation_jensen_only.py

EXPECTED RUNTIME on RTX 4060:
    ~15–40 minutes (no FLORIS bottleneck)

OUTPUT FILES:
    ablation_jensen_only_layout.png   — optimized turbine layout
    ablation_convergence.png          — fitness vs generation curve
    ablation_result.npy               — best layout positions (N,2)
    ablation_summary.txt              — key numbers for paper Table II
"""

import numpy as np
import pygad
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import torch
import time
import warnings

warnings.filterwarnings("ignore")

# ====================== CUDA SETUP ======================
USE_CUDA = True
if USE_CUDA and torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA detected — RTX 4060 active")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("Running on CPU (no CUDA found)")

# ====================== FARM PARAMETERS ======================
# IDENTICAL to final.py — do not change
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
# IDENTICAL to final.py
WIND_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
WIND_SPEEDS_MS = np.array(
    [7.5, 8.5, 10.0, 11.5, 12.5, 11.0, 9.0, 8.0], dtype=np.float32
)
WIND_FREQS = np.array(
    [0.08, 0.12, 0.15, 0.18, 0.22, 0.13, 0.07, 0.05], dtype=np.float32
)
WIND_FREQS = WIND_FREQS / WIND_FREQS.sum()

WAKE_CONE_DEG = 20.0

# ====================== GPU TENSORS ======================
t_dirs = torch.tensor(WIND_DIRECTIONS_DEG, device=device)
t_speeds = torch.tensor(WIND_SPEEDS_MS, device=device)
t_freqs = torch.tensor(WIND_FREQS, device=device)
t_thetas = torch.deg2rad(t_dirs)
CONE_RAD = torch.deg2rad(torch.tensor(WAKE_CONE_DEG, device=device))

_ONE_MINUS_SQRT_1_MINUS_CT = 1.0 - (1.0 - CT) ** 0.5


# ====================== POWER CURVE ======================
def power_curve_vectorized(u: torch.Tensor) -> torch.Tensor:
    """Piecewise 5 MW power curve — identical to final.py. Returns kW."""
    p = torch.zeros_like(u)
    rated = 12.0
    p[(u >= 3.0) & (u < rated)] = 5000.0 * (u[(u >= 3.0) & (u < rated)] / rated) ** 3
    p[(u >= rated) & (u <= 25.0)] = 5000.0
    return p


# ====================== JENSEN-ONLY FITNESS ======================
def fitness_func(ga_instance, solution, solution_idx):
    """
    Pure GPU Jensen fitness — no FLORIS, no hybrid gate.

    Tensor layout:
      diff = pos.unsqueeze(0) - pos.unsqueeze(1)
      → diff[a, b] = pos[b] - pos[a]  (vector FROM turbine a TO turbine b)
      → valid_wake[d, a, b] = True means turbine a wakes turbine b
      → deficit[d, a, b] = wake deficit AT turbine b CAUSED BY turbine a

    To get total deficit RECEIVED by turbine b:
      sum over a  →  .sum(dim=1)   ← CORRECT
      (dim=2 would sum over b = "total wake a sends out" = wrong)

    Returns AEP in GWh/yr.
    """
    pos = torch.tensor(
        solution.reshape((2, NUM_TURBINES)).T, dtype=torch.float32, device=device
    )
    pos_np = pos.cpu().numpy()

    # --- Feasibility checks (CPU, fast) ---
    in_poly = np.array([FARM_POLYGON.contains(Point(x, y)) for x, y in pos_np])
    if not in_poly.all():
        return -1e9

    dist_mat = np.linalg.norm(pos_np[:, None] - pos_np[None, :], axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    if (dist_mat < MIN_DISTANCE).any():
        return -1e9

    # --- GPU Jensen wake (D×N×N tensor, zero Python loops) ---
    # diff[a, b] = pos[b] - pos[a]  →  vector FROM a TO b
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 2)
    dists = torch.hypot(diff[..., 0], diff[..., 1])  # (N, N)
    angle_ij = torch.atan2(diff[..., 1], diff[..., 0])  # (N, N)

    wind_to_dir = 1.5 * torch.pi - t_thetas  # (D,)
    delta = angle_ij.unsqueeze(0) - wind_to_dir.view(-1, 1, 1)
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))  # wrap to [-π, π]

    # downwind_d[d, a, b] > 0 when b is downwind of a
    downwind_d = dists.unsqueeze(0) * torch.cos(delta)  # (D, N, N)
    valid_wake = (
        (torch.abs(delta) <= CONE_RAD) & (downwind_d > 0) & (dists.unsqueeze(0) > 1e-6)
    )

    safe_dist = downwind_d.clamp(min=1e-6)
    deficit = (
        _ONE_MINUS_SQRT_1_MINUS_CT * (ROTOR_R / (ROTOR_R + ALPHA * safe_dist)) ** 2
    )
    deficit = deficit * valid_wake.float()

    # deficit[d, a, b] = deficit AT b FROM a
    # sum over a (dim=1) → total deficit received by each turbine b
    total_deficit = torch.sqrt((deficit**2).sum(dim=1))  # (D, N)

    u_eff = t_speeds.view(-1, 1) * (1.0 - total_deficit)  # (D, N)
    power_kw = power_curve_vectorized(u_eff)  # (D, N)

    # Frequency-weighted expected power → AEP in GWh/yr
    expected_power_kw = float((power_kw.sum(dim=1) * t_freqs).sum().item())
    aep_gwh = expected_power_kw * HOURS_PER_YEAR / 1e6

    return aep_gwh


# ====================== VALID PLACEMENT HELPERS ======================
# IDENTICAL to final.py
def generate_valid_position(existing_positions):
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
        f"({len(existing_positions)} existing turbines)."
    )


def create_valid_individual():
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
                        f"{len(xs) + 1}/{NUM_TURBINES} after 10 x 200 attempts."
                    )
        xs.append(x)
        ys.append(y)
        existing.append((x, y))
    return np.concatenate((xs, ys))


def repair_individual(solution):
    """IDENTICAL to final.py — full repair, shuffled order, preserves valid turbines."""
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

    # (N,2).T → (2,N) → flatten gives [x0..xN, y0..yN]  ← matches reshape((2,N)) everywhere
    return matrix.T.flatten()


# ====================== OPERATORS (IDENTICAL to final.py) ======================
def custom_crossover(parents, offspring_size, ga_instance):
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


def geometric_mutation(offspring, ga_instance):
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


# ====================== TIMING CALLBACK ======================
_start_time = None


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best = max(ga_instance.previous_generation_fitness)
    elapsed = time.time() - _start_time
    print(f"  Gen {gen:3d} | AEP: {best:.2f} GWh/yr | Elapsed: {elapsed:.0f}s")


# ====================== MAIN ======================
def main():
    global _start_time

    print("\nGenerating valid initial population (seed=42, identical to final.py)...", flush=True)
    np.random.seed(42)
    initial_population_list = []
    for idx in range(50):
        initial_population_list.append(create_valid_individual())
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  initial population: {idx + 1}/50", flush=True)
    initial_population = np.array(initial_population_list)
    print(f"Done — {len(initial_population)} individuals ready.\n", flush=True)

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
        save_best_solutions=True,
        on_generation=on_generation,
    )

    print("=" * 55, flush=True)
    print("ABLATION: Jensen-Only GA  |  pop=50, gen=200, seed=42", flush=True)
    print("=" * 55, flush=True)
    _start_time = time.time()
    ga_instance.run()
    total_time = time.time() - _start_time

    solution, best_aep_gwh, _ = ga_instance.best_solution()
    positions = solution.reshape((2, NUM_TURBINES)).T  # (N, 2)

    print("\n" + "=" * 55)
    print("ABLATION RESULTS")
    print("=" * 55)
    print(f"  Jensen fitness   : {best_aep_gwh:.2f} GWh/yr")
    print(f"  Generations run  : {ga_instance.generations_completed}")
    print(f"  Total time       : {total_time / 60:.1f} min")
    print("  FLORIS calls     : 0  (ablation — Jensen only)")
    print("  Compare to random only after FLORIS rescoring with score_floris.py")
    print("=" * 55)

    # --- Save results ---
    np.save("ablation_result.npy", positions)

    with open("ablation_summary.txt", "w", encoding="utf-8") as f:
        f.write("ABLATION STUDY — Jensen-Only GA\n")
        f.write("=" * 40 + "\n")
        f.write(f"Jensen-Only AEP  : {best_aep_gwh:.2f} GWh/yr\n")
        f.write("Scoring note     : internal Jensen fitness, not Table II FLORIS AEP\n")
        f.write(f"Generations run  : {ga_instance.generations_completed}\n")
        f.write(f"Evaluations      : {ga_instance.generations_completed * 50}\n")
        f.write(f"Runtime (min)    : {total_time / 60:.1f}\n")
        f.write("FLORIS calls     : 0\n")
        f.write("Hardware         : GPU (Jensen) only\n")
    print("Summary saved -> ablation_summary.txt")

    # --- Layout plot ---
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
        f"Jensen-Only Ablation — Optimized Layout\n"
        f"Jensen fitness = {best_aep_gwh:.2f} GWh/yr"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("ablation_jensen_only_layout.png", dpi=150, bbox_inches="tight")
    print("Layout plot saved -> ablation_jensen_only_layout.png")

    # --- Convergence plot ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    gens = np.arange(1, len(ga_instance.best_solutions_fitness) + 1)
    fitns = np.array(ga_instance.best_solutions_fitness)

    ax2.plot(gens, fitns, color="steelblue", linewidth=2, label="Jensen-Only GA")
    ax2.axhline(
        y=best_aep_gwh,
        color="steelblue",
        linestyle="--",
        alpha=0.5,
        label=f"Final: {best_aep_gwh:.2f} GWh/yr",
    )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best AEP (GWh/yr)")
    ax2.set_title("Jensen-Only Ablation — Convergence Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ablation_convergence.png", dpi=150, bbox_inches="tight")
    print("Convergence plot saved -> ablation_convergence.png")

    plt.show()
    print("\nDone. Run score_floris.py next to get the FLORIS-scored Table II numbers.")


if __name__ == "__main__":
    main()
