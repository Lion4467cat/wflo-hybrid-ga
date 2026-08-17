"""
TOPFARM Baseline — Correct implementation from official DTU notebooks
======================================================================
References used:
  - topfarm.pages.windenergy.dtu.dk/TopFarm2/notebooks/drivers.html
  - topfarm.pages.windenergy.dtu.dk/TopFarm2/notebooks/constraints.html
  - topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/Optimization.html

Key API facts from official docs:
  - EasyRandomSearchDriver(randomize_func, max_iter=N)  ← correct signature
  - Design vars: {'x': (init, lb, ub), 'y': (init, lb, ub)}
  - XYBoundaryConstraint(boundary_coords, 'polygon')
  - SpacingConstraint(min_distance)

Install:
    pip install topfarm py_wake
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from shapely.geometry import Point, Polygon as ShapelyPolygon
from floris import FlorisModel, WindRose
import os

# ====================== TOPFARM / PYWAKE IMPORTS ======================
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
import xarray as xr

from topfarm import TopFarmProblem
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.easy_drivers import EasyRandomSearchDriver

# ====================== YOUR EXACT FARM PARAMETERS ======================
NUM_TURBINES = 150
MIN_DISTANCE = 400.0
RATED_KW = 5000.0
ROTOR_D = 126.0

# Your exact irregular polygon (same as final.py)
POLYGON_COORDS = np.array(
    [
        [1250, 500],
        [8750, 500],
        [9500, 2500],
        [9500, 5000],
        [7500, 7000],
        [5000, 8750],
        [2000, 8000],
        [750, 6250],
        [500, 2500],
    ]
)  # no repeated last point for TOPFARM

# Your exact wind rose
WIND_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float64)
WIND_SPEEDS_MS = np.array(
    [7.5, 8.5, 10.0, 11.5, 12.5, 11.0, 9.0, 8.0], dtype=np.float64
)
WIND_FREQS = np.array(
    [0.08, 0.12, 0.15, 0.18, 0.22, 0.13, 0.07, 0.05], dtype=np.float64
)
WIND_FREQS = WIND_FREQS / WIND_FREQS.sum()

print("=" * 60)
print("TOPFARM BASELINE — DTU EasyRandomSearchDriver")
print("=" * 60)
print(f"  Turbines : {NUM_TURBINES} | D={ROTOR_D}m | {RATED_KW}kW")
print(f"  Min dist : {MIN_DISTANCE}m")
print(f"  Wind dirs: {len(WIND_DIRECTIONS_DEG)}")
print("=" * 60)

# ====================== BUILD PYWAKE SITE ======================
# One speed per direction → freq_table shape (n_wd, n_ws)
_ws_sorted = np.sort(np.unique(WIND_SPEEDS_MS))
_freq_table = np.zeros((len(WIND_DIRECTIONS_DEG), len(_ws_sorted)))
for i, (spd, frq) in enumerate(zip(WIND_SPEEDS_MS, WIND_FREQS)):
    j = int(np.argmin(np.abs(_ws_sorted - spd)))  # robust float comparison
    _freq_table[i, j] = frq

site = XRSite(
    ds=xr.Dataset(
        data_vars={
            "P": (["wd", "ws"], _freq_table),
            "TI": 0.06,
        },
        coords={"wd": WIND_DIRECTIONS_DEG, "ws": _ws_sorted},
    )
)

# ====================== BUILD PYWAKE TURBINE ======================
_ws_curve = np.linspace(0, 30, 200)
_rated = 12.0
_p_curve = np.zeros_like(_ws_curve)
_m_partial = (_ws_curve >= 3.0) & (_ws_curve < _rated)
_m_full = (_ws_curve >= _rated) & (_ws_curve <= 25.0)
_p_curve[_m_partial] = RATED_KW * (_ws_curve[_m_partial] / _rated) ** 3
_p_curve[_m_full] = RATED_KW
_ct_curve = np.where((_ws_curve >= 3.0) & (_ws_curve <= 25.0), 0.8, 0.0)

turbine = WindTurbines(
    names=["Custom5MW"],
    diameters=[ROTOR_D],
    hub_heights=[90.0],
    powerCtFunctions=[
        PowerCtTabular(ws=_ws_curve, power=_p_curve, power_unit="kW", ct=_ct_curve)
    ],
)

# ====================== WAKE MODEL ======================
# k=0.0324555 is standard Bastankhah 2014 expansion coefficient
wake_model = Bastankhah_PorteAgel_2014(site, turbine, k=0.0324555)

# ====================== VALID INITIAL LAYOUT ======================
print("\nGenerating valid initial layout...")
np.random.seed(42)
shapely_poly = ShapelyPolygon(np.vstack([POLYGON_COORDS, POLYGON_COORDS[0]]))
bounds = shapely_poly.bounds

xs_init, ys_init, existing = [], [], []
attempts = 0
while len(xs_init) < NUM_TURBINES and attempts < 200000:
    x = np.random.uniform(bounds[0], bounds[2])
    y = np.random.uniform(bounds[1], bounds[3])
    attempts += 1
    if not shapely_poly.contains(Point(x, y)):
        continue
    if all(np.hypot(x - ex, y - ey) >= MIN_DISTANCE for ex, ey in existing):
        xs_init.append(x)
        ys_init.append(y)
        existing.append((x, y))

xs_init = np.array(xs_init)
ys_init = np.array(ys_init)
print(f"  Placed {len(xs_init)} / {NUM_TURBINES} turbines")
assert len(xs_init) == NUM_TURBINES, (
    f"Only placed {len(xs_init)}/{NUM_TURBINES} turbines — "
    "increase attempt limit or reduce MIN_DISTANCE"
)

# ====================== BASELINE AEP (before optimization) ======================
print("Computing initial layout AEP...")
sim_init = wake_model(xs_init, ys_init)
aep_init_gwh = float(sim_init.aep().sum().values)  # PyWake always returns GWh
print(f"  Initial AEP: {aep_init_gwh:.2f} GWh/yr")

# ====================== CUSTOM RANDOMIZE FUNCTION FOR POLYGON ======================
# From official docs: randomize_func receives desvar_dict and returns modified dict
# We implement polygon-aware random search

_poly_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])

_move_attempts = {"total": 0, "success": 0}  # track move success rate


def randomize_positions(desvar_dict):
    """
    Custom randomize function for EasyRandomSearchDriver.
    Vectorized spacing check: O(N) per candidate instead of O(N²).

    TOPFARM passes tuples (values, lb, ub) - we extract values[0]
    """
    # Extract values from tuple format: (values, lower_bound, upper_bound)
    if isinstance(desvar_dict["x"], tuple):
        xs = np.array(desvar_dict["x"][0]).copy()
        ys = np.array(desvar_dict["y"][0]).copy()
        x_bounds = (desvar_dict["x"][1], desvar_dict["x"][2])
        y_bounds = (desvar_dict["y"][1], desvar_dict["y"][2])
        is_tuple_format = True
    else:
        xs = desvar_dict["x"].copy()
        ys = desvar_dict["y"].copy()
        is_tuple_format = False

    n = len(xs)

    # Randomly select 10-20% of turbines to move
    n_move = max(1, np.random.randint(int(0.1 * n), int(0.2 * n) + 1))
    indices = np.random.choice(n, n_move, replace=False)

    moved = 0
    for idx in indices:
        _move_attempts["total"] += 1
        # Try up to 100 times (increased from 50 for dense farms)
        for _ in range(100):
            new_x = np.random.uniform(_poly_bounds[0], _poly_bounds[2])
            new_y = np.random.uniform(_poly_bounds[1], _poly_bounds[3])
            if not shapely_poly.contains(Point(new_x, new_y)):
                continue
            # Vectorized spacing check — O(N) not O(N²)
            mask = np.arange(n) != idx
            dists = np.hypot(new_x - xs[mask], new_y - ys[mask])
            if dists.min() >= MIN_DISTANCE:
                xs[idx] = new_x
                ys[idx] = new_y
                moved += 1
                _move_attempts["success"] += 1
                break

    # Return in the same format we received
    if is_tuple_format:
        desvar_dict["x"] = (xs, x_bounds[0], x_bounds[1])
        desvar_dict["y"] = (ys, y_bounds[0], y_bounds[1])
    else:
        desvar_dict["x"] = xs
        desvar_dict["y"] = ys

    return desvar_dict


# ====================== TOPFARM PROBLEM ======================
# Coordinate bounds so optimizer works in metre-space
_xmin, _ymin = float(bounds[0]), float(bounds[1])
_xmax, _ymax = float(bounds[2]), float(bounds[3])

aep_comp = PyWakeAEPCostModelComponent(
    windFarmModel=wake_model,
    n_wt=len(xs_init),  # use actual placed count, not NUM_TURBINES (defensive)
)

problem = TopFarmProblem(
    design_vars={
        "x": (xs_init, _xmin, _xmax),
        "y": (ys_init, _ymin, _ymax),
    },
    cost_comp=aep_comp,
    constraints=[
        XYBoundaryConstraint(POLYGON_COORDS, "polygon"),
        SpacingConstraint(MIN_DISTANCE),
    ],
    # EasyRandomSearchDriver(randomize_func, max_iter=N) — from official docs
    driver=EasyRandomSearchDriver(
        randomize_func=randomize_positions,
        max_iter=300,
        max_time=7200,  # 2 hour wall clock limit
        disp=True,  # print iteration progress
    ),
    plot_comp=None,
    reports=False,
)

# ====================== RUN ======================
print("\nRunning TOPFARM EasyRandomSearchDriver (300 iterations)...")
print("Non-gradient method. Estimated time: 30-90 min.\n")

t_start = time.time()
cost, state, recorder = problem.optimize()
runtime = time.time() - t_start

xs_opt = state["x"]
ys_opt = state["y"]

# ====================== FINAL AEP ======================
sim_opt = wake_model(xs_opt, ys_opt)
aep_opt_gwh = float(sim_opt.aep().sum().values)  # PyWake always returns GWh
improvement = (aep_opt_gwh - aep_init_gwh) / aep_init_gwh * 100

print("\n" + "=" * 60)
print("TOPFARM BASELINE RESULTS")
print("=" * 60)
print(f"  Initial AEP  : {aep_init_gwh:.2f} GWh/yr")
print(f"  Optimized AEP: {aep_opt_gwh:.2f} GWh/yr")
print(f"  Improvement  : {improvement:+.2f}%")
print(f"  Runtime      : {runtime / 60:.1f} min")
print("  Method       : EasyRandomSearchDriver (300 iter)")
print("  Wake model   : Bastankhah-PorteAgel 2014 (PyWake)")
print("  Reference    : Pedersen et al. 2025, DTU TOPFARM v2.6.x")
print("=" * 60)
_total = _move_attempts["total"]
_succ = _move_attempts["success"]
_rate = 100 * _succ / _total if _total > 0 else 0
print(f"  Move success rate: {_succ}/{_total} ({_rate:.1f}%)")
if _rate < 50:
    print(
        "  WARNING: Move success rate < 50% — farm may be too dense for random search"
    )

os.makedirs("../results", exist_ok=True)
np.save("../results/topfarm_rs_layout.npy", np.column_stack([xs_opt, ys_opt]))
print("Layout saved → ../results/topfarm_rs_layout.npy")

# ====================== FLORIS RESCORE ======================
# Clean rescore with a fresh FlorisModel — confirms reproducibility
# and gives the authoritative number for Table II.
print("\nRescoring best layout with fresh FLORIS instance...")
_speeds_grid = np.arange(7.5, 13.0, 0.5, dtype=np.float32)
_freq_table = np.zeros((len(WIND_DIRECTIONS_DEG), len(_speeds_grid)), dtype=np.float32)
for i, (spd, frq) in enumerate(zip(WIND_SPEEDS_MS, WIND_FREQS)):
    j = int(round((spd - 7.5) / 0.5))
    _freq_table[i, j] = frq

fm_rescore = FlorisModel(configuration="defaults")
fm_rescore.set(
    wind_data=WindRose(
        wind_directions=WIND_DIRECTIONS_DEG,
        wind_speeds=_speeds_grid,
        freq_table=_freq_table,
        ti_table=0.06,
    )
)
fm_rescore.set(layout_x=xs_opt, layout_y=ys_opt)
fm_rescore.run()
aep_floris_gwh = fm_rescore.get_farm_AEP() / 1e9

print(f"  FLORIS rescored AEP  : {aep_floris_gwh:.4f} GWh/yr")
print("  (Use this number in Table II)")

# ====================== SAVE SUMMARY ======================
with open("../results/topfarm_rs_summary.txt", "w") as f:
    f.write("TOPFARM RANDOM SEARCH RESULTS (EasyRandomSearchDriver)\n")
    f.write("=" * 40 + "\n")
    f.write(f"Initial AEP (PyWake)      : {aep_init_gwh:.4f} GWh/yr\n")
    f.write(f"Optimized AEP (PyWake)    : {aep_opt_gwh:.4f} GWh/yr\n")
    f.write(f"Optimized AEP (FLORIS)    : {aep_floris_gwh:.4f} GWh/yr\n")
    f.write(f"Improvement              : {improvement:+.2f}%\n")
    f.write(f"Runtime (min)            : {runtime / 60:.1f}\n")
    f.write("Method                   : EasyRandomSearchDriver (300 iter)\n")
    f.write("Wake model (internal)     : Bastankhah-PorteAgel 2014 (PyWake)\n")
    f.write("Wake model (rescore)      : FLORIS (Jensen)\n")
    f.write("Hardware                 : CPU (no GPU)\n")
    f.write("Move success rate        : {_rate:.1f}%\n")
print("Summary saved: ../results/topfarm_rs_summary.txt\n")

# ====================== PLOT ======================
poly_closed = np.vstack([POLYGON_COORDS, POLYGON_COORDS[0]])
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, xs, ys, title in [
    (
        axes[0],
        xs_init,
        ys_init,
        f"Initial Random Layout\nAEP = {aep_init_gwh:.2f} GWh/yr",
    ),
    (
        axes[1],
        xs_opt,
        ys_opt,
        f"TOPFARM Optimized Layout\nAEP = {aep_opt_gwh:.2f} GWh/yr ({improvement:+.2f}%)",
    ),
]:
    ax.fill(poly_closed[:, 0], poly_closed[:, 1], alpha=0.12, color="red")
    ax.plot(
        poly_closed[:, 0], poly_closed[:, 1], "r-", linewidth=2, label="Farm Boundary"
    )
    ax.scatter(xs, ys, s=80, c="blue", zorder=5, label="Turbines")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle(
    "TOPFARM Baseline — DTU EasyRandomSearchDriver\n"
    "Bastankhah-PorteAgel 2014 | 150 Turbines | Irregular Polygon",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("../results/topfarm_baseline_result.png", dpi=150, bbox_inches="tight")
print("Plot saved: ../results/topfarm_baseline_result.png")
plt.show()
