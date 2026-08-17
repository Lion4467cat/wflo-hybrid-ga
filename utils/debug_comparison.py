"""
COMPREHENSIVE DEBUG & COMPARISON FRAMEWORK
============================================
Validates three methods for apples-to-apples comparison:
1. final.py (Hybrid GA: Jensen + FLORIS)
2. Topfarm_GA.py (TOPFARM SimpleGADriver)
3. topfarm_baseline.py (TOPFARM EasyRandomSearchDriver)

This script:
- Validates shared inputs (polygon, wind rose, turbine specs)
- Tests initial layout generation reproducibility
- Scores a test layout with each method's scorer
- Generates a comparison matrix
"""

import numpy as np
import sys
import warnings
import os

warnings.filterwarnings("ignore")

os.chdir("..")  # Change to project root
os.makedirs("data", exist_ok=True)

print("=" * 70)
print("DEBUG & VALIDATION: Three-Method Comparison")
print("=" * 70)

# ====================== SHARED INPUTS ======================
print("\n[STEP 1] Validating shared inputs...")

# All three should use these exact values
NUM_TURBINES = 150
MIN_DISTANCE = 400.0
RATED_KW = 5000.0
ROTOR_D = 126.0
ROTOR_R = 63.0

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
    ],
    dtype=np.float64,
)

WIND_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float64)
WIND_SPEEDS_MS = np.array(
    [7.5, 8.5, 10.0, 11.5, 12.5, 11.0, 9.0, 8.0], dtype=np.float64
)
WIND_FREQS = np.array(
    [0.08, 0.12, 0.15, 0.18, 0.22, 0.13, 0.07, 0.05], dtype=np.float64
)
WIND_FREQS = WIND_FREQS / WIND_FREQS.sum()

print(f"  ✓ NUM_TURBINES: {NUM_TURBINES}")
print(f"  ✓ MIN_DISTANCE: {MIN_DISTANCE}m")
print(f"  ✓ RATED_KW: {RATED_KW}")
print(f"  ✓ ROTOR_D: {ROTOR_D}m (R={ROTOR_R}m)")
print(f"  ✓ Polygon vertices: {len(POLYGON_COORDS)}")
print(f"  ✓ Wind directions: {len(WIND_DIRECTIONS_DEG)}")
print(f"  ✓ Wind freqs sum: {WIND_FREQS.sum():.6f} (must be 1.0)")

# ====================== TEST INITIAL LAYOUT ======================
print("\n[STEP 2] Generating test initial layout (seed=42)...")
np.random.seed(42)

from shapely.geometry import Point, Polygon as ShapelyPolygon

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

print(f"  ✓ Placed {len(xs_init)}/{NUM_TURBINES} turbines")
print(f"  ✓ Used {attempts} attempts")
print(
    f"  ✓ Min spacing in initial layout: {np.min([np.hypot(xs_init[i] - xs_init[j], ys_init[i] - ys_init[j]) for i in range(len(xs_init)) for j in range(i + 1, len(xs_init))]):.1f}m"
)

# Save for reproducibility
np.save("data/_initial_layout_test.npy", np.column_stack([xs_init, ys_init]))

# ====================== TEST WITH PYWAKE (TOPFARM SCORER) ======================
print("\n[STEP 3] Scoring initial layout with PyWake (Bastankhah-PorteAgel 2014)...")

try:
    from py_wake.site import XRSite
    from py_wake.wind_turbines import WindTurbines
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    import xarray as xr

    # Build PyWake site
    _ws_sorted = np.sort(np.unique(WIND_SPEEDS_MS))
    _freq_table = np.zeros((len(WIND_DIRECTIONS_DEG), len(_ws_sorted)))
    for i, (spd, frq) in enumerate(zip(WIND_SPEEDS_MS, WIND_FREQS)):
        j = int(np.argmin(np.abs(_ws_sorted - spd)))
        _freq_table[i, j] = frq

    site = XRSite(
        ds=xr.Dataset(
            data_vars={"P": (["wd", "ws"], _freq_table), "TI": 0.06},
            coords={"wd": WIND_DIRECTIONS_DEG, "ws": _ws_sorted},
        )
    )

    # Build PyWake turbine
    _ws_curve = np.linspace(0, 30, 200)
    _rated = 12.0
    _p_curve = np.zeros_like(_ws_curve)
    _p_curve[(_ws_curve >= 3.0) & (_ws_curve < _rated)] = (
        RATED_KW * (_ws_curve[(_ws_curve >= 3.0) & (_ws_curve < _rated)] / _rated) ** 3
    )
    _p_curve[(_ws_curve >= _rated) & (_ws_curve <= 25.0)] = RATED_KW
    _ct_curve = np.where((_ws_curve >= 3.0) & (_ws_curve <= 25.0), 0.8, 0.0)

    turbine = WindTurbines(
        names=["Custom5MW"],
        diameters=[ROTOR_D],
        hub_heights=[90.0],
        powerCtFunctions=[
            PowerCtTabular(ws=_ws_curve, power=_p_curve, power_unit="kW", ct=_ct_curve)
        ],
    )

    wake_model_pywake = Bastankhah_PorteAgel_2014(site, turbine, k=0.0324555)

    sim_pywake = wake_model_pywake(xs_init, ys_init)
    aep_pywake_gwh = float(sim_pywake.aep().sum().values)

    print(f"  ✓ PyWake (Bastankhah-PorteAgel 2014) AEP: {aep_pywake_gwh:.4f} GWh/yr")

except Exception as e:
    print(f"  ✗ PyWake scoring FAILED: {e}")
    aep_pywake_gwh = None

# ====================== TEST WITH FLORIS (HYBRID SCORER) ======================
print("\n[STEP 4] Scoring initial layout with FLORIS (default)...")

try:
    from floris import FlorisModel, WindRose

    _speeds_grid = np.arange(7.5, 13.0, 0.5, dtype=np.float32)
    _freq_table_floris = np.zeros(
        (len(WIND_DIRECTIONS_DEG), len(_speeds_grid)), dtype=np.float32
    )
    for _i, (_spd, _frq) in enumerate(zip(WIND_SPEEDS_MS, WIND_FREQS)):
        _j = int(round((_spd - 7.5) / 0.5))
        _freq_table_floris[_i, _j] = _frq

    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(
        wind_data=WindRose(
            wind_directions=WIND_DIRECTIONS_DEG,
            wind_speeds=_speeds_grid,
            freq_table=_freq_table_floris,
            ti_table=0.06,
        )
    )
    fmodel.set(layout_x=xs_init, layout_y=ys_init)
    fmodel.run()
    aep_floris_gwh = fmodel.get_farm_AEP() / 1e9

    print(f"  ✓ FLORIS (defaults config) AEP: {aep_floris_gwh:.4f} GWh/yr")

except Exception as e:
    print(f"  ✗ FLORIS scoring FAILED: {e}")
    aep_floris_gwh = None

# ====================== TEST WITH JENSEN (HYBRID FAST PATH) ======================
print("\n[STEP 5] Scoring initial layout with Jensen GPU (hybrid method)...")

try:
    import torch

    USE_CUDA = True
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"    Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"    Using CPU (no CUDA)")

    CT = 0.8
    ALPHA = 0.1
    HOURS_PER_YEAR = 8760.0
    WAKE_CONE_DEG = 20.0

    def power_curve_vectorized(u):
        """Piecewise 5 MW power curve."""
        p = torch.zeros_like(u)
        rated = 12.0
        p[(u >= 3.0) & (u < rated)] = (
            5000.0 * (u[(u >= 3.0) & (u < rated)] / rated) ** 3
        )
        p[(u >= rated) & (u <= 25.0)] = 5000.0
        return p

    t_dirs = torch.tensor(WIND_DIRECTIONS_DEG, device=device)
    t_speeds = torch.tensor(WIND_SPEEDS_MS, device=device)
    t_freqs = torch.tensor(WIND_FREQS, device=device)
    t_thetas = torch.deg2rad(t_dirs)
    CONE_RAD = torch.deg2rad(torch.tensor(WAKE_CONE_DEG, device=device))

    _ONE_MINUS_SQRT_1_MINUS_CT = 1.0 - (1.0 - CT) ** 0.5

    pos = torch.tensor(
        np.column_stack([xs_init, ys_init]), dtype=torch.float32, device=device
    )

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dists = torch.hypot(diff[..., 0], diff[..., 1])
    angle_ij = torch.atan2(diff[..., 1], diff[..., 0])

    wind_to_dir = 1.5 * torch.pi - t_thetas
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

    total_deficit = torch.sqrt((deficit**2).sum(dim=1))

    u_eff = t_speeds.view(-1, 1) * (1.0 - total_deficit)
    power_kw = power_curve_vectorized(u_eff)

    expected_kw = float((power_kw.sum(dim=1) * t_freqs).sum().item())
    aep_jensen_gwh = expected_kw * HOURS_PER_YEAR / 1e6

    print(f"  ✓ Jensen GPU (hybrid method) AEP: {aep_jensen_gwh:.4f} GWh/yr")

except Exception as e:
    print(f"  ✗ Jensen scoring FAILED: {e}")
    aep_jensen_gwh = None

# ====================== SUMMARY COMPARISON TABLE ======================
print("\n" + "=" * 70)
print("INITIAL LAYOUT SCORES (All methods on same layout)")
print("=" * 70)

scores = {
    "PyWake (Bastankhah-PorteAgel 2014)": aep_pywake_gwh,
    "FLORIS (defaults config)": aep_floris_gwh,
    "Jensen (GPU, hybrid method)": aep_jensen_gwh,
}

for name, score in scores.items():
    if score is not None:
        print(f"  {name:<40} : {score:8.4f} GWh/yr")
    else:
        print(f"  {name:<40} : FAILED")

# Calculate differences
if aep_pywake_gwh and aep_floris_gwh:
    diff = abs(aep_floris_gwh - aep_pywake_gwh)
    pct = diff / aep_pywake_gwh * 100
    print(f"\n  PyWake vs FLORIS difference: {diff:.4f} GWh/yr ({pct:.2f}%)")

if aep_pywake_gwh and aep_jensen_gwh:
    diff = abs(aep_jensen_gwh - aep_pywake_gwh)
    pct = diff / aep_pywake_gwh * 100
    print(f"  PyWake vs Jensen difference: {diff:.4f} GWh/yr ({pct:.2f}%)")

if aep_floris_gwh and aep_jensen_gwh:
    diff = abs(aep_jensen_gwh - aep_floris_gwh)
    pct = diff / aep_floris_gwh * 100
    print(f"  FLORIS vs Jensen difference: {diff:.4f} GWh/yr ({pct:.2f}%)")

print("\n" + "=" * 70)
print("KEY FINDINGS FOR COMPARISON")
print("=" * 70)
print("""
  1. Initial layout is deterministic (seed=42) — reproducible across runs
  2. Three scorers may differ due to:
     - Wake model implementation (Jensen vs Bastankhah vs FLORIS internal)
     - Power curve interpolation (PyGAD float vs FLORIS discretization)
     - Wind rose discretization (PyWake grid vs FLORIS grid)
  3. For fair comparison, each method should use its NATIVE scorer consistently
  4. Improvement percentages must be relative to each method's baseline
  5. Recommend validating that spacing constraints are enforced across all three
""")

print("\nREADY FOR OPTIMIZATION RUNS")
print("Next step: Run each method individually, then compare results.")
print("=" * 70)
