"""
Random baseline for the WFLO ablation study.

Generates one feasible 150-turbine layout with seed=42, scores it with the
same FLORIS wind rose used by the validation scripts, and saves:

    random_baseline_layout.npy
    random_baseline.txt
"""

import logging
import warnings

import numpy as np
from floris import FlorisModel, WindRose
from shapely.geometry import Point, Polygon

logging.getLogger("floris").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

NUM_TURBINES = 150
AREA_SIZE = 10000.0
MIN_DISTANCE = 400.0

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

WIND_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
WIND_SPEEDS_MS = np.array(
    [7.5, 8.5, 10.0, 11.5, 12.5, 11.0, 9.0, 8.0], dtype=np.float32
)
WIND_FREQS = np.array(
    [0.08, 0.12, 0.15, 0.18, 0.22, 0.13, 0.07, 0.05], dtype=np.float32
)
WIND_FREQS = WIND_FREQS / WIND_FREQS.sum()

_SPEEDS_GRID = np.arange(7.5, 13.0, 0.5, dtype=np.float32)
_FREQ_TABLE = np.zeros((len(WIND_DIRECTIONS_DEG), len(_SPEEDS_GRID)), dtype=np.float32)
for _i, (_spd, _frq) in enumerate(zip(WIND_SPEEDS_MS, WIND_FREQS)):
    _j = int(round((_spd - 7.5) / 0.5))
    _FREQ_TABLE[_i, _j] = _frq

WIND_ROSE = WindRose(
    wind_directions=WIND_DIRECTIONS_DEG,
    wind_speeds=_SPEEDS_GRID,
    freq_table=_FREQ_TABLE,
    ti_table=0.06,
)


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
    raise RuntimeError("Could not place a turbine within the attempt budget.")


def create_valid_individual():
    xs, ys, existing = [], [], []
    for turbine_idx in range(NUM_TURBINES):
        for attempt in range(10):
            try:
                x, y = generate_valid_position(existing)
                break
            except RuntimeError:
                if attempt == 9:
                    raise RuntimeError(
                        f"Failed to place turbine {turbine_idx + 1}/{NUM_TURBINES}."
                    )
        xs.append(x)
        ys.append(y)
        existing.append((x, y))
    return np.column_stack([xs, ys])


def floris_score(positions_np):
    fm = FlorisModel(configuration="defaults")
    fm.set(wind_data=WIND_ROSE)
    fm.set(layout_x=positions_np[:, 0], layout_y=positions_np[:, 1])
    fm.run()
    return fm.get_farm_AEP() / 1e9


def load_random_baseline_gwh(path="random_baseline.txt"):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if ":" not in line or "GWh" not in line:
                continue
            try:
                return float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                continue
    raise ValueError(f"Could not parse a GWh value from {path}.")


def main():
    np.random.seed(42)
    positions = create_valid_individual()
    aep_gwh = floris_score(positions)

    dmat = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    min_spacing = float(dmat.min())

    np.save("random_baseline_layout.npy", positions)
    with open("random_baseline.txt", "w", encoding="utf-8") as f:
        f.write("RANDOM BASELINE\n")
        f.write("=" * 40 + "\n")
        f.write(f"FLORIS AEP      : {aep_gwh:.2f} GWh/yr\n")
        f.write(f"Min spacing     : {min_spacing:.3f} m\n")
        f.write("Seed            : 42\n")
        f.write("Scorer          : FLORIS defaults\n")

    print(f"Random baseline FLORIS AEP: {aep_gwh:.2f} GWh/yr")
    print(f"Minimum spacing: {min_spacing:.3f} m")
    print("Saved random_baseline_layout.npy and random_baseline.txt")


if __name__ == "__main__":
    main()
