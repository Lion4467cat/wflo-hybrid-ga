"""
Score Ablation Layout with FLORIS
===================================
FLORIS-scores both the random baseline layout and the Jensen-only GA
best layout, giving an apples-to-apples comparison against final.py.

HOW TO RUN:
    python score_floris.py

REQUIRES:
    ablation_result.npy          — best layout from ablation_jensen_only.py
    random_baseline_layout.npy   — best random layout from random_baseline.py
    floris v4 installed
"""

import numpy as np
import warnings
import pathlib
import inspect
from floris import FlorisModel

warnings.filterwarnings("ignore")

# ====================== WIND ROSE (identical to final.py) ======================
WIND_DIRECTIONS = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
WIND_SPEEDS = [7.5, 8.5, 10.0, 11.5, 12.5, 11.0, 9.0, 8.0]
WIND_FREQS = np.array([0.08, 0.12, 0.15, 0.18, 0.22, 0.13, 0.07, 0.05])
WIND_FREQS = WIND_FREQS / WIND_FREQS.sum()

# ====================== FLORIS SETUP ======================
DEFAULT_CONFIG = str(
    pathlib.Path(inspect.getfile(FlorisModel)).parent / "default_inputs.yaml"
)
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACT_LAYOUT_DIR = REPO_ROOT / "paper_artifacts" / "layouts"


def resolve_layout(filename):
    path = try_resolve_layout(filename)
    if path is not None:
        return path

    candidates = [
        ARTIFACT_LAYOUT_DIR / filename,
        SCRIPT_DIR / filename,
        REPO_ROOT / "results" / filename,
    ]
    raise FileNotFoundError(
        f"Missing {filename}. Looked in: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def try_resolve_layout(filename):
    candidates = [
        ARTIFACT_LAYOUT_DIR / filename,
        SCRIPT_DIR / filename,
        REPO_ROOT / "results" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def floris_score(npy_path, label):
    positions = np.load(npy_path)
    assert positions.shape == (150, 2), f"Unexpected shape {positions.shape}"
    fmodel = FlorisModel(DEFAULT_CONFIG)
    fmodel.set(
        layout_x=positions[:, 0].tolist(),
        layout_y=positions[:, 1].tolist(),
        wind_directions=WIND_DIRECTIONS,
        wind_speeds=WIND_SPEEDS,
        turbulence_intensities=[0.06] * 8,
    )
    fmodel.run()
    aep_gwh = fmodel.get_farm_AEP(freq=WIND_FREQS) / 1e9
    print(f"  {label}: {aep_gwh:.2f} GWh/yr")
    return aep_gwh


# ====================== SCORE BOTH LAYOUTS ======================
print("Scoring layouts with FLORIS (Gauss wake model)...\n")
random_aep = floris_score(resolve_layout("random_baseline_layout.npy"), "Random baseline ")
ablation_aep = floris_score(resolve_layout("ablation_result.npy"), "Jensen-only GA  ")

hybrid_path = try_resolve_layout("hybrid_best_layout.npy")
if hybrid_path is not None:
    hybrid_aep = floris_score(hybrid_path, "Hybrid GA      ")
    hybrid_label = "Hybrid GA (rescored)"
else:
    hybrid_aep = 4029.16
    hybrid_label = "Hybrid GA (paper ref)"
    print("  Hybrid GA      : no saved layout found; using paper reference 4029.16 GWh/yr")

# ====================== RESULTS ======================
ablation_vs_random = (ablation_aep - random_aep) / random_aep * 100
hybrid_vs_random = (hybrid_aep - random_aep) / random_aep * 100
hybrid_vs_ablation = (hybrid_aep - ablation_aep) / ablation_aep * 100

print("\n" + "=" * 58)
print("TABLE II  -  All three rows FLORIS-scored")
print("=" * 58)
print(f"  {'Method':<25} {'AEP (GWh/yr)':>14}  {'vs Random':>10}")
print(f"  {'-' * 25} {'-' * 14}  {'-' * 10}")
print(f"  {'Random baseline':<25} {random_aep:>14.2f}  {'-':>10}")
print(f"  {'Jensen-only GA':<25} {ablation_aep:>14.2f}  {ablation_vs_random:>+9.2f}%")
print(f"  {hybrid_label:<25} {hybrid_aep:>14.2f}  {hybrid_vs_random:>+9.2f}%")
print("=" * 58)
print(
    f"\n  Hybrid gain over Jensen-only: "
    f"+{hybrid_aep - ablation_aep:.2f} GWh/yr ({hybrid_vs_ablation:+.2f}%)"
)
print("  -> This is the contribution of FLORIS validation in the fitness loop.")
print("=" * 58)

# ====================== SAVE ======================
with open("ablation_floris_score.txt", "w") as f:
    f.write("FLORIS-SCORED ABLATION RESULT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Random baseline  : {random_aep:.2f} GWh/yr\n")
    f.write(
        f"Jensen-only GA   : {ablation_aep:.2f} GWh/yr  ({ablation_vs_random:+.2f}% vs random)\n"
    )
    f.write(
        f"{hybrid_label:<16}: {hybrid_aep:.2f} GWh/yr  ({hybrid_vs_random:+.2f}% vs random)\n"
    )
    f.write(
        f"\nHybrid gain over Jensen-only : +{hybrid_aep - ablation_aep:.2f} GWh/yr ({hybrid_vs_ablation:+.2f}%)\n"
    )
    f.write("Scorer           : FLORIS v4 (Gauss wake model)\n")

print("\nSaved -> ablation_floris_score.txt")
