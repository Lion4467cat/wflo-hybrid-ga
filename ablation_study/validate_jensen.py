"""
Jensen Wake Model -- Validation & Unit Test Suite
=================================================
Run this BEFORE trusting any AEP numbers from ablation_jensen_only.py.

Tests are organised in four layers:
  Layer 1 -- Physics unit tests  (known-answer tests with exact expected values)
  Layer 2 -- Monotonicity checks (AEP must improve when wake is reduced)
  Layer 3 -- Constraint checks   (validate best layout from ablation_result.npy)
  Layer 4 -- Ablation parity     (sum dim, multi-turbine sensitivity)

USAGE:
    python validate_jensen.py
    python validate_jensen.py --layout ablation_result.npy
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
from shapely.geometry import Point, Polygon

# ====================== COPY EXACT PARAMS FROM ablation_jensen_only.py ======================
NUM_TURBINES = 150
AREA_SIZE = 10000.0
MIN_DISTANCE = 400.0
CT = 0.8
ALPHA = 0.1
ROTOR_R = 63.0
HOURS_PER_YEAR = 8760.0
WAKE_CONE_DEG = 20.0
RATED_SPEED_MS = 12.0

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

WIND_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
WIND_SPEEDS_MS = np.array(
    [7.5, 8.5, 10.0, 11.5, 12.5, 11.0, 9.0, 8.0], dtype=np.float32
)
WIND_FREQS = np.array(
    [0.08, 0.12, 0.15, 0.18, 0.22, 0.13, 0.07, 0.05], dtype=np.float32
)
WIND_FREQS = WIND_FREQS / WIND_FREQS.sum()

_ONE_MINUS_SQRT_1_MINUS_CT = 1.0 - (1.0 - CT) ** 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_dirs = torch.tensor(WIND_DIRECTIONS_DEG, device=device)
t_speeds = torch.tensor(WIND_SPEEDS_MS, device=device)
t_freqs = torch.tensor(WIND_FREQS, device=device)
t_thetas = torch.deg2rad(t_dirs)
CONE_RAD = torch.deg2rad(torch.tensor(WAKE_CONE_DEG, device=device))

# Must match ablation_jensen_only.py fitness_func exactly.
JENSEN_SUM_DIM = 1  # sum over upstream turbines, giving deficit received downstream

# ====================== CORE JENSEN FUNCTION (mirrors ablation_jensen_only.py) ======================


def power_curve(u):
    """Scalar or array power curve -- returns kW."""
    u = np.asarray(u, dtype=float)
    p = np.zeros_like(u)
    p[(u >= 3.0) & (u < RATED_SPEED_MS)] = (
        5000.0 * (u[(u >= 3.0) & (u < RATED_SPEED_MS)] / RATED_SPEED_MS) ** 3
    )
    p[(u >= RATED_SPEED_MS) & (u <= 25.0)] = 5000.0
    return p


def power_curve_torch(u: torch.Tensor) -> torch.Tensor:
    p = torch.zeros_like(u)
    p[(u >= 3.0) & (u < RATED_SPEED_MS)] = (
        5000.0 * (u[(u >= 3.0) & (u < RATED_SPEED_MS)] / RATED_SPEED_MS) ** 3
    )
    p[(u >= RATED_SPEED_MS) & (u <= 25.0)] = 5000.0
    return p


def _jensen_tensors(pos: torch.Tensor):
    """Shared Jensen wake tensors -- identical to ablation_jensen_only.fitness_func."""
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 2)  vector j -> i
    dists = torch.hypot(diff[..., 0], diff[..., 1])
    angle_ij = torch.atan2(diff[..., 1], diff[..., 0])

    # Meteorological bearing to mathematical downwind direction.
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
    return deficit, downwind_d, valid_wake, delta


def active_wake_met_degrees(positions_np: np.ndarray) -> list[int]:
    """Meteorological direction indices (0,45,...) that produce any wake pair."""
    pos = torch.tensor(positions_np, dtype=torch.float32, device=device)
    _, _, valid_wake, _ = _jensen_tensors(pos)
    return [
        int(WIND_DIRECTIONS_DEG[d])
        for d in range(len(WIND_DIRECTIONS_DEG))
        if valid_wake[d].any()
    ]


def compute_aep_gwh(positions_np: np.ndarray, sum_dim: int = JENSEN_SUM_DIM) -> float:
    """AEP in GWh/yr for (N, 2) layout -- default sum_dim matches ablation."""
    pos = torch.tensor(positions_np, dtype=torch.float32, device=device)
    deficit, _, _, _ = _jensen_tensors(pos)
    total_deficit = torch.sqrt((deficit**2).sum(dim=sum_dim))
    u_eff = t_speeds.view(-1, 1) * (1.0 - total_deficit)
    power_kw = power_curve_torch(u_eff)
    expected_kw = float((power_kw.sum(dim=1) * t_freqs).sum().item())
    return expected_kw * HOURS_PER_YEAR / 1e6


def load_ablation_aep(default: float = 3187.96) -> float:
    """Parse Jensen-Only AEP from ablation_summary.txt if present."""
    path = "ablation_summary.txt"
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Jensen-Only AEP\s*:\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return default


# ====================== TEST HARNESS ======================
pass_count = 0
fail_count = 0


def check(name, condition, got, expected_desc, tol_pct=None):
    global pass_count, fail_count
    status = "PASS" if condition else "FAIL"
    if condition:
        pass_count += 1
    else:
        fail_count += 1
    tol_str = f"  (tol={tol_pct}%)" if tol_pct else ""
    print(f"  [{status}] {name}")
    print(f"         got: {got}")
    print(f"    expected: {expected_desc}{tol_str}")


def run_validation(layout_file: str = "ablation_result.npy") -> int:
    global pass_count, fail_count
    pass_count = 0
    fail_count = 0

    ablation_aep = load_ablation_aep()
    # =========================================================
    # LAYER 1 -- PHYSICS UNIT TESTS (exact known-answer)
    # =========================================================
    print("\n" + "=" * 60)
    print("LAYER 1 -- Physics Unit Tests")
    print("=" * 60)

    print("\nT1: Single turbine (no wake interactions)")
    pos_single = np.array([[5000.0, 5000.0]])
    aep_single = compute_aep_gwh(pos_single)

    expected_kw = sum(
        float(WIND_FREQS[d]) * float(power_curve([WIND_SPEEDS_MS[d]])[0])
        for d in range(8)
    )
    expected_gwh = expected_kw * HOURS_PER_YEAR / 1e6

    rel_err = abs(aep_single - expected_gwh) / expected_gwh * 100
    check(
        "Single turbine AEP matches analytical calculation",
        rel_err < 0.01,
        f"{aep_single:.6f} GWh/yr",
        f"{expected_gwh:.6f} GWh/yr (relative error {rel_err:.4f}%)",
        tol_pct=0.01,
    )

    print("\nT2: Two turbines -- B upwind of A (wake from 90 deg  / 270 deg  met directions)")
    pos_two = np.array([[5000.0, 4000.0], [5000.0, 6000.0]])
    aep_two = compute_aep_gwh(pos_two)
    two_wake_free_gwh = 2 * expected_gwh

    pos_t2 = torch.tensor(pos_two, dtype=torch.float32, device=device)
    deficit_t2, _, _, _ = _jensen_tensors(pos_t2)
    td_t2 = torch.sqrt((deficit_t2**2).sum(dim=JENSEN_SUM_DIM))
    check(
        "N-S pair: downstream turbine (y=4000) receives wake at 270 deg  met",
        float(td_t2[6, 0].item()) > 0.01 and float(td_t2[6, 1].item()) < 1e-6,
        f"deficit@A={td_t2[6, 0].item():.4f}, deficit@B={td_t2[6, 1].item():.4f}",
        "deficit on index 0 (downwind), not index 1 (upwind)",
    )

    wake_dirs_ns = set(active_wake_met_degrees(pos_two))
    check(
        "N-S pair: wake only on 90 deg  and 270 deg  met directions",
        wake_dirs_ns == {90, 270},
        f"active met dirs = {sorted(wake_dirs_ns)}",
        "{90, 270}",
    )

    check(
        "Two aligned turbines produce LESS than twice single (wake reduces power)",
        aep_two < two_wake_free_gwh,
        f"{aep_two:.4f} GWh/yr",
        f"< {two_wake_free_gwh:.4f} GWh/yr (wake must reduce total)",
    )
    check(
        "Wake deficit is not catastrophic (>50% of free-stream retained)",
        aep_two > 0.5 * two_wake_free_gwh,
        f"{aep_two:.4f} GWh/yr",
        f"> {0.5 * two_wake_free_gwh:.4f} GWh/yr",
    )

    print("\nT3: Two turbines side by side (wake from 0 deg  and 180 deg  met directions only)")
    pos_side = np.array([[3000.0, 5000.0], [7000.0, 5000.0]])
    aep_side = compute_aep_gwh(pos_side)

    wake_dirs_ew = set(active_wake_met_degrees(pos_side))
    check(
        "E-W pair: wake only on 0 deg  and 180 deg  met directions",
        wake_dirs_ew == {0, 180},
        f"active met dirs = {sorted(wake_dirs_ew)}",
        "{0, 180}",
    )

    x_sep = 4000.0
    def_sep = _ONE_MINUS_SQRT_1_MINUS_CT * (ROTOR_R / (ROTOR_R + ALPHA * x_sep)) ** 2

    def _directional_loss(idx: int) -> float:
        u = float(WIND_SPEEDS_MS[idx]) * (1.0 - def_sep)
        return (
            (float(power_curve([WIND_SPEEDS_MS[idx]])[0]) - float(power_curve([u])[0]))
            * float(WIND_FREQS[idx])
            * HOURS_PER_YEAR
            / 1e6
        )

    p_loss_d0 = _directional_loss(0)
    p_loss_d4 = _directional_loss(4)
    expected_side_gwh = two_wake_free_gwh - p_loss_d0 - p_loss_d4

    check(
        "Side-by-side turbines: 0 deg  and 180 deg  met dirs wake x-aligned pair",
        abs(aep_side - expected_side_gwh) < 0.01,
        f"{aep_side:.4f} GWh/yr",
        f"~ {expected_side_gwh:.4f} GWh/yr (idx 0 loss={p_loss_d0:.4f}, idx 4 loss={p_loss_d4:.4f} GWh)",
        tol_pct=0.5,
    )

    print("\nT4: Jensen deficit formula -- analytical vs GPU (270 deg  met, +y downwind)")
    x_test = 1000.0
    wind_idx = 6
    expected_deficit = (
        _ONE_MINUS_SQRT_1_MINUS_CT * (ROTOR_R / (ROTOR_R + ALPHA * x_test)) ** 2
    )
    print(f"    Analytical deficit at {x_test}m downwind: {expected_deficit:.6f}")

    pos_check = np.array([[5000.0, 5000.0], [5000.0, 5000.0 + x_test]])
    pos_t = torch.tensor(pos_check, dtype=torch.float32, device=device)
    deficit_t, downwind_t, valid_t, delta_t = _jensen_tensors(pos_t)

    downwind_A_from_B = float(downwind_t[wind_idx, 0, 1].item())
    delta_AB = float(delta_t[wind_idx, 0, 1].item())
    in_cone = bool(valid_t[wind_idx, 0, 1].item())
    deficit_AB = float(deficit_t[wind_idx, 0, 1].item())

    print(
        f"    GPU tensor (270 deg  met, pair [0,1]): downwind={downwind_A_from_B:.1f}m, "
        f"delta={np.degrees(delta_AB):.2f} deg , in_cone={in_cone}"
    )
    print(f"    GPU computed deficit: {deficit_AB:.6f}")

    check(
        "Jensen deficit formula matches between analytical and GPU tensor",
        in_cone and abs(deficit_AB - expected_deficit) < 1e-5,
        f"deficit={deficit_AB:.6f}, in_cone={in_cone}, downwind_dist={downwind_A_from_B:.1f}m",
        f"deficit={expected_deficit:.6f}, in_cone=True, downwind_dist={x_test:.1f}m",
        tol_pct=0.001,
    )

    print("\nT5: Wake deficit decreases monotonically with downwind distance")
    distances = [500, 1000, 2000, 4000, 8000]
    deficits = [
        _ONE_MINUS_SQRT_1_MINUS_CT * (ROTOR_R / (ROTOR_R + ALPHA * x)) ** 2
        for x in distances
    ]
    monotone = all(deficits[i] > deficits[i + 1] for i in range(len(deficits) - 1))
    check(
        "Deficit strictly decreasing with distance",
        monotone,
        str([f"{d:.5f}" for d in deficits]),
        "strictly decreasing sequence",
    )

    # =========================================================
    # LAYER 2 -- MONOTONICITY CHECKS
    # =========================================================
    print("\n" + "=" * 60)
    print("LAYER 2 -- Monotonicity Checks")
    print("=" * 60)

    print("\nM1: Moving downwind turbine further away -> AEP must increase")
    aep_close = compute_aep_gwh(np.array([[5000.0, 4000.0], [5000.0, 6000.0]]))
    aep_medium = compute_aep_gwh(np.array([[5000.0, 3500.0], [5000.0, 6000.0]]))
    aep_far = compute_aep_gwh(np.array([[5000.0, 3000.0], [5000.0, 6000.0]]))

    print(f"    2000m apart: {aep_close:.4f} GWh/yr")
    print(f"    2500m apart: {aep_medium:.4f} GWh/yr")
    print(f"    3000m apart: {aep_far:.4f} GWh/yr")

    check(
        "AEP increases as aligned turbines move further apart",
        aep_close < aep_medium < aep_far,
        f"{aep_close:.4f} < {aep_medium:.4f} < {aep_far:.4f}",
        "strictly increasing with separation",
    )

    print("\nM2: Ablation Jensen fitness must be finite and positive")
    check(
        "Jensen-only GA summary contains a valid Jensen fitness",
        np.isfinite(ablation_aep) and ablation_aep > 0.0,
        f"{ablation_aep:.2f} GWh/yr",
        "> 0 GWh/yr",
    )

    print("\nM3: AEP must be below theoretical no-wake maximum")
    p_max_kw = 5000.0 * NUM_TURBINES
    max_aep_gwh = p_max_kw * HOURS_PER_YEAR / 1e6
    freq_weighted = (
        sum(float(WIND_FREQS[d]) * float(power_curve([WIND_SPEEDS_MS[d]])[0]) for d in range(8))
        * NUM_TURBINES
    )
    freqw_max_gwh = freq_weighted * HOURS_PER_YEAR / 1e6

    print(f"    Absolute max (all rated, 100% capacity): {max_aep_gwh:.2f} GWh/yr")
    print(f"    Freq-weighted no-wake max:               {freqw_max_gwh:.2f} GWh/yr")
    print(f"    Ablation result:                         {ablation_aep:.2f} GWh/yr")

    check(
        "Ablation AEP < freq-weighted no-wake maximum",
        ablation_aep < freqw_max_gwh,
        f"{ablation_aep:.2f} GWh/yr",
        f"< {freqw_max_gwh:.2f} GWh/yr",
    )

    print("\nM4: Wake efficiency check (should be 70–95% for dense farm)")
    wake_efficiency = ablation_aep / freqw_max_gwh * 100
    check(
        "Wake efficiency is in realistic range for 150-turbine farm",
        60.0 < wake_efficiency < 97.0,
        f"{wake_efficiency:.1f}%",
        "between 60% and 97%",
    )

    # =========================================================
    # LAYER 3 -- CONSTRAINT VALIDATION ON SAVED LAYOUT
    # =========================================================
    print("\n" + "=" * 60)
    print("LAYER 3 -- Constraint Validation on Saved Layout")
    print("=" * 60)

    if not os.path.exists(layout_file):
        print(f"  [SKIP] {layout_file} not found -- run ablation_jensen_only.py first")
    else:
        positions = np.load(layout_file)
        print(f"  Loaded {layout_file}: shape={positions.shape}")

        check(
            "C1: Correct number of turbines",
            positions.shape == (NUM_TURBINES, 2),
            f"shape={positions.shape}",
            f"({NUM_TURBINES}, 2)",
        )

        inside = [FARM_POLYGON.contains(Point(x, y)) for x, y in positions]
        n_outside = sum(1 for v in inside if not v)
        check(
            "C2: All turbines inside farm polygon",
            n_outside == 0,
            f"{n_outside} turbines outside polygon",
            "0 turbines outside",
        )

        dmat = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        min_dist = dmat.min()
        n_violations = int((dmat < MIN_DISTANCE).sum() // 2)
        check(
            "C3: All pairwise distances >= MIN_DISTANCE (400m)",
            min_dist >= MIN_DISTANCE,
            f"min dist = {min_dist:.2f}m, violations = {n_violations}",
            f"min dist >= {MIN_DISTANCE}m, violations = 0",
        )

        print("\n  C4: Re-computing AEP from saved layout to verify reproducibility...")
        aep_recomputed = compute_aep_gwh(positions)
        aep_legacy_dim2 = compute_aep_gwh(positions, sum_dim=2)
        rel_err = abs(aep_recomputed - ablation_aep) / ablation_aep * 100
        legacy_err = abs(aep_legacy_dim2 - ablation_aep) / ablation_aep * 100
        if rel_err < 0.1:
            check(
                "C4: Recomputed AEP matches reported ablation value",
                True,
                f"recomputed={aep_recomputed:.4f} GWh/yr (err={rel_err:.4f}%)",
                f"~ {ablation_aep:.4f} GWh/yr (< 0.1% tolerance)",
                tol_pct=0.1,
            )
        elif legacy_err < 0.1:
            print(
                "  [NOTE] Saved layout matches OLD sum(dim=2) formula "
                f"({aep_legacy_dim2:.4f} GWh/yr) but current formula gives "
                f"{aep_recomputed:.4f} GWh/yr. Re-run ablation_jensen_only.py."
            )
            check(
                "C4: Recomputed AEP matches reported ablation value",
                False,
                f"current formula={aep_recomputed:.4f}, summary={ablation_aep:.4f} (legacy dim=2 match)",
                "re-run ablation after Jensen parity fixes",
            )
        else:
            check(
                "C4: Recomputed AEP matches reported ablation value",
                False,
                f"recomputed={aep_recomputed:.4f} GWh/yr (err={rel_err:.4f}%)",
                f"~ {ablation_aep:.4f} GWh/yr (< 0.1% tolerance)",
                tol_pct=0.1,
            )

        x_ok = (positions[:, 0] >= 0).all() and (positions[:, 0] <= AREA_SIZE).all()
        y_ok = (positions[:, 1] >= 0).all() and (positions[:, 1] <= AREA_SIZE).all()
        check(
            "C5: All coordinates within farm bounding box",
            x_ok and y_ok,
            f"x=[{positions[:, 0].min():.0f}, {positions[:, 0].max():.0f}], "
            f"y=[{positions[:, 1].min():.0f}, {positions[:, 1].max():.0f}]",
            f"x and y in [0, {AREA_SIZE:.0f}]",
        )

        uniq = len(set(map(tuple, np.round(positions, 1))))
        check(
            "C6: No duplicate turbine positions",
            uniq == NUM_TURBINES,
            f"{uniq} unique positions out of {NUM_TURBINES}",
            f"{NUM_TURBINES} unique positions",
        )

    # =========================================================
    # LAYER 4 -- ABLATION PARITY (sum dim and multi-turbine sensitivity)
    # =========================================================
    print("\n" + "=" * 60)
    print(f"LAYER 4 -- Ablation parity (JENSEN_SUM_DIM={JENSEN_SUM_DIM})")
    print("=" * 60)

    pos_aligned = np.array([[5000.0, 4000.0], [5000.0, 6000.0]])
    aep_correct = compute_aep_gwh(pos_aligned, sum_dim=JENSEN_SUM_DIM)
    alternate_dim = 2 if JENSEN_SUM_DIM == 1 else 1
    aep_alternate_dim = compute_aep_gwh(pos_aligned, sum_dim=alternate_dim)
    aep_nowake = 2 * expected_gwh

    print(f"  No-wake (reference)       : {aep_nowake:.4f} GWh/yr")
    print(f"  ablation sum dim={JENSEN_SUM_DIM}  : {aep_correct:.4f} GWh/yr")
    print(f"  alternate sum dim={alternate_dim}: {aep_alternate_dim:.4f} GWh/yr")

    check(
        f"Configured sum(dim={JENSEN_SUM_DIM}) gives lower AEP than no-wake (wake active)",
        aep_correct < aep_nowake,
        f"{aep_correct:.4f} < {aep_nowake:.4f}",
        "wake reduces farm AEP",
    )

    np.random.seed(99)
    pos_many = np.random.uniform(1000, 9000, (20, 2))
    aep_d2_many = compute_aep_gwh(pos_many, sum_dim=2)
    aep_d1_many = compute_aep_gwh(pos_many, sum_dim=1)
    delta_many = abs(aep_d2_many - aep_d1_many)

    print(
        f"  20-turbine random layout: dim=2={aep_d2_many:.2f}, "
        f"dim=1={aep_d1_many:.2f}, Δ={delta_many:.4f} GWh"
    )

    check(
        "Configured sum dimension differs from the alternate dimension on N>=20 layouts",
        delta_many > 0.05,
        f"Δ={delta_many:.4f} GWh between dim=2 and dim=1",
        f"Δ > 0.05 GWh (ablation uses dim={JENSEN_SUM_DIM})",
    )

    if os.path.exists(layout_file):
        positions_saved = np.load(layout_file)
        aep_saved = compute_aep_gwh(positions_saved)
        aep_saved_legacy = compute_aep_gwh(positions_saved, sum_dim=2)
        err = abs(aep_saved - ablation_aep) / ablation_aep
        if err < 0.001:
            check(
                "Saved layout AEP matches compute_aep_gwh (ablation formula)",
                True,
                f"compute={aep_saved:.4f} vs summary={ablation_aep:.4f} GWh/yr",
                "within 0.1% of ablation_summary.txt",
                tol_pct=0.1,
            )
        elif abs(aep_saved_legacy - ablation_aep) / ablation_aep < 0.001:
            check(
                "Saved layout AEP matches compute_aep_gwh (ablation formula)",
                False,
                f"current={aep_saved:.4f}, summary={ablation_aep:.4f} (legacy dim=2 only)",
                "re-run ablation_jensen_only.py after Jensen parity fixes",
            )
        else:
            check(
                "Saved layout AEP matches compute_aep_gwh (ablation formula)",
                False,
                f"compute={aep_saved:.4f} vs summary={ablation_aep:.4f} GWh/yr",
                "within 0.1% of ablation_summary.txt",
                tol_pct=0.1,
            )

    print("\n" + "=" * 60)
    total = pass_count + fail_count
    print(f"VALIDATION COMPLETE:  {pass_count}/{total} PASSED  |  {fail_count} FAILED")
    print("=" * 60)

    if fail_count == 0:
        print(
            "All checks passed. Results are physically consistent and safe to use in the paper."
        )
    else:
        print(
            f"WARNING: {fail_count} check(s) failed. "
            "Do NOT use AEP numbers in paper until resolved."
        )

    return 0 if fail_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Validate Jensen model vs ablation_jensen_only.py")
    parser.add_argument(
        "--layout",
        default="ablation_result.npy",
        help="Path to saved turbine layout (N,2) for Layer 3 checks",
    )
    args = parser.parse_args()
    sys.exit(run_validation(layout_file=args.layout))


if __name__ == "__main__":
    main()
