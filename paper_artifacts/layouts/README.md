This directory contains the saved layout artifacts (`.npy` files) generated from verified runs (`seed=42`) using the `wflo` environment:

- `random_baseline_layout.npy`: Feasible random 150-turbine layout (FLORIS AEP: 3939.33 GWh/yr).
- `ablation_result.npy`: Best layout from Jensen-only GA ablation (FLORIS AEP: 3963.22 GWh/yr).
- `hybrid_best_layout.npy`: Best layout from Hybrid GA optimization (FLORIS AEP: 4029.16 GWh/yr).

These layouts can be rescored directly using `ablation_study/score_floris.py` or `ablation_study/validate_ablation_final.py` to confirm model-consistent AEP evaluation without waiting for full GA optimization reruns.

