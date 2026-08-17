# WFLO Hybrid GA

GPU-accelerated hybrid genetic algorithm for wind farm layout optimization.

## Overview

This project optimizes wind farm layouts for a 150-turbine irregular site using a hybrid genetic algorithm. A GPU-vectorized Jensen wake model provides fast candidate screening, while selected elite layouts are validated with FLORIS to improve the fidelity of the final optimization signal.

This work is accepted at IEEE AIC 2026.

## Key Results

All reported AEP values are FLORIS-rescored (defaults configuration, 8-direction wind rose) for model-consistent apples-to-apples comparison.

### Accepted Paper Results (IEEE AIC 2026 Reference)
These are the peer-reviewed, citable headline numbers published in the paper:

| Method | FLORIS AEP (GWh/yr) | Gain vs. Random | Status |
| :--- | :---: | :---: | :---: |
| **Random baseline** | 3923.46 | Reference | Feasible (>= 400 m) |
| **Jensen-only GA (Ablation)** | 3950.41 | +0.69% | Feasible (>= 400 m) |
| **TOPFARM Random Search** | 3952.01 | +0.73% | Feasible (>= 400 m) |
| **Hybrid GA (Proposed)** | **4029.16** | **+2.69%** | Feasible (>= 400 m) |
| *TOPFARM GA (Diagnostic)* | *3938.96* | *+0.39%* | *Infeasible (398.48 m < 400 m)* |

### Fresh Reproduction Run Results (Shipped Layout Artifacts)
Generated via fresh stochastic optimization runs (`seed=42`) in the `wflo` environment and saved under `paper_artifacts/layouts/`:

| Method | Fresh FLORIS AEP (GWh/yr) | Gain vs. Fresh Random | Saved Layout Artifact |
| :--- | :---: | :---: | :--- |
| **Random baseline** | 3939.33 | Reference | `paper_artifacts/layouts/random_baseline_layout.npy` |
| **Jensen-only GA (Ablation)** | 3963.22 | +0.61% | `paper_artifacts/layouts/ablation_result.npy` |
| **Hybrid GA (Proposed)** | **4029.16** | **+2.28%** | `paper_artifacts/layouts/hybrid_best_layout.npy` |

> [!NOTE]
> The shipped `.npy` files in `paper_artifacts/layouts/` are fresh, verified runs from this codebase using `seed=42`. The original paper arrays were generated during prior experimental runs; the fresh runs reproduce the consistent ~2.3%–2.7% optimization gain.

## Repository Structure

```text
.
|-- scripts/
|   |-- final.py                  # Hybrid GA: GPU Jensen screening plus FLORIS validation
|   |-- topfarm_baseline.py       # TOPFARM random-search baseline
|   `-- topfarm_ga_baseline.py    # TOPFARM SimpleGADriver GA baseline
|-- ablation_study/
|   |-- ablation_jensen_only.py   # Jensen-only GA ablation
|   |-- random_baseline.py        # Random baseline layout generation and scoring
|   |-- score_floris.py           # FLORIS rescoring for random and ablation layouts
|   |-- validate_ablation_final.py
|   |-- validate_jensen.py
|   `-- plot_ablation.py
|-- paper_artifacts/
|   `-- layouts/                  # Saved layouts for exact paper-table reproduction
|-- utils/
|   |-- debug_comparison.py       # Cross-method input and scorer checks
|   |-- generate_comparison.py    # Aggregates result summaries into CSV/plots
|   |-- validate_jensen_angles.py # Jensen angle-convention diagnostics
|   `-- VALIDATION_CHECKLIST.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

Generated plots, logs, PDFs, and result tables are intentionally excluded from version control. Stochastic optimization reruns can regenerate comparable outputs, but see the Key Results section for the distinction between the accepted-paper numbers and this repo's fresh-reproduction numbers.

## Setup

```bash
git clone https://github.com/Lion4467cat/wflo-hybrid-ga.git
cd wflo-hybrid-ga
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use conda, create or activate an environment with Python 3.11 and install the same requirements:

```bash
conda activate wflo
pip install -r requirements.txt
```

The pinned package versions in `requirements.txt` were generated from the local `wflo` conda environment. The original experiment environment used PyTorch 2.8.0 with CUDA 12.9 available; `requirements.txt` uses the pip-installable `torch==2.8.0` package.

## Usage

The current scripts write outputs using paths relative to their own folders, so run them from the indicated directories.

Run the hybrid GA:

```bash
cd scripts
python final.py
```

Run TOPFARM baselines:

```bash
cd scripts
python topfarm_baseline.py
python topfarm_ga_baseline.py
```

Run the Jensen-only ablation and FLORIS rescore:

```bash
cd ablation_study
python random_baseline.py
python ablation_jensen_only.py
python score_floris.py
python validate_ablation_final.py
```

The optimization scripts are stochastic. The saved layout artifacts below let you rescore this repo's fresh-reproduction results (see Key Results) without rerunning full optimization. Note: only the Hybrid GA artifact reproduces the accepted-paper AEP exactly (4029.16 GWh/yr); the random baseline and Jensen-only artifacts reproduce this repo's own fresh-run values (3939.33 / 3963.22 GWh/yr), which differ slightly from the original paper-reported values (3923.46 / 3950.41 GWh/yr) since those original runs were unseeded. The shipped artifact files are:

```text
paper_artifacts/layouts/random_baseline_layout.npy
paper_artifacts/layouts/ablation_result.npy
paper_artifacts/layouts/hybrid_best_layout.npy
```

Generate comparison artifacts after the optimization summaries exist:

```bash
cd utils
python generate_comparison.py
```

## Method Summary

The optimizer represents each layout as continuous turbine coordinates and enforces polygon containment plus a 400 m spacing constraint through feasibility checks and repair. It evaluates every candidate with a GPU-vectorized Jensen wake calculation over the eight-direction wind rose, then sends only strong candidates to FLORIS when the Jensen score clears a 0.88 running-best gate and the generation has not exceeded a 10-call FLORIS cap. Specifically, the PyGAD configuration uses `sol_per_pop=50` with `keep_parents=10`, yielding exactly $50 - 10 = 40$ new candidate layout evaluations per generation (matching Table I's reported 40-candidate count and the total evaluation formula: $50 + (58 - 1) \times 40 = 2330$). Runs are configured with a 200-generation limit and `saturate_30` early stopping; final AEP values are reported after FLORIS rescoring.

## Acknowledgments

Documentation contributions by Jiya Nitturkar, Department of Computer Science, B.M.S. College of Engineering.

## Citation

```bibtex
@inproceedings{wflo_hybrid_ga_2026,
  title = {GPU-Accelerated Hybrid Two-Fidelity Genetic Algorithm for Wind Farm Layout Optimization on Irregular Polygon Domains},
  author = {Sheshadri, Shailashree K. and Swamy, S. S. Gokula and V., Roshanth},
  booktitle = {Proceedings of IEEE AIC},
  year = {2026}
}
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
