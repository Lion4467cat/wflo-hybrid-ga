"""
UNIFIED COMPARISON RESULTS
===========================
Compares results from three methods using their native outputs.

Files expected:
  - ../results/hybrid_summary.txt (from scripts/final.py)
  - ../results/topfarm_ga_summary.txt (from scripts/topfarm_ga_baseline.py)
  - ../results/topfarm_rs_summary.txt (from scripts/topfarm_baseline.py)

Plus their optimized layouts:
  - ../results/hybrid_best_layout.npy
  - ../results/topfarm_ga_layout.npy
  - ../results/topfarm_rs_layout.npy

Generates:
  - ../results/comparison_results.csv
  - ../results/comparison_visualization.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

os.chdir("..")  # Change to project root

print("=" * 70)
print("UNIFIED COMPARISON FRAMEWORK")
print("=" * 70)

# ====================== READ RESULTS ======================
print("\n[STEP 1] Reading results from all three methods...\n")

results = {}
files_to_check = {
    "Hybrid GA": "results/hybrid_summary.txt",
    "TOPFARM GA": "results/topfarm_ga_summary.txt",
    "TOPFARM Baseline": "results/topfarm_rs_summary.txt",
}

for method_name, filename in files_to_check.items():
    if os.path.exists(filename):
        with open(filename, "r") as f:
            content = f.read()
            results[method_name] = content
            print(f"✓ {filename}")
    else:
        print(f"✗ {filename} NOT FOUND")

# ====================== PARSE RESULTS ======================
print("\n[STEP 2] Parsing results...\n")

parsed = {}
for method_name, content in results.items():
    data = {}
    for line in content.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Extract numeric values
            if (
                "AEP" in key
                or "Improvement" in key
                or "Runtime" in key
                or "rate" in key
            ):
                try:
                    # Handle percentage signs
                    val = val.replace("%", "").split()[0]
                    data[key] = float(val)
                except:
                    data[key] = val
            else:
                data[key] = val
    parsed[method_name] = data
    print(f"\n{method_name}:")
    for k, v in data.items():
        print(f"  {k}: {v}")

# ====================== CREATE COMPARISON TABLE ======================
print("\n[STEP 3] Creating comparison table...\n")

comparison_data = {
    "Method": [],
    "Initial AEP (GWh/yr)": [],
    "Optimized AEP (GWh/yr)": [],
    "Improvement (%)": [],
    "Runtime (min)": [],
}

for method_name in ["Hybrid GA", "TOPFARM GA", "TOPFARM Baseline"]:
    if method_name in parsed:
        data = parsed[method_name]

        # Try to extract values - handle different formats
        if method_name == "Hybrid GA":
            # Hybrid GA uses "Best AEP (FLORIS)" instead of "Optimized AEP"
            init_aep = np.nan  # Hybrid GA doesn't report initial AEP
            opt_aep = data.get("Best AEP (FLORIS)", np.nan)
            improvement = np.nan  # Hybrid GA doesn't calculate improvement %
            runtime = data.get("Runtime (min)", np.nan)
        else:
            # TOPFARM methods use standard format
            init_aep = data.get("Initial AEP (PyWake)", np.nan)
            opt_aep = data.get("Optimized AEP (FLORIS)", np.nan)
            improvement = data.get("Improvement", np.nan)
            runtime = data.get("Runtime (min)", np.nan)

        comparison_data["Method"].append(method_name)
        comparison_data["Initial AEP (GWh/yr)"].append(init_aep)
        comparison_data["Optimized AEP (GWh/yr)"].append(opt_aep)
        comparison_data["Improvement (%)"].append(improvement)
        comparison_data["Runtime (min)"].append(runtime)

df = pd.DataFrame(comparison_data)

print("Comparison Table:")
print(df.to_string(index=False))
print()

# ====================== SAVE CSV ======================
print("[STEP 4] Saving comparison CSV...\n")

csv_file = "results/comparison_results.csv"
df.to_csv(csv_file, index=False)
print(f"✓ Saved: {csv_file}")

# ====================== VISUALIZATION ======================
print("\n[STEP 5] Creating visualizations...\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Three-Method Comparison: Wind Farm Layout Optimization",
    fontsize=14,
    fontweight="bold",
)

# Plot 1: Initial vs Optimized AEP
ax = axes[0, 0]
methods = df["Method"].values
x = np.arange(len(methods))
width = 0.35

init_aeps = df["Initial AEP (GWh/yr)"].values
opt_aeps = df["Optimized AEP (GWh/yr)"].values

ax.bar(x - width / 2, init_aeps, width, label="Initial", alpha=0.8, color="lightcoral")
ax.bar(x + width / 2, opt_aeps, width, label="Optimized", alpha=0.8, color="steelblue")
ax.set_ylabel("AEP (GWh/yr)")
ax.set_title("Initial vs Optimized AEP")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Plot 2: Improvement percentage
ax = axes[0, 1]
improvements = df["Improvement (%)"].values
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ax.bar(methods, improvements, color=colors, alpha=0.8)
ax.set_ylabel("Improvement (%)")
ax.set_title("Optimization Improvement")
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(improvements):
    ax.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")

# Plot 3: Runtime comparison
ax = axes[1, 0]
runtimes = df["Runtime (min)"].values
ax.bar(methods, runtimes, color=colors, alpha=0.8)
ax.set_ylabel("Runtime (minutes)")
ax.set_title("Computation Time")
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(runtimes):
    ax.text(i, v + 2, f"{v:.1f}m", ha="center", fontweight="bold")

# Plot 4: Final AEP comparison
ax = axes[1, 1]
final_aeps = df["Optimized AEP (GWh/yr)"].values
ax.bar(methods, final_aeps, color=colors, alpha=0.8)
ax.set_ylabel("Final AEP (GWh/yr)")
ax.set_title("Final Optimized AEP (Lower = Better)")
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(final_aeps):
    ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("results/comparison_results.png", dpi=150, bbox_inches="tight")
print("✓ Saved: results/comparison_results.png")

# ====================== SUMMARY STATISTICS ======================
print("\n[STEP 6] Summary Statistics\n")

best_improvement_idx = np.nanargmax(improvements)
best_improvement = improvements[best_improvement_idx]
best_method = methods[best_improvement_idx]

print(f"Best improvement: {best_method} with {best_improvement:.2f}%")
print()

# Check if layouts are consistent
print("Checking layout reproducibility...\n")

layout_files = {
    "Hybrid GA": "results/hybrid_best_layout.npy",
    "TOPFARM GA": "results/topfarm_ga_layout.npy",
    "TOPFARM Baseline": "results/topfarm_rs_layout.npy",
}

layouts = {}
for name, filename in layout_files.items():
    if os.path.exists(filename):
        layout = np.load(filename)
        layouts[name] = layout
        print(f"✓ {name}: {layout.shape[0]} turbines, {layout.shape[1]} coordinates")

print("\n" + "=" * 70)
print("READY FOR COMPARISON")
print("=" * 70)
print(f"""
All results compiled. Use these files for your analysis:

CSV Results:    results/comparison_results.csv
Visualization:  results/comparison_results.png

Key Findings:
  - Best improvement: {best_method} ({best_improvement:.2f}%)
  
Next Steps:
  1. Review results/comparison_results.csv
  2. View results/comparison_results.png
  3. Analyze layout files (*.npy) with separate visualizer
  4. Validate that initial layouts match (seed=42)
""")

plt.show()
