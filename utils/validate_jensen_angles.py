"""
VALIDATION: Jensen angle conversion in final.py
================================================

The issue: Line 146 in final.py uses:
  wind_to_dir = 1.5 * π - t_thetas
  
This should convert meteorological bearing (0°=North, cw) 
to mathematical angle (0°=East, ccw).

Let's verify this is correct.
"""

import numpy as np
import torch

print("=" * 70)
print("JENSEN ANGLE CONVERSION VALIDATION")
print("=" * 70)

# Meteorological bearings (from input)
WIND_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float64)

print("\nInput wind directions (meteorological bearings):")
print(f"  0°   = North wind (wind FROM North)")
print(f"  90°  = East wind (wind FROM East)")
print(f"  180° = South wind (wind FROM South)")
print(f"  270° = West wind (wind FROM West)")

# Test case: Wind from North (0°)
# In wind farm context: Wind blows FROM North TO South
# In math angle system: This is 180° (South direction) or π radians
print("\n" + "-" * 70)
print("TEST 1: Wind from North (bearing 0°)")
print("-" * 70)

theta_met = 0.0  # Bearing from North
theta_rad = np.deg2rad(theta_met)

# Current formula
wind_to_dir_current = 1.5 * np.pi - theta_rad
wind_to_dir_current_deg = np.rad2deg(wind_to_dir_current) % 360

print(f"  Meteorological bearing: {theta_met}°")
print(f"  Wind blows: North → South")
print(f"  Expected math angle: 180° (South direction)")
print(f"  Current formula result: {wind_to_dir_current_deg:.1f}°")
print(f"  MATCH: {'✓ YES' if abs(wind_to_dir_current_deg - 180.0) < 1 else '✗ NO'}")

# Test case 2: Wind from East (90°)
# Wind blows FROM East TO West
# In math: This is 180° (or π)... NO wait, that's also South
# Actually, in math angle system:
#   0° = East, 90° = North, 180° = West, 270° = South
# So wind FROM East going West is 180° math angle
print("\n" + "-" * 70)
print("TEST 2: Wind from East (bearing 90°)")
print("-" * 70)

theta_met = 90.0  # Bearing from East
theta_rad = np.deg2rad(theta_met)

wind_to_dir_current = 1.5 * np.pi - theta_rad
wind_to_dir_current_deg = np.rad2deg(wind_to_dir_current) % 360

print(f"  Meteorological bearing: {theta_met}°")
print(f"  Wind blows: East → West")
print(f"  Expected math angle: 180° (West direction in math = π)")
print(f"  Current formula result: {wind_to_dir_current_deg:.1f}°")
print(f"  MATCH: {'✓ YES' if abs(wind_to_dir_current_deg - 180.0) < 1 else '✗ NO'}")

# Test case 3: Wind from South (180°)
# Wind blows FROM South TO North
# In math angle: 90° (North)
print("\n" + "-" * 70)
print("TEST 3: Wind from South (bearing 180°)")
print("-" * 70)

theta_met = 180.0  # Bearing from South
theta_rad = np.deg2rad(theta_met)

wind_to_dir_current = 1.5 * np.pi - theta_rad
wind_to_dir_current_deg = np.rad2deg(wind_to_dir_current) % 360

print(f"  Meteorological bearing: {theta_met}°")
print(f"  Wind blows: South → North")
print(f"  Expected math angle: 90° (North direction in math = π/2)")
print(f"  Current formula result: {wind_to_dir_current_deg:.1f}°")
print(f"  MATCH: {'✓ YES' if abs(wind_to_dir_current_deg - 90.0) < 1 else '✗ NO'}")

# Test case 4: Wind from West (270°)
# Wind blows FROM West TO East
# In math angle: 0° (East)
print("\n" + "-" * 70)
print("TEST 4: Wind from West (bearing 270°)")
print("-" * 70)

theta_met = 270.0  # Bearing from West
theta_rad = np.deg2rad(theta_met)

wind_to_dir_current = 1.5 * np.pi - theta_rad
wind_to_dir_current_deg = np.rad2deg(wind_to_dir_current) % 360

print(f"  Meteorological bearing: {theta_met}°")
print(f"  Wind blows: West → East")
print(f"  Expected math angle: 0° (East direction in math)")
print(f"  Current formula result: {wind_to_dir_current_deg:.1f}°")
print(f"  MATCH: {'✓ YES' if abs(wind_to_dir_current_deg - 0.0) < 1 or abs(wind_to_dir_current_deg - 360.0) < 1 else '✗ NO'}")

# Summary
print("\n" + "=" * 70)
print("GEOMETRIC ANALYSIS")
print("=" * 70)

t_thetas = torch.deg2rad(torch.tensor(WIND_DIRECTIONS_DEG, dtype=torch.float64))
wind_to_dir = 1.5 * torch.pi - t_thetas
wind_to_dir_deg = torch.rad2deg(wind_to_dir) % 360

print("\nAll wind directions (current formula):")
print("Met Bearing | Math Angle | Wind Blows")
print("-" * 40)
for met_deg, math_deg in zip(WIND_DIRECTIONS_DEG, wind_to_dir_deg.numpy()):
    print(f"  {met_deg:5.0f}°    | {math_deg:5.1f}°    | ", end="")
    if abs(math_deg - 0) < 1 or abs(math_deg - 360) < 1:
        print("West → East")
    elif abs(math_deg - 90) < 1:
        print("South → North")
    elif abs(math_deg - 180) < 1:
        print("North → South or East → West")
    elif abs(math_deg - 270) < 1:
        print("North → South")
    else:
        print(f"({math_deg:.0f}° in math)")

print("\n" + "=" * 70)
print("FORMULA VALIDATION")
print("=" * 70)

# The correct formula should be:
# Meteorological bearing θ_met: 0°=N, 90°=E, 180°=S, 270°=W (clockwise from North)
# Math angle θ_math: 0°=E, 90°=N, 180°=W, 270°=S (counter-clockwise from East)
# 
# Relationship: θ_math = 90° - θ_met (mod 360°)
# Proof:
#   θ_met = 0° (North) → θ_math = 90° (North in math) ✓
#   θ_met = 90° (East) → θ_math = 0° (East in math) ✓
#   θ_met = 180° (South) → θ_math = -90° = 270° (South in math) ✗ Should be 270°
#   θ_met = 270° (West) → θ_math = -180° = 180° (West in math) ✓

# Better formula: θ_math = π/2 - θ_met (in radians)
# This gives upwind direction. For downwind direction in Jensen, we need π/2 - θ_met + π = 3π/2 - θ_met

print("\nCurrent formula: θ_wind = 3π/2 - θ_met")
print("This gives DOWNWIND direction in math angles")
print("\nVerification by case:")
print("  θ_met=0° (N) → θ_wind = 270° = South in math = Downwind South ✓")
print("  θ_met=90° (E) → θ_wind = 0° = East in math... wait, that's wrong!")
print("    If wind FROM East, downwind is West = 180° in math, not 0°")
print("\nFormula appears to have an error!")

print("\n" + "=" * 70)
print("CORRECT FORMULA")
print("=" * 70)

# For meteorological wind bearing θ_met (0=N, 90=E, 180=S, 270=W):
# The downwind direction in math angles should be:
# θ_downwind_math = θ_met + 90° (in meteorological frame, this converts bearing to downwind azimuth)

# Actually, let's think differently:
# In wind farm optimization, we care about the wind direction vector
# If wind bearing is θ_met (wind comes FROM this direction):
#   - Wind vector points at angle (θ_met + 180°) in met frame
#   - This needs conversion to math frame where 0°=East, 90°=North (ccw)
#   - Math bearing = 90° - meteorological bearing
#   - So wind vector in math = (θ_met + 180°) in met → 90° - (θ_met + 180°) = -90° - θ_met = 270° - θ_met

# Hmm, let me verify with actual cosine/sine logic
print("\nUsing cosine/sine decomposition:")
print("If bearing is θ_met (met convention): cos(θ), sin(θ) in met frame")
print("In math frame: cos(θ_math) = sin(θ_met), sin(θ_math) = cos(θ_met)")
print("This gives: θ_math = 90° - θ_met OR θ_math = π/2 - θ_met (rad)")

print("\nBut we need DOWNWIND, so add 180°:")
print("θ_downwind_math = π/2 - θ_met + π = 3π/2 - θ_met")
print("This IS the current formula! So it seems CORRECT")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The formula wind_to_dir = 3π/2 - θ_met appears mathematically correct.

However, to be absolutely sure, we should:
1. Check against a reference Jensen implementation
2. Verify with a known test case (e.g., 2 turbines, compute deficit)
3. Compare against PyWake's implementation (Bastankhah model)

Recommendation: KEEP current formula, but add diagnostic output to 
verify wind direction vectors are sensible.
""")
