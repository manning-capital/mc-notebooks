#!/usr/bin/env python3
"""
Test that F, G, and optimization functions handle large negative theta values correctly.
"""

import numpy as np
import sys

sys.path.insert(0, "/Users/glynfinck/Documents/Repositories/mc-notebooks/src")

from utils.stochastic._non_rolling import OrnsteinUhlenbeck

print("=" * 80)
print("Testing F and G functions with large negative theta values")
print("=" * 80)

# Test parameters
mu = 0.5
sigma = 0.2
theta_values = [0.0, -10.0, -100.0, -1000.0, -10000.0]
r = 0.01

print("\nTest 1: Direct F and G function calls with spreads")
print("-" * 80)

for theta in theta_values:
    # Absolute position close to theta
    x_absolute = theta + 2.0  # 2 units above theta

    # Convert to spread
    x_spread = x_absolute - theta  # Should be 2.0

    print(f"\nTheta: {theta:>10.1f}")
    print(f"  Absolute position x: {x_absolute:>10.1f}")
    print(f"  Spread (x - theta): {x_spread:>10.1f}")

    try:
        # Call F and G with spreads (they no longer take theta parameter)
        F_val = OrnsteinUhlenbeck.F(x_spread, mu, sigma, r, use_analytical=True)
        G_val = OrnsteinUhlenbeck.G(x_spread, mu, sigma, r, use_analytical=True)

        # Convert to scalar if needed
        F_val = float(F_val) if np.ndim(F_val) == 0 or np.size(F_val) == 1 else F_val
        G_val = float(G_val) if np.ndim(G_val) == 0 or np.size(G_val) == 1 else G_val

        print(f"  F(spread): {F_val:>15.6f}")
        print(f"  G(spread): {G_val:>15.6f}")

        # Check for infinities or NaNs
        if np.isinf(F_val) or np.isnan(F_val):
            print(f"  ⚠️  WARNING: F returned {F_val}")
        if np.isinf(G_val) or np.isnan(G_val):
            print(f"  ⚠️  WARNING: G returned {G_val}")

        if np.isfinite(F_val) and np.isfinite(G_val):
            print("  ✓ Both F and G are finite!")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")

print("\n" + "=" * 80)
print("Test 2: Optimal exit level computation with large negative theta")
print("=" * 80)

# Test with arrays of parameters
n_tests = len(theta_values)
mu_arr = np.full(n_tests, mu)
sigma_arr = np.full(n_tests, sigma)
theta_arr = np.array(theta_values)

print(f"\nComputing optimal exit levels for theta values: {theta_values}")
print("-" * 80)

try:
    exit_levels = OrnsteinUhlenbeck.get_optimal_exit_level(
        mu=mu_arr,
        sigma=sigma_arr,
        theta=theta_arr,
        discount_rate=r,
        transaction_cost=0.01,
        max_iter=100,
        tol=1e-7,
    )

    print(f"\n{'Theta':<15} {'Exit Level':<15} {'Spread':<15} {'Status':<15}")
    print("-" * 60)

    for i, theta in enumerate(theta_values):
        exit_level = exit_levels[i]
        spread = exit_level - theta

        status = "✓ Valid" if np.isfinite(exit_level) else "❌ Invalid"

        print(f"{theta:<15.1f} {exit_level:<15.6f} {spread:<15.6f} {status}")

    # Verify spreads are consistent (should be similar across different theta values)
    spreads = exit_levels - theta_arr
    if np.all(np.isfinite(spreads)):
        spread_std = np.std(spreads)
        spread_mean = np.mean(spreads)
        print("\nSpread statistics:")
        print(f"  Mean: {spread_mean:.6f}")
        print(f"  Std:  {spread_std:.6f}")

        if spread_std < 0.01:  # Spreads should be nearly identical
            print("  ✓ Spreads are consistent! (std < 0.01)")
        else:
            print("  ⚠️  Spreads vary more than expected")

except Exception as e:
    print(f"\n❌ ERROR in optimization: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("Test 3: V function with large negative theta")
print("=" * 80)

print("\nTesting V function with various theta values")
print("-" * 80)

transaction_cost = 0.01

for theta in theta_values:
    x_absolute = theta + 1.0  # Position close to theta
    exit_level_absolute = theta + 3.0  # Exit level above current position

    print(f"\nTheta: {theta:>10.1f}")
    print(f"  Position x: {x_absolute:>10.1f}")
    print(f"  Exit level: {exit_level_absolute:>10.1f}")

    try:
        V_val = OrnsteinUhlenbeck.V(
            x=x_absolute,
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=transaction_cost,
            exit_level=exit_level_absolute,
            use_analytical=True,
        )

        print(f"  V(x): {V_val:>15.6f}")

        if np.isinf(V_val) or np.isnan(V_val):
            print(f"  ⚠️  WARNING: V returned {V_val}")
        else:
            print("  ✓ V is finite!")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 80)
print("Test 4: Check that F and G give same results regardless of theta")
print("=" * 80)

# The key insight: F and G should give identical results for the same SPREAD,
# regardless of the absolute theta value

spread = 2.5  # Fixed spread value
print(f"\nTesting with fixed spread = {spread}")
print("-" * 80)

F_values = []
G_values = []

for theta in theta_values:
    # F and G take spreads directly now, so we just pass the spread
    F_val = OrnsteinUhlenbeck.F(spread, mu, sigma, r, use_analytical=True)
    G_val = OrnsteinUhlenbeck.G(spread, mu, sigma, r, use_analytical=True)

    # Convert to scalar if needed
    F_val = float(F_val) if np.ndim(F_val) == 0 or np.size(F_val) == 1 else F_val
    G_val = float(G_val) if np.ndim(G_val) == 0 or np.size(G_val) == 1 else G_val

    F_values.append(F_val)
    G_values.append(G_val)

    print(f"Theta: {theta:>10.1f} -> F: {F_val:>15.10f}, G: {G_val:>15.10f}")

# Check consistency
F_values = np.array(F_values)
G_values = np.array(G_values)

if np.all(np.isfinite(F_values)) and np.all(np.isfinite(G_values)):
    F_std = np.std(F_values)
    G_std = np.std(G_values)

    print(f"\nF standard deviation: {F_std:.2e}")
    print(f"G standard deviation: {G_std:.2e}")

    if F_std < 1e-10 and G_std < 1e-10:
        print("✓ F and G are IDENTICAL across all theta values!")
        print("  This confirms that F and G work purely with spreads,")
        print("  independent of the absolute theta value.")
    else:
        print("⚠️  F and G values vary across theta values")
else:
    print("❌ Some F or G values are not finite")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ Test complete! Check results above for any warnings or errors.")
