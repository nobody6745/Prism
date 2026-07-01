import torch
import os
import time 

script_start = time.perf_counter()

# Load previous experiment results
try:
    res1 = torch.load("precision_test_results.pt", map_location="cpu")
    res2 = torch.load("perturbation_results.pt", map_location="cpu")
except FileNotFoundError:
    print("Error: .pt files not found. Ensure they are in the same directory as this script.")
    exit()

eager_output = res1["eager_output"]       
compiled_output = res1["compiled_output"] 
perturbed_output = res2["perturbed_output"] 

# 1. Define the Numerical Safety Interval
# The interval is centered around Eager, using Perturbation (P) as one boundary.
lower_bound = perturbed_output
upper_bound = 2 * eager_output - perturbed_output

# 2. Calculate Interval Statistics
interval_width = (upper_bound - lower_bound).abs()
max_interval = interval_width.max().item()
avg_interval = interval_width.mean().item()

# 3. Final Numerical Judgment
is_in_range = (compiled_output >= lower_bound) and (compiled_output <= upper_bound)

print("="*65)
print("Numerical Judge Report: Compiled Result Consistency")
print("="*65)
print(f"Eager Baseline (E)    : {eager_output.item():.10f}")
print(f"Lower Bound (P)       : {lower_bound.item():.10f}")
print(f"Upper Bound (2E - P)  : {upper_bound.item():.10f}")
print("-" * 65)
print(f"Avg Interval Width    : {avg_interval:.10e}")
print(f"Max Interval Width    : {max_interval:.10e}")
print("-" * 65)
print(f"Triton Compiled (C)   : {compiled_output.item():.10f}")
print("="*65)

if is_in_range:
    print(" JUDGMENT: PASS (Compiled result within 1 ULP perturbation range)")
else:
    print(" JUDGMENT: FAILED (Compiled result outside the perturbation interval)")
    if torch.isinf(compiled_output):
        print(" DIAGNOSIS: Result is 'inf'. Triton fused kernel likely caused catastrophic overflow.")

script_end = time.perf_counter()
print(f"\n Total Analysis Duration: {script_end - script_start:.6f} seconds")