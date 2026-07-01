import numpy as np
import os
import time 

def check_y1_inclusion(d):
    filename = f"perturb__d{d}.npz"
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    try:
        data = np.load(filename)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return

    y1_target = data.get('y1_old', data.get('y1'))
    y2_center = data.get('y2_old', data.get('y2'))
    y2_perturb = data.get('y2_perturb')

    if y1_target is None or y2_center is None or y2_perturb is None:
        print(f"Missing required keys in {filename}")
        return
    bound_a = y2_perturb
    bound_b = 2 * y2_center - y2_perturb
    
    interval_min = np.minimum(bound_a, bound_b)
    interval_max = np.maximum(bound_a, bound_b)

    interval_widths = interval_max - interval_min
    avg_width = np.mean(interval_widths)
    max_width = np.max(interval_widths)

    is_inside = (y1_target >= interval_min) & (y1_target <= interval_max)

    total = y1_target.size
    count_pass = np.sum(is_inside)
    pass_rate = (count_pass / total) * 100.0

    print(f"[{'d=' + str(d):<5}] Analysis Report:")
    print(f"  Interval Width (Avg): {avg_width:.6e}")
    print(f"  Interval Width (Max): {max_width:.6e}")
    print(f"  Total Elements:        {total}")
    print(f"  Inside Range:          {count_pass}")
    print(f"  Pass Rate:           {pass_rate:.2f}%")

    idx = 0
    print(f"  [Sample at index {idx}]")
    print(f"    Low Bound:  {interval_min[idx]:.10f}")
    print(f"    Target:     {y1_target[idx]:.10f}")
    print(f"    High Bound: {interval_max[idx]:.10f}")
    
    if pass_rate > 99.0:
        print("  PERFECT: Numerical drift is covered by the symmetric perturbation range.")
    else:
        print("  WARN: Some elements drift beyond the noise-defined boundary.")
    print("-" * 70)

if __name__ == "__main__":
    script_start = time.perf_counter()

    print(f"{'='*70}")
    print("Judge Script: y1_old vs y2_perturb_range")
    print(f"{'='*70}\n")
    
    check_y1_inclusion(5)
    check_y1_inclusion(6)
    total_time = time.perf_counter() - script_start
    print(f"Total Execution Time: {total_time:.4f} seconds")