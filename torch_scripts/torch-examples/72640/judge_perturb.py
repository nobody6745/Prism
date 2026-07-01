import numpy as np
import os
import time

def check_batch_consistency_in_range():
    script_start = time.perf_counter()
    file_clean = "linear_batch_diff.npz"      
    file_perturb = "linear_batch_perturb.npz" 

    if not os.path.exists(file_clean) or not os.path.exists(file_perturb):
        print("❌ Error: Files missing. Please re-run the previous two scripts.")
        return

    data_clean = np.load(file_clean)
    data_perturb = np.load(file_perturb)

    b1_clean = np.squeeze(data_clean['out_batch1']) 
    b2_clean = np.squeeze(data_clean['out_batch2_slice'])
    b1_perturb = np.squeeze(data_perturb['output'])

    bound_a = b1_perturb
    bound_b = 2 * b1_clean - b1_perturb
        epsilon = 1e-15
    interval_min = np.minimum(bound_a, bound_b) - epsilon
    interval_max = np.maximum(bound_a, bound_b) + epsilon

    widths = interval_max - interval_min
    avg_interval_width = np.mean(widths)
    max_interval_width = np.max(widths)
    is_inside = (b2_clean >= interval_min) & (b2_clean <= interval_max)
    pass_rate = (np.sum(is_inside) / b2_clean.size) * 100.0

    print(f"\n{'='*60}")
    print(f" Final Diagnostic Report")
    print(f"{'='*60}")
    print(f"Pass Rate: {pass_rate:.2f}%")
    print(f"Average Interval Width (Avg Width): {avg_interval_width:.6e}")
    print(f"Maximum Interval Width (Max Width): {max_interval_width:.6e}")
    print(f"{'-'*60}")
    
    script_end = time.perf_counter()
    print(f"Total Execution Time: {script_end - script_start:.6f} s")

if __name__ == "__main__":
    check_batch_consistency_in_range()