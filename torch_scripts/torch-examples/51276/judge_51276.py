import numpy as np
import os

def check_matmul_consistency():

    file_clean = "matmul_comparison.npz"   
    file_perturb = "matmul_perturb.npz"    

    if not os.path.exists(file_clean) or not os.path.exists(file_perturb):
        print(f" Missing files. Please ensure both '{file_clean}' and '{file_perturb}' exist.")
        return

    print(f"Loading {file_clean} and {file_perturb}...")
    data_clean = np.load(file_clean)
    data_perturb = np.load(file_perturb)

    y_clean = data_clean['res1_comp']
    y_target = data_clean['res2_comp']
    y_perturb = data_perturb['res1_perturb']
    
    if y_perturb.shape != y_clean.shape:
        y_perturb = y_perturb.reshape(y_clean.shape)


    bound_a = y_perturb
    bound_b = 2 * y_clean - y_perturb
    
    interval_min = np.minimum(bound_a, bound_b)
    interval_max = np.maximum(bound_a, bound_b)

    interval_widths = interval_max - interval_min
    avg_interval_width = np.mean(interval_widths)
    max_interval_width = np.max(interval_widths)



    is_inside = (y_target >= interval_min) & (y_target <= interval_max)

    total = y_target.size
    pass_count = np.sum(is_inside)
    pass_rate = (pass_count / total) * 100.0
    
    perturb_diff = np.abs(y_clean - y_perturb)  
    batch_diff = np.abs(y_clean - y_target)     
    
    avg_perturb_mag = np.mean(perturb_diff)
    avg_batch_error = np.mean(batch_diff)

    print(f"\n{'='*60}")
    print(f"JUDGE REPORT: MatMul Batch Consistency")
    print(f"{'='*60}")
    print(f"Pass Rate:       {pass_rate:.2f}% ({pass_count}/{total})")
    print(f"{'-'*60}")
    print(f"Interval Analysis (Noise Tolerance Range):")
    print(f"  Average Interval Width: {avg_interval_width:.4e}")
    print(f"  Maximum Interval Width: {max_interval_width:.4e}")
    print(f"{'-'*60}")
    print(f"Error Magnitude Analysis:")
    print(f"  Avg Perturbation (Noise):  {avg_perturb_mag:.4e}")
    print(f"  Avg Batch Error (Actual):  {avg_batch_error:.4e}")
    print(f"  Max Batch Error (Actual):  {np.max(batch_diff):.4e}")

    if avg_perturb_mag < 1e-16:
        print("\n  WARNING: Perturbation range is zero or near machine epsilon.")
    elif avg_batch_error > max_interval_width:
         print("\n FAIL: Batch discrepancy EXCEEDS the maximum noise interval.")
    elif pass_rate > 99.0:
        print("\n PERFECT: Batch 1 and Batch 2 are consistent within noise limits.")
    else:
        print("\n  MIXED: Some drift observed outside the perturbation range.")

    idx = 0 
    print(f"\n[Sample Value at index {idx}]")
    print(f"  Interval: [{interval_min[0, idx]:.18e}, {interval_max[0, idx]:.18e}]")
    print(f"  Width:    {interval_widths[0, idx]:.18e}")
    print(f"  Target:   {y_target[0, idx]:.18e}")
    print(f"  Inside?   {is_inside[0, idx]}")

if __name__ == "__main__":
    check_matmul_consistency()