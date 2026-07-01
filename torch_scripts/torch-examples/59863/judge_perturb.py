import numpy as np
import os
import time

def evaluate_summation_consistency():
    start_time = time.perf_counter()

    file_clean = "sum_thread_comparison.npz"  
    file_perturb = "sum_thread_perturb.npz"   

    print(f" Loading {file_clean} and {file_perturb}...")

    if not os.path.exists(file_clean) or not os.path.exists(file_perturb):
        print(" Error: Files missing. Please run the generation script first.")
        return

    try:
        data_clean = np.load(file_clean)
        data_perturb = np.load(file_perturb)

        s1_clean = float(data_clean['val_1thread'])
        s2_clean = float(data_clean['val_2threads'])
        s1_perturb = float(data_perturb['val_1thread'])

        print("Data loaded successfully.")

    except Exception as e:
        print(f"Failed to parse data: {e}")
        return

    bound_a = s1_perturb
    bound_b = 2 * s1_clean - s1_perturb
    
    interval_min = min(bound_a, bound_b)
    interval_max = max(bound_a, bound_b)
    
    width = interval_max - interval_min
    max_interval_width = np.max(width)
    avg_interval_width = np.mean(width)
    

    epsilon = 1e-15
    is_inside = (s2_clean >= interval_min - epsilon) and (s2_clean <= interval_max + epsilon)

    thread_diff = abs(s1_clean - s2_clean)
    
    print(f"\n{'='*60}")
    print(f" Summation Consistency Range Check")
    print(f"{'='*60}")
    
    print(f"[Numerical Values]")
    print(f"  Center (S1 Clean):    {s1_clean:.6f}")
    print(f"  Target (S2 Clean):    {s2_clean:.6f}")
    
    print(f"\n[Interval Width Statistics]")
    print(f"  Average Interval Width: {avg_interval_width:.6e}")
    print(f"  Maximum Interval Width: {max_interval_width:.6e}")
    print(f"  Current Thread Diff:    {thread_diff:.6e}")

    print(f"{'-'*60}")
    print(f"Decision Result: {'WITHIN RANGE' if is_inside else 'OUT OF RANGE'}")
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    
    print(f"\n[Execution Information]")
    print(f"  Total Execution Time: {total_duration:.6f} s")
    
    print(f"\n Conclusion:")
    if is_inside:
        # Categorized as Type-I: Benign error [cite: 129, 609]
        print("   Multi-thread deviation falls within the perturbation interval; numerical consistency achieved.")
    else:
        # Potential implementation discrepancy or ineffective perturbation [cite: 393, 1371]
        if avg_interval_width < 1e-12:
             print("   Failure: Interval width is negligible, indicating the perturbation was ineffective.")
        else:
             print("   Failure: Discrepancy from Parallel Reduction exceeded the perturbation envelope.")

if __name__ == "__main__":
    evaluate_summation_consistency()