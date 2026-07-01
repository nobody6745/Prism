import torch
import os
import time  

def check_interval():

    script_start = time.perf_counter()

    file1 = "distributive_law_fp16.pt"
    file2 = "distributive_perturb.pt"

    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Error: Data files not found. Please run the generation scripts first.")
        return

    print(f"Loading {file1} and {file2} ...")
    data1 = torch.load(file1)
    data2 = torch.2(file2)

    # 1. Extract tensors and cast to float32
    X = data1["result_AB_plus_AC"].float()
    Y = data1["result_A_BplusC"].float()
    P = data2["result_AB_plus_AC_perturb"].float()

    print(f"Data Shape: {X.shape}")
    
    # 2. Calculate interval boundaries
    bound_1 = P
    bound_2 = 2 * X - P

    lower_bound = torch.minimum(bound_1, bound_2)
    upper_bound = torch.maximum(bound_1, bound_2)
    widths = upper_bound - lower_bound
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    # 3. Interval membership check
    inside_mask = (Y >= lower_bound) & (Y <= upper_bound)
    num_inside = inside_mask.sum().item()
    total_elements = Y.numel()
    ratio = num_inside / total_elements * 100

    print("\n" + "="*50)
    print(" Interval Judge Report")
    print("="*50)
    print(f"Average Interval Width: {avg_width:.6e}")
    print(f"Maximum Interval Width: {max_width:.6e}")
    print(f"{'-'*50}")
    print(f"Decision Formula: Y ∈ [min(P, 2X-P), max(P, 2X-P)]")
    print(f"Total Elements: {total_elements}")
    print(f"Passed Elements: {num_inside}")
    print(f"Pass Rate: {ratio:.4f}%")

    # 4. Error Analysis
    if ratio < 100:
        dist_YX = (Y - X).abs()
        dist_PX = (P - X).abs()
        avg_dist_YX = dist_YX.mean().item()
        avg_dist_PX = dist_PX.mean().item()
        
        print(f"\n[Analysis] Deviation Comparison:")
        print(f"  - Avg deviation due to Distributive Law (YX): {avg_dist_YX:.6e}")
        print(f"  - Avg deviation due to System Perturbation (PX): {avg_dist_PX:.6e}")
        
        if avg_dist_YX > avg_dist_PX * 10:
            print("\nConclusion: Failed. Drift from distributive law failure far exceeds system noise.")

    # 5. Timing Statistics
    script_duration = time.perf_counter() - script_start
    print(f"{'-'*50}")
    print(f"Total Execution Time: {script_duration:.4f} seconds")

if __name__ == "__main__":
    check_interval()