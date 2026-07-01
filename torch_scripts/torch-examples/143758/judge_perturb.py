import torch
import os
import time

def judge():
    script_start = time.perf_counter()

    file_orig = "linear_vs_manual.pt"
    file_pert = "linear_vs_manual_perturb.pt"

    if not os.path.exists(file_orig) or not os.path.exists(file_pert):
        print(f" Error: Both {file_orig} and {file_pert} must exist.")
        return

    print(f"Loading data files...")
    data_orig = torch.load(file_orig)
    data_pert = torch.load(file_pert)
    try:
        # val_center: Baseline from Linear Layer
        # val_target: Result from Manual Implementation
        val_center = data_orig["out_lin_layer"]
        val_target = data_orig["out_manual"]
        
        # Handle potential variations in key naming
        if "out_lin_layer_perturb" in data_pert:
            val_perturb = data_pert["out_lin_layer_perturb"]
        elif "out_lin_layer_perturb." in data_pert:
            val_perturb = data_pert["out_lin_layer_perturb."]
        else:
            val_perturb = data_pert["out_lin_layer"]
    except KeyError as e:
        print(f" Error: Required key missing in dictionary: {e}")
        return

    # 4. Shape Alignment Check
    if val_center.shape != val_perturb.shape:
        print(" Shape mismatch, cannot perform comparison.")
        return

    # 5. Construct Decision Interval [P, 2C - P]
    # This creates a safety envelope centered around the baseline (C)
    bound_1 = val_perturb
    bound_2 = 2 * val_center - val_perturb
    lower_bound = torch.minimum(bound_1, bound_2)
    upper_bound = torch.maximum(bound_1, bound_2)

    widths = (upper_bound - lower_bound).abs()
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    # 6. Numerical Inclusion Check
    is_inside = (val_target >= lower_bound) & (val_target <= upper_bound)
    total_elements = val_target.numel()
    passed_elements = is_inside.sum().item()
    pass_rate = passed_elements / total_elements * 100

    # Calculate absolute deviations
    diff_manual = (val_target - val_center).abs()
    diff_perturb = (val_perturb - val_center).abs()

    print("\n" + "="*65)
    print("        LINEAR LAYER INTERVAL JUDGE REPORT")
    print("="*65)
    
    print(f" Tolerance Band Statistics:")
    print(f"   Avg Interval Width: {avg_width:.6e}")
    print(f"   Max Interval Width: {max_width:.6e}")
    print("-" * 65)
    
    print(f" Judgment Results:")
    print(f"   Total Elements:     {total_elements}")
    print(f"   Pass Rate:          {pass_rate:.2f}%")
    print("-" * 65)
    
    print(f" Error Magnitude Comparison:")
    print(f"   Manual Path Avg Deviation:   {diff_manual.mean().item():.10e}")
    print(f"   Layer Perturbation Avg Dev:  {diff_perturb.mean().item():.10e}")

    # Analysis for failed cases
    if pass_rate < 100:
        print("\n[Analysis & Suggestions]")
        if avg_width < 1e-12:
            print("Note: Interval width is extremely small, indicating almost no random noise.")
            print("Minor deviations (~10^-7) caused by op-order differences in the manual path")
            print(" will easily exceed this tight interval.")
        elif pass_rate > 0:
            print("Partial pass: The error distribution of the manual implementation overlaps")
            print(" with the system's numerical perturbation.")

    print(f"\n Analysis Duration: {time.perf_counter() - script_start:.4f} s")
    print("="*65)

if __name__ == "__main__":
    judge()