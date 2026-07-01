import torch
import os
import time

def judge():
    script_start = time.perf_counter()

    # Define file paths for the error snapshot and the perturbation baseline
    file_error = "block_diag_error.pt"  
    file_perturb = "perturb.pt"         
    
    if not os.path.exists(file_error) or not os.path.exists(file_perturb):
        print(f" Error: Both {file_error} and {file_perturb} must exist.")
        return

    print(f" Loading data for comparative analysis...")
    data_error = torch.load(file_error)
    data_perturb = torch.load(file_perturb)

    try:
        # Extract mean values and ensure they are in float32 for consistent comparison
        mean_s = data_error["mean_s"].float()
        mean_s_block = data_error["mean_s_block"].float()
        mean_s_perturb = (data_perturb.get("mean_s_perturb") or data_perturb.get("mean_s")).float()
    except Exception as e:
        print(f" Error: Failed to extract data: {e}")
        return

    # Construct the decision interval [P, 2C - P]
    # This creates a safety envelope centered around the baseline (mean_s)
    bound_1 = mean_s_perturb
    bound_2 = 2 * mean_s - mean_s_perturb

    lower_bound = torch.minimum(bound_1, bound_2)
    upper_bound = torch.maximum(bound_1, bound_2)

    # Calculate interval statistics
    widths = (upper_bound - lower_bound).abs()
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()

    # Numerical inclusion check
    is_inside = (mean_s_block >= lower_bound) and (mean_s_block <= upper_bound)

    print("      BLOCK_DIAG NUMERICAL INTERVAL REPORT  ⚖️")
    print(f" Tolerance Bandwidth Statistics:")
    print(f"   Avg Interval Width: {avg_width:.6e}")
    print(f"   Max Interval Width: {max_width:.6e}")
    
    print(f"\n Key Numerical Comparison:")
    print(f"   Center (C):  {mean_s.item():.15f}")
    print(f"   Target (T):  {mean_s_block.item():.15f}")
    print(f"   Perturb (P): {mean_s_perturb.item():.15f}")

    print(f" Final Judgment:")
    if is_inside:
        print(" PASS: Result is within the numerical safety envelope.")
    else:
        print(" FAIL: Result is outside the safety envelope!")
        # Calculate the magnitude of the violation
        violation = (mean_s_block - upper_bound if mean_s_block > upper_bound else lower_bound - mean_s_block).item()
        print(f"   Violation Magnitude: {violation:.6e}")

    print(f"\n Analysis Duration: {time.perf_counter() - script_start:.4f} seconds")

if __name__ == "__main__":
    judge()