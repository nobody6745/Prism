import torch
import os
import time

# Directory and File Configuration
SAVE_DIR = "./output_results"
BASE_FILE = os.path.join(SAVE_DIR, "linear_slice_data.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit(1)

script_start = time.perf_counter()

check_file_exists(BASE_FILE)
check_file_exists(PERTURB_FILE)

# Loading results from previous experiments
data_base = torch.load(BASE_FILE)
data_perturb = torch.load(PERTURB_FILE)

# Target (Direct Sliced Calc), Center (Full Calc -> Sliced), and Perturbed Baseline
# Cast to float32 for stable interval judgment
val_target = data_base['outputs']['out2_direct'].cpu().float()
val_center = data_base['outputs']['out1_slice'].cpu().float()
val_perturb = data_perturb['outputs']['out1_perturb'].cpu().float()

def verify_range(target, center, perturb):
    print(f"\nNumerical Consistency Judge Report")
    
    # 1. Calculate boundaries for the safety envelope
    bound_a = perturb
    bound_b = 2 * center - perturb

    # 2. Determine initial interval [min, max]
    raw_lower = torch.min(bound_a, bound_b)
    raw_upper = torch.max(bound_a, bound_b)

    # 3. Apply 1 ULP (Unit in the Last Place) relaxation for bit-level tolerance
    lower_bound = torch.nextafter(raw_lower, torch.full_like(raw_lower, float('-inf')))
    upper_bound = torch.nextafter(raw_upper, torch.full_like(raw_upper, float('inf')))
    
    # Calculate interval statistics
    widths = (upper_bound - lower_bound).abs()
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    # 4. Inclusion Check
    is_inside = (target >= lower_bound) & (target <= upper_bound)
    total_elements = target.numel()
    passed_elements = is_inside.sum().item()
    pass_rate = (passed_elements / total_elements) * 100
    
    print(f"Statistical Summary:")
    print(f"   Avg Interval Width: {avg_width:.6e}")
    print(f"   Max Interval Width: {max_width:.6e}")
    print(f"   Total Elements:     {total_elements}")
    print(f"   Pass Rate:          {pass_rate:.4f}%")
    
    if passed_elements < total_elements:
        # Calculate the magnitude of numerical drift for out-of-bounds elements
        diff_lower = torch.relu(lower_bound - target)
        diff_upper = torch.relu(target - upper_bound)
        max_violation = torch.max(diff_lower + diff_upper).item()
        print(f"   Max Violation Magnitude: {max_violation:.6e}")

    return passed_elements == total_elements

# Execute verification
passed = verify_range(val_target, val_center, val_perturb)

script_end = time.perf_counter()
print(f"\nTotal Analysis Duration: {script_end - script_start:.6f} seconds")

if passed:
    print(" ALL PASS: Results are within the predefined ULP tolerance envelope.")
else:
    print(" WARNING: Verification failed. Path deviation exceeds the ULP perturbation range.")