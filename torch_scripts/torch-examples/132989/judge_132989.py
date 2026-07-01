import torch
import os
import time  

# Configuration and File Paths
SAVE_DIR = "./output_results"
MAIN_FILE = os.path.join(SAVE_DIR, "reproduce_issue_data.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit(1)

script_start = time.perf_counter()

check_file_exists(MAIN_FILE)
check_file_exists(PERTURB_FILE)

# Loading results from previous experiments
data_main = torch.load(MAIN_FILE)
data_perturb = torch.load(PERTURB_FILE)

# Target (Matmul result), Center (Loop baseline), and Perturbed reference
val_target = data_main['outputs']['matmul_result'].cpu()
val_center = data_main['outputs']['loop_result'].cpu()
val_perturb = data_perturb['outputs']['loop_result_perturb'].cpu()

def verify_range(target, center, perturb):
    print(f"\n{'='*20} Starting Verification {'='*20}")

    # 1. Define the safety interval boundaries
    bound_a = perturb
    bound_b = 2 * center - perturb
    
    # 2. Determine initial [min, max] interval
    raw_lower = torch.min(bound_a, bound_b)
    raw_upper = torch.max(bound_a, bound_b)
    
    # 3. Apply ULP (Unit in the Last Place) relaxation
    # Lower bound expanded by 1 ULP
    lower_bound = torch.nextafter(raw_lower, torch.full_like(raw_lower, float('-inf')))
    
    # Upper bound expanded by 3 ULPs (based on sequential nextafter calls)
    upper_bound = torch.nextafter(raw_upper, torch.full_like(raw_upper, float('inf')))
    upper_bound = torch.nextafter(upper_bound, torch.full_like(upper_bound, float('inf')))
    upper_bound = torch.nextafter(upper_bound, torch.full_like(upper_bound, float('inf')))
    
    # Interval Statistics
    widths = upper_bound - lower_bound
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    # 4. Numerical Inclusion Check
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

# Execute Numerical Consistency Check
passed = verify_range(val_target, val_center, val_perturb)

script_end = time.perf_counter()
print(f"Total Analysis Duration: {script_end - script_start:.6f} seconds")

if passed:
    print("ALL PASS: Both implementations are consistent within the specified ULP tolerance.")
else:
    print("WARNING: Verification failed. Numerical drift between paths exceeds ULP tolerance.")