import torch
import os
import time 

# Directory and File Configuration
SAVE_DIR = "./output_results"
BASE_FILE = os.path.join(SAVE_DIR, "batch_diff_data.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    if not os.path.exists(path):
        print(f" Error: File not found: {path}")
        exit(1)

script_start = time.perf_counter()

check_file_exists(BASE_FILE)
check_file_exists(PERTURB_FILE)

print(f"📂 Loading data for comparative analysis...")
data_base = torch.load(BASE_FILE)
data_perturb = torch.load(PERTURB_FILE)

try:
    # Target (Results from Batch 2), Center (Baseline from Batch 1), and Perturbed Baseline
    val_target = data_base['outputs']['result_batch_2'].cpu()
    val_center = data_base['outputs']['result_batch_1'].cpu()
    val_perturb = data_perturb['outputs']['result_batch_1_perturb'].cpu()
except KeyError as e:
    print(f" Error: Missing key: {e}")
    exit(1)

def verify_range(target, center, perturb):
    print(f"\n{'='*20} Starting Verification {'='*20}")
    
    # 1. Calculate boundaries for the numerical safety envelope
    bound_a = perturb
    bound_b = 2 * center - perturb
    raw_lower = torch.min(bound_a, bound_b)
    raw_upper = torch.max(bound_a, bound_b)
    
    # 2. Apply 1 ULP (Unit in the Last Place) relaxation for bit-level tolerance
    lower_bound = torch.nextafter(raw_lower, torch.full_like(raw_lower, float('-inf')))
    upper_bound = torch.nextafter(raw_upper, torch.full_like(raw_upper, float('inf')))
    
    # Calculate interval statistics
    widths = upper_bound - lower_bound
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    # 3. Inclusion Check
    is_inside = (target >= lower_bound) & (target <= upper_bound)
    total_elements = target.numel()
    passed_elements = is_inside.sum().item()
    pass_rate = (passed_elements / total_elements) * 100
    
    print(f"  Statistical Summary:")
    print(f"   Avg Interval Width: {avg_width:.6e}")
    print(f"   Max Interval Width: {max_width:.6e}")
    print(f"   Total Elements:     {total_elements}")
    print(f"   Pass Rate:          {pass_rate:.4f}%")
    
    if passed_elements < total_elements:
        # Calculate the magnitude of drift for out-of-bounds elements
        diff_lower = torch.relu(lower_bound - target)
        diff_upper = torch.relu(target - upper_bound)
        max_violation = torch.max(diff_lower + diff_upper).item()
        print(f"   Max Violation Magnitude: {max_violation:.6e}")

    return passed_elements == total_elements

# Execute numerical verification
passed = verify_range(val_target, val_center, val_perturb)

script_end = time.perf_counter()
print("\n" + "="*50)
print(f"  Analysis Duration: {script_end - script_start:.6f} seconds")

if passed:
    print(" ALL PASS: Batch 2 results are within the numerical safety envelope.")
else:
    print(" WARNING: Out-of-bounds values detected. This usually indicates that changing \n"
          " the Batch Size triggered a Kernel switch or a significant shift in accumulation order.")
print("="*50)