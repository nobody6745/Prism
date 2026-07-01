import torch
import os
import time  

# Configuration
SAVE_DIR = "./output_results"
BATCH_FILE = os.path.join(SAVE_DIR, "linear_batch_vs_loop.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit(1)

script_start = time.perf_counter()

check_file_exists(BATCH_FILE)
check_file_exists(PERTURB_FILE)

print(f"Loading data...")
data_batch = torch.load(BATCH_FILE)
data_perturb = torch.load(PERTURB_FILE)

try:
    # Target value (Loop/GEMV baseline), Center (Batch/GEMM), and Perturbed value
    val_target = data_batch['outputs']['output_loop'].cpu()
    val_center = data_batch['outputs']['output_batch'].cpu()
    val_perturb = data_perturb['outputs']['output_batch_perturb'].cpu()
except KeyError as e:
    print(f"Error: Missing key: {e}")
    exit(1)

def verify_range(target, center, perturb):
    print(f"\n{'='*20} Starting Verification {'='*20}")
    
    # 1. Calculate the two potential boundaries
    bound_a = perturb
    bound_b = 2 * center - perturb
    
    # 2. Determine initial interval [min, max]
    raw_lower = torch.min(bound_a, bound_b)
    raw_upper = torch.max(bound_a, bound_b)
    
    # Expand boundaries by 1 ULP (Bit-level relaxation)
    lower_bound = torch.nextafter(raw_lower, torch.full_like(raw_lower, float('-inf')))
    upper_bound = torch.nextafter(raw_upper, torch.full_like(raw_upper, float('inf')))
    
    widths = upper_bound - lower_bound
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()

    # Inclusion check
    is_inside = (target >= lower_bound) & (target <= upper_bound)
    total_elements = target.numel()
    passed_elements = is_inside.sum().item()
    pass_rate = (passed_elements / total_elements) * 100
    
    print(f" Statistical Summary:")
    print(f"   Avg Interval Width: {avg_width:.6e}")
    print(f"   Max Interval Width: {max_width:.6e}")
    print(f"   Total Elements:     {total_elements}")
    print(f"   Pass Rate:          {pass_rate:.4f}%")
    
    if passed_elements < total_elements:
        # Calculate deviation for elements outside the safe envelope
        diff_lower = torch.relu(lower_bound - target)
        diff_upper = torch.relu(target - upper_bound)
        max_violation = torch.max(diff_lower + diff_upper).item()
        print(f"   Max Violation Magnitude: {max_violation:.6e}")

    return passed_elements == total_elements

# Execute verification
passed = verify_range(val_target, val_center, val_perturb)

script_end = time.perf_counter()
print("\n" + "="*50)
print(f" Total Script Execution Time: {script_end - script_start:.4f} seconds")

if passed:
    print(" ALL PASS: Result consistency is within the numerical safety envelope.")
else:
    print(" WARNING: Out-of-bounds values detected. This usually indicates significant \n algorithmic differences between GEMV and GEMM implementations.")

print("="*50)