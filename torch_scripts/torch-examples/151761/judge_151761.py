import torch
import os
import time

# Directory and File Configuration
SAVE_DIR = "./output_results"
BASE_FILE = os.path.join(SAVE_DIR, "dot_sum_thread_diff.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    if not os.path.exists(path):
        print(f" Error: File not found: {path}")
        exit(1)

script_start = time.perf_counter()

check_file_exists(BASE_FILE)
check_file_exists(PERTURB_FILE)

print(f" Loading data for analysis...")
data_base = torch.load(BASE_FILE)
data_perturb = torch.load(PERTURB_FILE)

def verify_metric(metric_name):
    print(f"\n{'='*25} Verifying Metric: {metric_name} {'='*25}")
    
    try:
        # Target (Multi-threaded execution), Center (Single-threaded baseline), and Perturbed Baseline
        val_target = data_base[metric_name]['thread_10'].cpu()
        val_center = data_base[metric_name]['thread_1'].cpu()
        val_perturb = data_perturb[metric_name]['thread_1_perturb'].cpu()
    except KeyError as e:
        print(f" Error: Missing data. KeyError: {e}")
        return

    # 1. Define the Numerical Safety Interval boundaries
    bound_a = val_perturb
    bound_b = 2 * val_center - val_perturb
    
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
    is_inside = (val_target >= lower_bound) & (val_target <= upper_bound)
    
    print(f"  Avg Interval Width: {avg_width:.6e}")
    print(f"  Max Interval Width: {max_width:.6e}")
    print(f"  Judgment Result:    {' PASS' if is_inside else ' FAIL'}")
    
    if not is_inside:
        # Calculate the magnitude of numerical drift for out-of-bounds elements
        violation = torch.max(torch.relu(lower_bound - val_target), torch.relu(val_target - upper_bound)).item()
        print(f"  Max Violation Magnitude: {violation:.6e}")

# Execute verification for Summation and Dot Product
verify_metric("sum")
verify_metric("dot")

script_total = time.perf_counter() - script_start
print(f"\n⏱ Total Analysis Duration: {script_total:.6f} seconds")