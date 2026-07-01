import torch
import os
import time  

# Configuration and File Paths
SAVE_DIR = "./output_results"
DIFF_FILE = os.path.join(SAVE_DIR, "digamma_diff.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "digamma_perturb_test.pt")

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit(1)

script_start = time.perf_counter()

check_file_exists(DIFF_FILE)
check_file_exists(PERTURB_FILE)

print(f" Loading comparison data for analysis...")
data_diff = torch.load(DIFF_FILE)
data_perturb = torch.load(PERTURB_FILE)

try:
    # Converting target (TensorFlow), center (PyTorch), and perturbed baseline to float32
    val_target = torch.tensor(data_diff['outputs']['tensorflow'], dtype=torch.float32)
    val_center = torch.tensor(data_diff['outputs']['pytorch'], dtype=torch.float32)
    val_perturb = data_perturb['outputs']['perturbed'].cpu()
except KeyError as e:
    print(f" Error: Missing key: {e}")
    exit(1)

def verify_range(target, center, perturb):
    print(f"\n{'='*20} Numerical Consistency Judge Report {'='*20}")
    print("Formula: TF_Result ∈ [P, 2C - P] (Relaxed by 1 ULP)")
    
    # 1. Calculate the boundaries of the safety envelope
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
    
    print(f" Statistical Summary:")
    print(f"   Avg Interval Width: {avg_width:.6e}")
    print(f"   Max Interval Width: {max_width:.6e}")
    
    # 4. Inclusion Check
    is_inside = (target >= lower_bound) & (target <= upper_bound)
    
    if is_inside.all():
        print(" PASS: TensorFlow result is within PyTorch's numerical perturbation envelope.")
        print("   Conclusion: Framework differences stem only from floating-point rounding or minor accumulation order.")
    else:
        print(" FAIL: TensorFlow result exceeds the numerical perturbation envelope!")
        violation = torch.max(torch.relu(lower_bound - target) + torch.relu(target - upper_bound))
        print(f"   Max Violation Magnitude: {violation.item():.6e}")

# Execute verification
verify_range(val_target, val_center, val_perturb)

script_end = time.perf_counter()

print(f" Total Analysis Duration: {script_end - script_start:.6f} seconds")