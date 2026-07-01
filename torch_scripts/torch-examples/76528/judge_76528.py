import torch
import os
import time

# Directory and file path configurations
SAVE_DIR = "./output_results"
MAIN_FILE = os.path.join(SAVE_DIR, "einsum_batch_diff.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    """Checks if the required data file exists."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        exit(1)

script_start = time.perf_counter()

# Ensure prerequisites are met
check_file_exists(MAIN_FILE)
check_file_exists(PERTURB_FILE)

# Load baseline and comparison results
print(f"Loading {MAIN_FILE} ...")
data_main = torch.load(MAIN_FILE)

# Load perturbation-injected results
print(f"Loading {PERTURB_FILE} ...")
data_perturb = torch.load(PERTURB_FILE)

try:
    # Target: Batched execution result (e.g., batch_0)
    val_target = data_main['outputs']['batch_0'].cpu()
    # Center: Unperturbed single-operator result (baseline)
    val_center = data_main['outputs']['single'].cpu()
    # Perturb: Result after -1 ULP perturbation injection
    val_perturb = data_perturb['outputs']['single_perturb'].cpu()
except KeyError as e:
    print(f"Error: Dictionary key mismatch. Missing key: {e}")
    exit(1)

def verify_range(target, center, perturb):
    """
    Verifies if the target results fall within the numerical safety envelope 
    defined by the perturbation injection.
    """
    print(f"\n{'='*20} Starting Verification {'='*20}")
    
    # Calculate interval boundaries based on PRISM's symmetric logic
    bound_a = perturb
    bound_b = 2 * center - perturb
    
    raw_lower = torch.min(bound_a, bound_b)
    raw_upper = torch.max(bound_a, bound_b)
    
    # Apply ULP compensation to account for hardware-level rounding behaviors
    lower_bound = torch.nextafter(raw_lower, torch.full_like(raw_lower, float('-inf')))
    upper_bound = torch.nextafter(raw_upper, torch.full_like(raw_upper, float('inf')))
    
    # Calculate interval statistics
    widths = upper_bound - lower_bound
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    # Determine classification (In-bound corresponds to Type-I benign errors)
    is_inside = (target >= lower_bound) & (target <= upper_bound)
    total_elements = target.numel()
    passed_elements = is_inside.sum().item()
    failed_elements = total_elements - passed_elements
    pass_rate = (passed_elements / total_elements) * 100
    
    print(f"Statistical Summary:")
    print(f"   Average Width: {avg_width:.6e}")
    print(f"   Maximum Width: {max_width:.6e}")
    print(f"   Total Elements: {total_elements}")
    print(f"   Pass Rate: {pass_rate:.4f}%")
    
    if failed_elements > 0:
        # Quantify the magnitude of deviation for Type-II error identification
        diff_lower = torch.relu(lower_bound - target)
        diff_upper = torch.relu(target - upper_bound)
        max_violation = torch.max(diff_lower + diff_upper).item()
        print(f"Maximum Out-of-Bounds Violation: {max_violation:.6e}")

    return failed_elements == 0
passed = verify_range(val_target, val_center, val_perturb)

script_end = time.perf_counter()
print("\n" + "="*50)
print(f"Total Script Execution Time: {script_end - script_start:.4f} seconds")

if passed:
    print("ALL PASS: Results are consistent within the numerical safety envelope.")
else:
    print("WARNING: Some results fall outside the predicted perturbation range.")
    print("   This typically indicates that parallel computation (Einsum Batching)")
    print("   altered the associativity of floating-point addition.")

print("="*50)