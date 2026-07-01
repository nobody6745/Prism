import torch
import os
import time 

# Directory and file path configurations
SAVE_DIR = "./output_results"
MATMUL_FILE = os.path.join(SAVE_DIR, "matmul_diff.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    """Checks if the required data file exists."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        exit(1)

script_start = time.perf_counter()

# Ensure prerequisites are met before execution
check_file_exists(MATMUL_FILE)
check_file_exists(PERTURB_FILE)

# Load baseline comparison and perturbation-injected data
print(f"Loading {MATMUL_FILE} ...")
data_matmul = torch.load(MATMUL_FILE)

print(f"Loading {PERTURB_FILE} ...")
data_perturb = torch.load(PERTURB_FILE)

# Extract tensors for consistency analysis
# Target: Direct MatMul execution result (potential Type-II source)
val_target = data_matmul['outputs']['y2_direct'].cpu()
# Center: Unperturbed sliced execution result (baseline reference)
val_center = data_matmul['outputs']['y1_slice'].cpu()
# Perturb: Result after -1 ULP perturbation injection [cite: 510, 753]
val_perturb = data_perturb['outputs']['y1_slice_perturb'].cpu()

def verify_range(target, center, perturb):
    """
    Verifies if numerical discrepancies fall within the estimated error bounds 
    derived from perturbation injection[cite: 207, 605].
    """
    print(f"\n{'='*20} Starting Verification {'='*20}")
    
    # Calculate initial boundaries using PRISM's symmetric logic 
    bound_a = perturb
    bound_b = 2 * center - perturb
    
    # Define the numerical safety envelope boundaries
    raw_lower = torch.min(bound_a, bound_b)
    raw_upper = torch.max(bound_a, bound_b)
    
    # Apply ULP compensation to account for hardware rounding [cite: 303, 753]
    lower_bound = torch.nextafter(raw_lower, torch.full_like(raw_lower, float('-inf')))
    upper_bound = torch.nextafter(raw_upper, torch.full_like(raw_upper, float('inf')))
    
    # Calculate statistics for the error envelope
    widths = upper_bound - lower_bound
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    # Determine if results are benign (Type-I) or implementation defects (Type-II) [cite: 139, 373]
    is_inside = (target >= lower_bound) & (target <= upper_bound)
    
    total_elements = target.numel()
    passed_elements = is_inside.sum().item()
    pass_rate = (passed_elements / total_elements) * 100
    
    print(f"Statistical Summary:")
    print(f"Average Width: {avg_width:.6e}")
    print(f"Maximum Width (Max Width): {max_width:.6e}")
    print(f"Total Elements: {total_elements}")
    print(f"Pass Rate: {pass_rate:.4f}%")
    
    if passed_elements < total_elements:
        # Quantify the magnitude of deviation for elements identified as Type-II errors [cite: 610]
        diff_lower = torch.relu(lower_bound - target)
        diff_upper = torch.relu(target - upper_bound)
        max_violation = torch.max(diff_lower + diff_upper).item()
        print(f"Maximum Out-of-Bounds Violation: {max_violation:.6e}")

    return passed_elements == total_elements

# Execute verification and provide final diagnostic status
passed = verify_range(val_target, val_center, val_perturb)

script_end = time.perf_counter()
print("\n" + "="*50)
print(f"Total Script Execution Time: {script_end - script_start:.4f} seconds")

if passed:
    print("ALL PASS: Numerical consistency is within the safety envelope.")
else:
    print("WARNING: Out-of-bounds values detected (Potential Type-II defect).")

print("="*50)