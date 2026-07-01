import torch
import os
import time  

SAVE_DIR = "./output_results"
TENSORS_FILE = os.path.join(SAVE_DIR, "tensors.pt")
PERTURB_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Error: Cannot find file {path}")
        exit(1)

start_time = time.perf_counter()

check_file_exists(TENSORS_FILE)
check_file_exists(PERTURB_FILE)

print(f"Loading {TENSORS_FILE} ...")
data_base = torch.load(TENSORS_FILE)

print(f"Loading {PERTURB_FILE} ...")
data_perturb = torch.load(PERTURB_FILE)

try:
    S_cpu_clean = data_base['cpu_results']['S']
    S_cuda = data_base['cuda_results']['S']

    if 'S_perturb' in data_perturb['cpu_results']:
        S_cpu_perturb = data_perturb['cpu_results']['S_perturb']
    else:
        S_cpu_perturb = data_perturb['cuda_results']['S_perturb']

except KeyError as e:
    print(f"Error: Dictionary key mismatch. Missing key: {e}")
    exit(1)

def verify_range(name, val_cpu_clean, val_cuda, val_cpu_perturb):
    """
    Verifies if CUDA results fall within the perturbation-injected error bounds.
    """
    print(f"\n{'='*20} Verifying: {name} {'='*20}")
    
    bound_a = val_cpu_perturb
    bound_b_raw = 2 * val_cpu_clean - val_cpu_perturb
    
    bound_b = torch.nextafter(bound_b_raw, torch.full_like(bound_b_raw, float('inf')))
    bound_b = torch.nextafter(bound_b, torch.full_like(bound_b, float('inf')))
    bound_b = torch.nextafter(bound_b, torch.full_like(bound_b, float('inf')))

    lower_bound = torch.min(bound_a, bound_b)
    upper_bound = torch.max(bound_a, bound_b)
    
    interval_widths = upper_bound - lower_bound
    avg_width = torch.mean(interval_widths).item()
    max_width = torch.max(interval_widths).item()

    is_inside = (val_cuda >= lower_bound) & (val_cuda <= upper_bound)

    total_elements = val_cuda.numel()
    passed_elements = is_inside.sum().item()
    failed_elements = total_elements - passed_elements
    pass_rate = (passed_elements / total_elements) * 100
    
    print(f"Decision Interval: [min(bound_a, bound_b), max(bound_a, bound_b)]")
    print(f"Average Interval Width: {avg_width:.6e}")
    print(f"Maximum Interval Width: {max_width:.6e}")
    print(f"Elements Passed: {passed_elements} / {total_elements}")
    print(f"Pass Rate:       {pass_rate:.4f}%")
    
    if failed_elements > 0:
        diff_lower = torch.relu(lower_bound - val_cuda)
        diff_upper = torch.relu(val_cuda - upper_bound)
        max_violation = torch.max(diff_lower + diff_upper).item()
        print(f"Max Out-of-Bound Violation: {max_violation:.6e}")
        
        failed_indices = torch.nonzero(~is_inside, as_tuple=False)
        print("Details for first 3 failed samples:")
        for idx in failed_indices[:3]:
            i = tuple(idx.tolist())
            print(f"  Index {i}: Cuda={val_cuda[i]:.8e}, Range=[{lower_bound[i]:.8e}, {upper_bound[i]:.8e}]")

    return failed_elements == 0

res_s = verify_range("Sum (S)", S_cpu_clean, S_cuda, S_cpu_perturb)

end_time = time.perf_counter()
total_duration = end_time - start_time

print("\n" + "="*50)
print("Final Diagnostic Report")
print("="*50)
print(f"⏱Total Execution Time: {total_duration:.4f} seconds")

if res_s:
    print("ALL PASS: All CUDA results fall within the predicted perturbation envelope!")
else:
    print("WARNING: Some results fell outside the predicted interval (potential Type-II defects).")