import torch
import math
import os
import numpy as np
import time 

def perturb_down_ulp(t):
    """Perturbs a value by 1 ULP (Unit in the Last Place) toward negative infinity."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    target = torch.full_like(t, float('-inf'))
    return torch.nextafter(t, target)

def local_perturbed_arange(start, end, step, dtype=torch.float32):
    """Custom arange implementation using intentional ULP downward perturbation."""
    span = end - start
    num_steps = math.ceil(span / step)
    res = torch.empty(num_steps, dtype=dtype)
    start_t = torch.tensor(start, dtype=dtype)
    step_t = torch.tensor(step, dtype=dtype)
    
    for i in range(num_steps):
        i_t = torch.tensor(i, dtype=dtype)
        # Apply perturbation during multiplication and addition
        offset = perturb_down_ulp(i_t * step_t)
        val = perturb_down_ulp(start_t + offset)
        res[i] = val
    return res

def run_judge():
    script_start = time.perf_counter()

    DATA_FILE = "./output_results/arange_data.pt"
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return
        
    data = torch.load(DATA_FILE)
    
    print("\n" + "="*70)
    print("📋 Numerical Consistency Judge Report")
    print("Formula: Torch ∈ [min(A, B), max(A, B)] where A=Local_P, B=2*Ref-Local_P")
    print("="*70)

    for test_name, test_data in data.items():
        case_start = time.perf_counter()
        print(f"\n Case: {test_name}")
        
        # 1. Extract parameters and data
        params = test_data['params']
        start, end, step, _, dtype = params
        val_torch = test_data['torch']
        val_tf_numpy = test_data['tensorflow']
        
        if val_tf_numpy is None:
            print(" Skipping: Missing TensorFlow reference data")
            continue
            
        val_ref = torch.from_numpy(val_tf_numpy)
        val_local_perturbed = local_perturbed_arange(start, end, step, dtype=dtype)
        
        # Length alignment
        min_len = min(val_torch.numel(), val_ref.numel(), val_local_perturbed.numel())
        val_torch = val_torch[:min_len]
        val_ref = val_ref[:min_len]
        val_local_p = val_local_perturbed[:min_len]

        # 2. Calculate the decision interval
        # Bound B uses the reference (TF) as the center point
        bound_a = val_local_p
        bound_b = 2 * val_ref - val_local_p
        lower_bound = torch.min(bound_a, bound_b)
        upper_bound = torch.max(bound_a, bound_b)

        # 3. Precision Analysis
        widths = (upper_bound - lower_bound).abs()
        avg_width = torch.mean(widths).item()
        max_width = torch.max(widths).item()
        
        # Inclusion check
        is_inside = (val_torch >= lower_bound) & (val_torch <= upper_bound)
        total = val_torch.numel()
        passed = is_inside.sum().item()
        pass_rate = (passed / total) * 100
        
        case_duration = time.perf_counter() - case_start
        print(f" Avg Interval Width: {avg_width:.6e}")
        print(f" Max Interval Width: {max_width:.6e}")
        print(f" Execution Duration: {case_duration:.4f} s")
        print(f" Pass Rate:          {pass_rate:.2f}% ({passed}/{total})")
        
        if passed < total:
            # Calculate the magnitude of drift for out-of-bounds elements
            max_violation = torch.max(torch.relu(lower_bound - val_torch) + torch.relu(val_torch - upper_bound)).item()
            print(f" FAILED: Max Violation Magnitude = {max_violation:.6e}")

    script_total = time.perf_counter() - script_start
    print("\n" + "="*70)
    print(f" Judgment Complete. Total Script Execution Time: {script_total:.4f} s")
    print("="*70)

if __name__ == "__main__":
    run_judge()