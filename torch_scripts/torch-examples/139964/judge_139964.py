import torch
import os
import time

def check_range(target, center, perturbed, tag):
    start_time = time.perf_counter()
    print(f"\n Checking configuration: '{tag}'")

    target = target.float()
    center = center.float()
    perturbed = perturbed.float()
    
    bound_a = perturbed
    bound_b = 2 * center - perturbed
    lower = torch.min(bound_a, bound_b)
    upper = torch.max(bound_a, bound_b)
    widths = (upper - lower).abs()
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    epsilon = 1e-9
    is_inside = (target >= (lower - epsilon)) & (target <= (upper + epsilon))
    
    total_elements = target.numel()
    passed_elements = is_inside.sum().item()
    pass_ratio = passed_elements / total_elements * 100
    
    duration = time.perf_counter() - start_time
    
    print(f"   > Avg Interval Width: {avg_width:.6e}")
    print(f"   > Max Interval Width: {max_width:.6e}")
    print(f"   > Total Elements: {total_elements}")
    print(f"   > Passed Elements: {passed_elements} ({pass_ratio:.2f}%)")
    print(f"   > ⏱Analysis Time: {duration:.4f} s")
    
    if passed_elements < total_elements:
        print(f"   > FAILED: {total_elements - passed_elements} elements are outside the error bar.")
    else:
        print(f"   > PASSED: FP32 Truth falls within the perturbation error bar.")

def main():
    script_start = time.perf_counter()
    file_clean = "outputs.pt"
    file_perturb = "outputs_perturb.pt"
    
    if not (os.path.exists(file_clean) and os.path.exists(file_perturb)):
        print(f"Error: Input files not found.")
        return

    print(f" Loading data files...")
    data_clean = torch.load(file_clean)
    data_perturb = torch.load(file_perturb)
    
    keys = sorted(data_clean.keys())
    
    print("\n" + "="*60)
    print(f"STARTING JUDGEMENT ({len(keys)} configurations)")
    print("="*60)
    
    for key in keys:
        if key not in data_perturb: continue
        result_clean = data_clean[key]
        result_perturb = data_perturb[key]
        
        check_range(target=result_clean["a2"], 
                    center=result_clean["a1"], 
                    perturbed=result_perturb["a1_perturb"], 
                    tag=key)

    total_duration = time.perf_counter() - script_start
    print(f"JUDGEMENT FINISHED. Total analysis time: {total_duration:.4f} s")

if __name__ == "__main__":
    main()