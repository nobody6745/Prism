import torch
import os
import time

def check_range(target, center, perturbed):
    analysis_start = time.perf_counter()
    print(f"\n Starting Judgment: Comparing CPU Result vs GPU Perturbation Range")
    
    target = target.float()
    center = center.float()
    perturbed = perturbed.float()
    
    if not (target.shape == center.shape == perturbed.shape):
        print(f" Shape mismatch!")
        return

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
    
    print(f"   > Avg Interval Width: {avg_width:.6e}")
    print(f"   > Max Interval Width: {max_width:.6e}")
    print(f"   > Total Elements: {total_elements}")
    print(f"   > Passed Elements: {passed_elements} ({pass_ratio:.2f}%)")
    
    if passed_elements < total_elements:
        print(f"   > FAILED: {total_elements - passed_elements} elements are outside.")
    else:
        print(f"   > PASSED: CPU Truth falls within the GPU perturbation error bar.")

    print(f"\n    Analysis Time: {time.perf_counter() - analysis_start:.4f} s")

def main():
    script_start = time.perf_counter()
    file_clean = "cumprod_results.pt"
    file_perturb = "perturb_results.pt"
    
    if not (os.path.exists(file_clean) and os.path.exists(file_perturb)):
        print(f" Error: Files not found.")
        return

    print(f" Loading data files...")
    data_clean = torch.load(file_clean)
    data_perturb = torch.load(file_perturb)
    
    try:
        check_range(target=data_clean["out_cpu"], 
                    center=data_clean["out_gpu"], 
                    perturbed=data_perturb["out_gpu_perturb"])
    except KeyError as e:
        print(f" Key Error: {e}")

    print(f"\n Total Execution Time: {time.perf_counter() - script_start:.4f} s")

if __name__ == "__main__":
    main()