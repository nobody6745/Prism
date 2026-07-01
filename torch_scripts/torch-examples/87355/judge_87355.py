import torch
import time

def check_range(target, center, perturbed, name_target, name_center, name_perturbed):
    print(f"\n Checking if '{name_target}' is within bounds...")
    
    target = target.float().cpu()
    center = center.float().cpu()
    perturbed = perturbed.float().cpu()
    
    bound1 = perturbed
    bound2 = 2 * center - perturbed
    
    lower = torch.min(bound1, bound2)
    upper = torch.max(bound1, bound2)
    
    widths = upper - lower
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()
    
    epsilon = 1e-7
    is_inside = (target >= (lower - epsilon)) & (target <= (upper + epsilon))
    
    total_elements = target.numel()
    passed_elements = is_inside.sum().item()
    pass_ratio = passed_elements / total_elements * 100

    print(f"   > Average Interval Width: {avg_width:.6e}")
    print(f"   > Maximum Interval Width: {max_width:.6e}")
    print(f"   > Passed Elements: {passed_elements} / {total_elements} ({pass_ratio:.2f}%)")
    
    if passed_elements < total_elements:
        print(f"   > FAILED: {total_elements - passed_elements} elements are outside the error bar.")
    else:
        print(f"   > PASSED: Target falls within the perturbation error bar.")

def main():
    script_start = time.perf_counter()
    
    try:
        print("📂 Loading data files...")
        clean_data = torch.load("linear_vs_gemm_debug.pt")
        perturb_data = torch.load("linear_perturb.pt")
        
        a = clean_data["a"]
        b = clean_data["b"]
        c = clean_data["c"]
        d = clean_data["d"]
        
        K = a.shape[0]
        a_perturb = perturb_data.get("a_perturb", perturb_data.get("a"))
        c_perturb = perturb_data.get("c_perturb", perturb_data.get("c"))

        if a_perturb is None:
            raise KeyError("Cannot find 'a' or 'a_perturb' in linear_perturb.pt")

        print("\n" + "="*60)
        print(f"TEST CONFIGURATION (K={K})")
        print("="*60)
        check_range(
            target=b[:K], 
            center=a, 
            perturbed=a_perturb, 
            name_target="b[:K] (FP32 Truth)", 
            name_center="a (FP16 Orig)", 
            name_perturbed="a_perturb"
        )

    except Exception as e:
        print(f" An error occurred: {e}")
    
    total_time = time.perf_counter() - script_start
    print(f"\n  Total Analysis Time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()