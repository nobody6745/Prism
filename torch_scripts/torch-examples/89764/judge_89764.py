import torch
import os
import time

def evaluate_numerical_stability():
    script_start = time.perf_counter()

    file_clean = "diff_analysis_data.pt"
    file_perturb = "perturb_data.pt"
    
    if not (os.path.exists(file_clean) and os.path.exists(file_perturb)):
        print(f"Error: Data files not found. Ensure {file_clean} and {file_perturb} exist.")
        return

    print(f"Loading data...")
    try:
        data_clean = torch.load(file_clean)
        data_perturb = torch.load(file_perturb)  
        val_diff = data_clean["diff_double"].cpu()       
        val_perturb = data_perturb["perturb_double"].cpu() 
    except KeyError as e:
        print(f"Key Error: Key {e} not found.")
        return
    except Exception as e:
        print(f"Load failed: {e}")
        return

    bound1 = val_perturb
    bound2 = -val_perturb
    
    lower = torch.min(bound1, bound2)
    upper = torch.max(bound1, bound2)
    widths = upper - lower
    avg_width = torch.mean(widths).item()
    max_width = torch.max(widths).item()

    is_inside = (val_diff >= lower) & (val_diff <= upper)
    total = val_diff.numel()
    passed = is_inside.sum().item()
    ratio = passed / total * 100

    print(f"\n{'='*60}")
    print(f"Numerical Stability Diagnostic Report (ULP Range Check)")
    print(f"{'='*60}")
    print(f"Tolerance Range Analysis:")
    print(f"  Average Interval Width (Avg Width): {avg_width:.6e}")
    print(f"  Maximum Interval Width (Max Width): {max_width:.6e}")
    print(f"{'-'*60}")
    print(f"Decision Statistics:")
    print(f"  Total Elements: {total}")
    print(f"  Pass Rate:      {ratio:.2f}% ({passed}/{total})")
    print(f"  Status:         {'PASS' if passed == total else 'FAIL'}")
    print(f"{'-'*60}")
    print(f" Total Script Execution Time: {script_end - script_start:.6f} seconds")

    if passed < total:
        print("\n Failure Case Analysis (Top 3 samples):")
        fail_indices = torch.nonzero(~is_inside, as_tuple=True)[0]
        for idx in fail_indices[:3]:
            i = idx.item()
            print(f"  Index {i}: Diff={val_diff[i]:.8e}, Range=[{lower[i]:.8e}, {upper[i]:.8e}]")

if __name__ == "__main__":
    evaluate_numerical_stability()