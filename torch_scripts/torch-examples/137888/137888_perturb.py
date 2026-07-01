import numpy as np
import torch
import os
import time  # 导入计时模块

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "digamma_perturb_test.pt")

def perturb_towards_neg_inf(tensor):
    target = torch.full_like(tensor, float('-inf'))
    return torch.nextafter(tensor, target)

def digamma_perturbed(x):
    x_p = perturb_towards_neg_inf(x)
    
    y = torch.digamma(x_p)
    
    y_p = perturb_towards_neg_inf(y)
    
    return y_p

def test_digamma():
    script_start = time.perf_counter()
    
    print("=== Testing torch.digamma with Perturbation ===")
    
    input_val = 0.10000000149011612
    input_data = np.array([input_val], dtype=np.float32)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    t0 = time.perf_counter()
    result_std_tensor = torch.digamma(input_tensor)
    duration_std = time.perf_counter() - t0
    
    result_std = result_std_tensor.item()
    
    t1 = time.perf_counter()
    result_ptb_tensor = digamma_perturbed(input_tensor)
    duration_ptb = time.perf_counter() - t1
    
    result_ptb = result_ptb_tensor.item()
    print(f" Standard Duration:  {duration_std:.8f} s")
    print(f"  Perturbed Duration: {duration_ptb:.8f} s")
    print(f"⚡ Overhead Ratio:     {duration_ptb / duration_std:.2f}x")
    print(f"Standard Result:    {result_std:.20e}")
    print(f"Perturbed Result:   {result_ptb:.20e}")
    
    diff = result_std - result_ptb
    is_valid = result_ptb <= result_std
    print(f"Difference:         {diff:.20e}")

    saved_data = {
        "outputs": {
            "standard": result_std_tensor,
            "perturbed": result_ptb_tensor
        },
        "timings": {
            "duration_std": duration_std,
            "duration_ptb": duration_ptb
        },
        "analysis": {
            "diff": diff,
            "is_valid": is_valid
        }
    }

    torch.save(saved_data, DATA_FILE)
    script_total = time.perf_counter() - script_start
    print(f"\n time: {script_total:.4f} s")

if __name__ == "__main__":
    test_digamma()