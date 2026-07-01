import torch
import numpy as np
import os
import time 


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


script_start = time.perf_counter()

print(f"{'d':<3} | {'AllClose':<8} | {'Norm Diff':<15} | {'Time(ms)':<10} | {'Action'}")
print("-" * 65)

for d in range(2, 10):
    loop_start = time.perf_counter()
    
    a = torch.randn(64, d, d)
    b = torch.randn(64, d)
    

    y1 = torch.einsum("bij,bj->bi", a, b).sum(dim=1) 
    y1_direct = torch.einsum("bij,bj->b", a, b)
    
    y2 = torch.einsum("bij,bj->bi", a, b).sum(dim=1)
    
    is_close = torch.allclose(y1_direct, y2)
    diff_norm = torch.linalg.norm(y1_direct - y2).item()
    
    loop_duration = (time.perf_counter() - loop_start) * 1000  
    
    action_msg = ""
    if not is_close:
        filename = f"mismatch_d{d}.npz"
        data_to_save = {
            "y1": y1_direct.detach().cpu().numpy(),
            "y2": y2.detach().cpu().numpy(),
            "a": a.detach().cpu().numpy(),
            "b": b.detach().cpu().numpy(),
            "diff_norm": diff_norm
        }
        np.savez(filename, **data_to_save)
        action_msg = f"Saved to {filename}"
    else:
        action_msg = "Pass"

    print(f"{d:<3} | {str(is_close):<8} | {diff_norm:<15.4e} | {loop_duration:<10.4f} | {action_msg}")

script_duration = time.perf_counter() - script_start

print("-" * 65)
print(f"Total script execution time: {script_duration:.4f} seconds")