import torch
import os
import time  

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "perturb.pt")

def reproduce_issue(seed):
    print(f"Testing with Random Seed: {seed}")
    torch.manual_seed(seed)
    B, T, C = 32, 64, 16
    x = torch.randn(B, T, C)
    t0 = time.perf_counter()
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1] 
            xbow[b, t] = torch.mean(xprev, 0)
    duration_loop = time.perf_counter() - t0
            
    t1 = time.perf_counter()
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    xbow2 = wei @ x  
    duration_matmul = time.perf_counter() - t1

    is_close = torch.allclose(xbow, xbow2)
    max_diff = (xbow - xbow2).abs().max()
    
    print(f"  Loop Time:   {duration_loop:.8f} s")
    print(f"  Matmul Time: {duration_matmul:.8f} s")
    print(f"Result match (allclose): {is_close}")
    print(f"Max difference: {max_diff.item():.10e}")

    return {
        "inputs": x,
        "outputs": {
            "loop_result_perturb": xbow,
        },
        "timings": {
            "loop_duration": duration_loop,
            "matmul_duration": duration_matmul
        }
    }

if __name__ == "__main__":
    script_start = time.perf_counter()
    data = reproduce_issue(seed=1337)
    torch.save(data, DATA_FILE)
    
    script_total = time.perf_counter() - script_start
    print(f"time : {script_total:.4f} s")