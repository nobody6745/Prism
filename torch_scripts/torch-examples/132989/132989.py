import torch
import os
import time  

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "reproduce_issue_data.pt")

def reproduce_issue(seed):
    print(f" Testing with Random Seed: {seed} ")
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
    is_close = torch.allclose(xbow, xbow2, atol=1e-7)
    diff_matrix = (xbow - xbow2).abs()
    max_diff = diff_matrix.max()
    
    print(f"  Loop Time:   {duration_loop:.6f} s")
    print(f"  Matmul Time: {duration_matmul:.6f} s")
    print(f"Result match (allclose): {is_close}")
    print(f" Max difference: {max_diff.item():.10e}")

    return {
        "outputs": {"loop_result": xbow, "matmul_result": xbow2},
        "timings": {"loop": duration_loop, "matmul": duration_matmul},
        "analysis": {"max_diff": max_diff.item(), "is_close": is_close}
    }

if __name__ == "__main__":
    script_start = time.perf_counter()
    data = reproduce_issue(seed=1337)

    torch.save(data, DATA_FILE)
    print(f"time: {time.perf_counter() - script_start:.4f} s")