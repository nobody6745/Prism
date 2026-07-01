import torch
import random
import numpy as np
import os
import time 

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "dot_sum_thread_diff.pt")

script_start = time.perf_counter()

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
N = 1_000_000_000
print(f"Initializing tensor with {N} elements (Approx 4GB)...")
t_init = time.perf_counter()
x = torch.randn(N, dtype=torch.float32, device='cpu')
print(f"Initialization took: {time.perf_counter() - t_init:.4f} s")

def get_calculations(label):
    print(f"\n[ {label} ]")
    
    t0 = time.perf_counter()
    val_norm = torch.linalg.norm(x) ** 2
    d_norm = time.perf_counter() - t0
    print(f"  - linalg.norm: {d_norm:.4f} s")
    t1 = time.perf_counter()
    val_dot = torch.dot(x, x)
    d_dot = time.perf_counter() - t1
    print(f"  - dot:         {d_dot:.4f} s")

    t2 = time.perf_counter()
    val_sum = (x * x).sum()
    d_sum = time.perf_counter() - t2
    print(f"  - sum:         {d_sum:.4f} s")
    
    return {
        "norm": val_norm, "dot": val_dot, "sum": val_sum,
        "timings": {"norm": d_norm, "dot": d_dot, "sum": d_sum}
    }

res_1 = get_calculations("1 Thread")

torch.set_num_threads(10)
res_10 = get_calculations("10 Threads")

print(f"{'Method':<10} | {'1-Th Time':<10} | {'10-Th Time':<10} | {'Speedup':<10} | {'Diff':<15}")


methods = ["norm", "dot", "sum"]
diffs = {}

for m in methods:
    v1, v2 = res_1[m].item(), res_10[m].item()
    t1, t10 = res_1["timings"][m], res_10["timings"][m]
    speedup = t1 / t10
    diff = v1 - v2
    diffs[m] = diff
    print(f"{m:<10} | {t1:7.4f}s  | {t10:8.4f}s  | {speedup:7.2f}x   | {diff:.4e}")

saved_data = {
    "meta": {"N": N},
    "dot": {"thread_1": res_1["dot"], "thread_10": res_10["dot"], "diff": diffs["dot"]},
    "sum": {"thread_1": res_1["sum"], "thread_10": res_10["sum"], "diff": diffs["sum"]},
    "norm": {"thread_1": res_1["norm"], "thread_10": res_10["norm"], "diff": diffs["norm"]}
}

torch.save(saved_data, DATA_FILE)
script_total = time.perf_counter() - script_start
print(f"\n time: {script_total:.4f} s")