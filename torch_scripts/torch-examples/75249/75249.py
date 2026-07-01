import torch
from torch import absolute as ab
from torch.linalg import pinv as pinv
import random
import numpy as np
import os
import time  

SAVE_DIR = "./output_results"
TENSOR_FILE = os.path.join(SAVE_DIR, "tensors.pt")
REPORT_FILE = os.path.join(SAVE_DIR, "report.txt")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

SEED = 1337
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


total_start = time.perf_counter()

I, J, K, L = 5, 10, 32, 32
w_ = torch.randn(I, J)
P_XIs = torch.randn(J, K, L)


t0 = time.perf_counter()
M_X_cpu = torch.einsum('ij,jkl->ijkl', w_, P_XIs)
M_X_cuda = torch.einsum('ij,jkl->ijkl', w_.cuda(), P_XIs.cuda())
torch.cuda.synchronize()  
duration_einsum = time.perf_counter() - t0


t1 = time.perf_counter()
S_cpu = M_X_cpu.sum(dim=1)
S_cuda = M_X_cuda.sum(dim=1)
torch.cuda.synchronize()
duration_sum = time.perf_counter() - t1


t2 = time.perf_counter()
Pinv_S_cpu = pinv(S_cpu)
Pinv_S_cuda = pinv(S_cuda)
torch.cuda.synchronize()
duration_pinv = time.perf_counter() - t2

diff_einsum = ab(M_X_cuda.cpu() - M_X_cpu).max().item()
diff_sum = ab(S_cuda.cpu() - S_cpu).max().item()
diff_pinv = ab(Pinv_S_cuda.cpu() - Pinv_S_cpu).max().item()

total_end = time.perf_counter()
duration_total = total_end - total_start


report_str = f"""
Seed: {SEED}
Tensor Shapes:
  w_: {w_.shape}
  P_XIs: {P_XIs.shape}
  M_X: {M_X_cpu.shape}
  S: {S_cpu.shape}

Time Statistics:
------------------------------------------
Einsum Step:    {duration_einsum:.6f} s
Sum Step:       {duration_sum:.6f} s
Pinv Step:      {duration_pinv:.6f} s
Total Run:      {duration_total:.4f} s

Max Absolute Difference:
------------------------------------------
1. Einsum (M_X) Diff:  {diff_einsum:.6e}
2. Sum (S) Diff:       {diff_sum:.6e}
3. Pinv Diff:          {diff_pinv:.6e}
==========================================
"""

print(report_str)


saved_data = {
    "inputs": {"w": w_, "P_XIs": P_XIs},
    "cpu_results": {"M_X": M_X_cpu, "S": S_cpu, "Pinv": Pinv_S_cpu},
    "cuda_results": {
        "M_X": M_X_cuda.cpu(),
        "S": S_cuda.cpu(),
        "Pinv": Pinv_S_cuda.cpu()
    },
    "timings": {
        "einsum": duration_einsum,
        "sum": duration_sum,
        "pinv": duration_pinv,
        "total": duration_total
    }
}

torch.save(saved_data, TENSOR_FILE)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report_str)