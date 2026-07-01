import torch
from torch import absolute as ab
from torch.linalg import pinv as pinv
import random
import numpy as np
import os
import time  

SAVE_DIR = "./output_results"
TENSOR_FILE = os.path.join(SAVE_DIR, "perturb.pt")
REPORT_FILE = os.path.join(SAVE_DIR, "report.txt")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

SEED = 1337

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_total = time.perf_counter()

I, J, K, L = 5, 10, 32, 32


w_ = torch.randn(I, J)
P_XIs = torch.randn(J, K, L)

M_X_cpu = torch.einsum('ij,jkl->ijkl', w_, P_XIs)

S_cpu = M_X_cpu.sum(dim=1)


end_total = time.perf_counter()
duration_total = end_total - start_total

# 构造报告字符串
report_str = f"""

Seed: {SEED}
Total Execution Time: {duration_total:.6f} seconds
------------------------------------------
Tensor Shapes:
  w_: {w_.shape}
  P_XIs: {P_XIs.shape}
  M_X: {M_X_cpu.shape}
  S: {S_cpu.shape}
"""

print(report_str)

print(f"正在保存数据到 {SAVE_DIR} ...")

# 准备要保存的数据字典
saved_data = {
    "inputs": {
        "w": w_,
        "P_XIs": P_XIs
    },
    "cpu_results": {
        "M_X": M_X_cpu,
        "S_perturb": S_cpu,
    },
    "meta": {
        "duration": duration_total,
        "seed": SEED
    }
}

torch.save(saved_data, TENSOR_FILE)

print(f"data = torch.load('{TENSOR_FILE}')")
print(f"execution_time = data['meta']['duration']")