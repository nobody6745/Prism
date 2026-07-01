import torch
import numpy as np
from torch import allclose as talc
from numpy import allclose as nalc
import os
import time 


torch.backends.cuda.matmul.allow_tf32 = False

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "perturb.pt")

torch.manual_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Phi = torch.randn((300, 300, 3)).to(device)
F = torch.randn((300, 3)).to(device)

Phi_b = Phi.unsqueeze(0).expand(16, -1, -1, -1)
F_b = F.unsqueeze(0).expand(16, -1, -1)


if device.type == 'cuda':
    torch.cuda.synchronize()

start_b = time.perf_counter()
torch_es_b = torch.einsum('bimd,bmp->bipd', Phi_b, F_b)
if device.type == 'cuda':
    torch.cuda.synchronize()
end_b = time.perf_counter()
time_batch = end_b - start_b

start_s = time.perf_counter()
torch_es = torch.einsum('imd,mp->ipd', Phi, F)
if device.type == 'cuda':
    torch.cuda.synchronize()
end_s = time.perf_counter()
time_single = end_s - start_s

result_batch_item = torch_es_b[0]
result_single = torch_es

diff_matrix = (result_batch_item - result_single).abs()
max_diff = diff_matrix.max().item()

print("-" * 30)
print(f"Time - Batch Mode (16): {time_batch:.6f} s")
print(f"Time - Single Mode:     {time_single:.6f} s")
print(f"Speedup per sample:     {(time_single * 16) / time_batch:.2f}x")
print("-" * 30)

saved_data = {
    "inputs": {
        "Phi": Phi.cpu(),
        "F": F.cpu()
    },
    "outputs": {
        "batch_0": result_batch_item.cpu(),
        "single_perturb": result_single.cpu()
    },
    "analysis": {
        "diff_matrix": diff_matrix.cpu(),
        "max_diff": max_diff,
        "time_batch": time_batch,
        "time_single": time_single
    }
}

torch.save(saved_data, DATA_FILE)
