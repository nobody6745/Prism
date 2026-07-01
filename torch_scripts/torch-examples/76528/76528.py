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
DATA_FILE = os.path.join(SAVE_DIR, "einsum_batch_diff.pt")

torch.manual_seed(2024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Phi = torch.randn((300, 300, 3)).to(device)
F = torch.randn((300, 3)).to(device)

Phi_b = Phi.unsqueeze(0).expand(16, -1, -1, -1)
F_b = F.unsqueeze(0).expand(16, -1, -1)
print("-" * 30)

if device.type == 'cuda': torch.cuda.synchronize()
start_b = time.perf_counter()
torch_es_b = torch.einsum('bimd,bmp->bipd', Phi_b, F_b)
if device.type == 'cuda': torch.cuda.synchronize()
end_b = time.perf_counter()
duration_b = end_b - start_b

if device.type == 'cuda': torch.cuda.synchronize()
start_s = time.perf_counter()
torch_es = torch.einsum('imd,mp->ipd', Phi, F)
if device.type == 'cuda': torch.cuda.synchronize()
end_s = time.perf_counter()
duration_s = end_s - start_s

Phi_double, F_double = Phi.double(), F.double()
Phi_b_double = Phi_double.unsqueeze(0).expand(16, -1, -1, -1)
F_b_double = F_double.unsqueeze(0).expand(16, -1, -1)

torch_es_b_double = torch.einsum('bimd,bmp->bipd', Phi_b_double, F_b_double)
torch_es_double = torch.einsum('imd,mp->ipd', Phi_double, F_double)

result_batch_item = torch_es_b[0]
result_single = torch_es

diff_matrix = (result_batch_item - result_single).abs()
max_diff = diff_matrix.max().item()

is_equal_f32 = talc(result_batch_item, result_single)

if not is_equal_f32:
    print(f"  Max Diff: {max_diff:.10e}")

saved_data = {
    "outputs": {
        "batch_0": result_batch_item.cpu(),
        "single": result_single.cpu(),
        "time_batch": duration_b,
        "time_single": duration_s
    },
    "analysis": {
        "max_diff": max_diff
    }
}

torch.save(saved_data, DATA_FILE)