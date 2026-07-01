import torch
from torch import nn
import os
import time 

# Directory and File Setup
SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "perturb.pt")

script_start = time.perf_counter()

torch.random.manual_seed(200)

# Configuration
dtype = torch.float16
batch, seq, embed_dim, delay = 2, 16, 4, 2
device = "cuda"

# Initialize Tensors
x1 = torch.randn(batch, seq, embed_dim, device=device, dtype=dtype)
x2 = x1[:, delay:] # Slice input for Path B

net = nn.Linear(embed_dim, embed_dim).to(device).to(dtype=dtype)

# GPU Warmup
_ = net(x1)
torch.cuda.synchronize()

# --- Path A: Full Calculation then Slicing ---
torch.cuda.synchronize()
t0 = time.perf_counter()

out1 = net(x1)
out1_slice = out1[:, delay:]

torch.cuda.synchronize()
duration_a = time.perf_counter() - t0

# --- Path B: Sliced Input then Calculation ---
torch.cuda.synchronize()
t1 = time.perf_counter()

out2 = net(x2)

torch.cuda.synchronize()
duration_b = time.perf_counter() - t1

# Numerical Consistency Analysis
diff_matrix = (out1_slice - out2).abs()
max_diff = diff_matrix.max()

print(f"  Path A (Full -> Slice) Duration: {duration_a:.8f} s")
print(f"  Path B (Slice -> Calc) Duration: {duration_b:.8f} s")

print(f" Maximum Absolute Difference: {max_diff.item():.10e}")

is_close = torch.allclose(out1_slice, out2)
print(f" Judgment Result (allclose): {is_close}")

# Data Persistence
saved_data = {
    "outputs": {
        "out1_perturb": out1_slice.detach().cpu(),
        "out2_direct": out2.detach().cpu()
    },
    "timings": {
        "duration_a": duration_a,
        "duration_b": duration_b
    },
    "diff": {
        "max_diff": max_diff.item(),
        "is_allclose": is_close
    }
}

torch.save(saved_data, DATA_FILE)
script_total = time.perf_counter() - script_start
print(f" Data saved successfully. Total script execution time: {script_total:.4f} s")