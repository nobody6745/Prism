import torch
import torch.nn as nn
import os
import time 

# Directory Configuration
SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "linear_batch_vs_loop.pt")

torch.manual_seed(2024)

# Environment Check
if not torch.cuda.is_available():
    print("Error: This script requires a CUDA environment")
    exit(1)

device = "cuda"
linear2 = nn.Linear(128, 550, bias=False).to(device)
input_tensor = torch.rand(32, 128).to(device)

torch.cuda.synchronize()
script_start = time.perf_counter()

# --- Method 1: Batch Mode ---
torch.cuda.synchronize()
t_batch_start = time.perf_counter()

output1 = linear2(input_tensor)

torch.cuda.synchronize()
time_batch = time.perf_counter() - t_batch_start

# --- Method 2: Loop Mode ---
torch.cuda.synchronize()
t_loop_start = time.perf_counter()

out = []
for i in range(input_tensor.shape[0]):
    cc = linear2(input_tensor[i])
    out.append(cc)
output2 = torch.stack(out)

torch.cuda.synchronize()
time_loop = time.perf_counter() - t_loop_start

# Precision and Performance Analysis
strict_equal = (output1 == output2).all().item()
diff_matrix = (output1 - output2).abs()
max_diff = diff_matrix.max().item()

print("-" * 30)
print(f"Time - Batch Mode: {time_batch:.6f} s")
print(f"Time - Loop Mode:  {time_loop:.6f} s")
print(f"Speedup Ratio:     {time_loop / time_batch:.2f}x")
print("-" * 30)
print(f"Strictly Equal:    {strict_equal}")

if not strict_equal:
    print(f"Max Absolute Difference: {max_diff:.10e}")
else:
    print("Strictly Equal.")
print("-" * 30)

# Save Results to File
saved_data = {
    "outputs": {
        "output_batch": output1.detach().cpu(),
        "output_loop": output2.detach().cpu(),
        "time_batch": time_batch,
        "time_loop": time_loop
    },
    "analysis": {
        "max_diff": max_diff,
        "strict_equal": strict_equal
    }
}

torch.save(saved_data, DATA_FILE)
script_total = time.perf_counter() - script_start
print(f"Total Execution Time (including sync): {script_total:.4f} s")