import torch
import torch.nn as nn
import os
import time  

# Configuration and Directory Setup
SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "perturb.pt")

torch.manual_seed(2024)

# Device Configuration
device = "cuda"
linear2 = nn.Linear(128, 550, bias=False).to(device)
input_tensor = torch.rand(32, 128).to(device)

# Performance Measurement Start
torch.cuda.synchronize()
start_all = time.perf_counter()
torch.cuda.synchronize()
t0 = time.perf_counter()

# Method 1: Batch Calculation
output1 = linear2(input_tensor)

torch.cuda.synchronize()
duration_batch = time.perf_counter() - t0
torch.cuda.synchronize()
t1 = time.perf_counter()

# Method 2: Loop-based Calculation (Row by Row)
out = []
for i in range(input_tensor.shape[0]):
    cc = linear2(input_tensor[i])
    out.append(cc)
output2 = torch.stack(out)

torch.cuda.synchronize()
duration_loop = time.perf_counter() - t1

# 1. Numerical Precision Analysis
strict_equal = (output1 == output2).all().item()
diff_matrix = (output1 - output2).abs()
max_diff = diff_matrix.max().item()

# 2. Print Performance Metrics
print("-" * 30)
print(f"Batch Execution Time: {duration_batch:.6f} s")
print(f"Loop Execution Time:  {duration_loop:.6f} s")
print(f"Speedup Ratio:        {duration_loop / duration_batch:.2f}x")
print("-" * 30)
print(f"Strictly Equal: {strict_equal}")
if not strict_equal:
    print(f"Max Absolute Difference: {max_diff:.10e}")
print("-" * 30)

# 3. Save Results
saved_data = {
    "outputs": {
        "output_batch_perturb": output1.detach().cpu(),
    },
    "timings": {
        "batch_time": duration_batch,
        "loop_time": duration_loop
    }
}

torch.save(saved_data, DATA_FILE)
end_all = time.perf_counter()
print(f"Data saved successfully. Total script execution time: {end_all - start_all:.4f} s")