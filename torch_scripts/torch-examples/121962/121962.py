import torch
import torch.nn.functional as F
import os
import time 

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "linear_split_data.pt")

script_start = time.perf_counter()

torch.manual_seed(1234)
torch.set_printoptions(precision=10)

inputs = torch.randn(2, 6, dtype=torch.float16)

if inputs.is_cuda: torch.cuda.synchronize()
t_full_start = time.perf_counter()

outputs_full = F.linear(inputs, inputs)

if inputs.is_cuda: torch.cuda.synchronize()
duration_full = time.perf_counter() - t_full_start

if inputs.is_cuda: torch.cuda.synchronize()
t_split_start = time.perf_counter()

input1, input2 = inputs.split(3, dim=1)
weight1, weight2 = inputs.split(3, dim=1)
dist_output = F.linear(input1, weight1) + F.linear(input2, weight2)

if inputs.is_cuda: torch.cuda.synchronize()
duration_split = time.perf_counter() - t_split_start

is_equal = (dist_output == outputs_full).all()
diff_matrix = (dist_output - outputs_full).abs()
max_diff = diff_matrix.max()

print("-" * 30)
print(f"Time - Full Computation:  {duration_full:.6f} s")
print(f"Time - Split Computation: {duration_split:.6f} s")
print("-" * 30)
print(f"Strict Equality: {is_equal}")
print(f"Max Absolute Difference: {max_diff.item():.10e}")
print("-" * 30)

saved_data = {
    "outputs": {
        "full": outputs_full,
        "distributed": dist_output
    },
    "analysis": {
        "max_diff": max_diff.item(),
        "is_equal": is_equal,
        "time_full": duration_full,
        "time_split": duration_split
    }
}

torch.save(saved_data, DATA_FILE)

script_total = time.perf_counter() - script_start
print(f"time: {script_total:.4f} s")