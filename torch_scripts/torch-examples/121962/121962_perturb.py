import torch
import torch.nn.functional as F
import os
import time 

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "perturb.pt")

script_start = time.perf_counter()

torch.manual_seed(1234)
torch.set_printoptions(precision=10)

inputs = torch.randn(2, 6, dtype=torch.float16)

t_full_start = time.perf_counter()
outputs_full = F.linear(inputs, inputs)
t_full_end = time.perf_counter()
duration_full = t_full_end - t_full_start

t_split_start = time.perf_counter()
input1, input2 = inputs.split(3, dim=1)
weight1, weight2 = inputs.split(3, dim=1)
dist_output = F.linear(input1, weight1) + F.linear(input2, weight2)
t_split_end = time.perf_counter()
duration_split = t_split_end - t_split_start
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
    "inputs": {
        "original": inputs,
        "split_1": input1,
        "split_2": input2
    },
    "outputs": {
        "full_perturb": outputs_full,
        "distributed": dist_output
    },
    "meta": {
        "time_full": duration_full,
        "time_split": duration_split
    }
}


torch.save(saved_data, DATA_FILE)

script_end = time.perf_counter()
print(f"time: {script_end - script_start:.4f} s")