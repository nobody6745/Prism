import torch
import os
import time

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "matmul_diff.pt")

script_start = time.perf_counter()

torch.manual_seed(0)

weights = torch.rand(256, 256)
x = torch.randn(2, 5, 256)

t1_start = time.perf_counter()
y1 = torch.matmul(x, weights)
t1_end = time.perf_counter()
duration_y1 = t1_end - t1_start

t2_start = time.perf_counter()
y2 = torch.matmul(x[-1], weights)
t2_end = time.perf_counter()
duration_y2 = t2_end - t2_start

y1_slice = y1[-1]
is_equal = torch.equal(y1_slice, y2)
diff_matrix = (y1_slice - y2).abs()
max_diff = diff_matrix.max()

print("-" * 30)
print(f"Shape x: {x.shape}")
print(f"Shape weights: {weights.shape}")
print("-" * 30)
print(f"Execution Time (Batch Mode): {duration_y1:.6f} s")
print(f"Execution Time (Slice Mode): {duration_y2:.6f} s")
print("-" * 30)
print(f"Strictly Equal: {is_equal}")

if not is_equal:
    print(f"Not Equal!")
    print(f"   Max Absolute Difference: {max_diff.item():.10e}")
else:
    print("Strictly Equal")

saved_data = {
    "outputs": {
        "y1_slice": y1_slice,
        "y2_direct": y2,
        "time_y1": duration_y1,
        "time_y2": duration_y2
    },
    "analysis": {
        "max_diff": max_diff.item(),
        "is_equal": is_equal
    }
}

torch.save(saved_data, DATA_FILE)

script_end = time.perf_counter()
print(f"time: {script_end - script_start:.4f} s")