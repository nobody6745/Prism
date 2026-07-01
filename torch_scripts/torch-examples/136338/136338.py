import torch
import os
import time 

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "batch_diff_data.pt")

script_start = time.perf_counter()

torch.manual_seed(0)

units = 1000
x = (torch.rand(1, units)).repeat(2, 1)
weights = torch.rand(units, units)
t0 = time.perf_counter()
result_single = torch.nn.functional.linear(x[:1], weights)
duration_b1 = time.perf_counter() - t0
t1 = time.perf_counter()
result_double = torch.nn.functional.linear(x[:2], weights)
duration_b2 = time.perf_counter() - t1

is_strict_equal = result_single[0].equal(result_double[0])
max_diff = (result_single[0] - result_double[0]).abs().max()

print(f"Single[0] == Double[0] (Strict)? {is_strict_equal}")
print(f"Max Absolute Difference: {max_diff.item():.10e}")
print("-" * 40)

saved_data = {
    "outputs": {
        "result_batch_1": result_single,
        "result_batch_2": result_double
    },
    "timings": {
        "duration_b1": duration_b1,
        "duration_b2": duration_b2
    },
    "analysis": {
        "max_diff": max_diff.item(),
        "is_strict_equal": is_strict_equal
    }
}
torch.save(saved_data, DATA_FILE)

script_total = time.perf_counter() - script_start
print(f"time: {script_total:.4f} s")