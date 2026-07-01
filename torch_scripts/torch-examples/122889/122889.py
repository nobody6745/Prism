import torch
import os
import time  

# Configuration and Directory Setup
SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "linear_gemm_vs_gemv.pt")

script_start = time.perf_counter()

torch.manual_seed(2024)

# Initialize input tensor and linear layer
# x shape: [Batch, Seq_Len, Hidden_Dim]
x = torch.rand(1, 10, 20)
l = torch.nn.Linear(20, 5)

# --- Method 1: Full Sequence Calculation (GEMM-based) ---
t1_start = time.perf_counter()
x1 = l(x)
t1_end = time.perf_counter()
duration_x1 = t1_end - t1_start

# Extract the last token slice as the reference
x1_slice = x1[:, -1:, :]

# --- Method 2: Sliced Input Calculation (GEMV-based) ---
t2_start = time.perf_counter()
x2 = l(x[:, -1:, :])
t2_end = time.perf_counter()
duration_x2 = t2_end - t2_start

# Performance and Numerical Consistency Analysis
print("-" * 30)
print(f"Time - Full calculation (x1):   {duration_x1:.6f} s")
print(f"Time - Sliced calculation (x2): {duration_x2:.6f} s")
print("-" * 30)
print(f"x2 == x1 slice? {x2.eq(x1_slice).all().item()}")

diff_matrix = (x2 - x1_slice).abs()
max_diff = diff_matrix.max()
print(f"  -> Max difference: {max_diff.item():.10e}")

# Prepare Data for Saving
saved_data = {
    "outputs": {
        "x2_gemv": x2,
        "x1_slice_gemm": x1_slice,
        "time_x1": duration_x1,
        "time_x2": duration_x2
    },
    "analysis": {
        "max_diff": max_diff.item()
    }
}

print(f"\nSaving data to {DATA_FILE} ...")
torch.save(saved_data, DATA_FILE)

script_end = time.perf_counter()
print(f"Data saved successfully. Total script execution time: {script_end - script_start:.4f} s")