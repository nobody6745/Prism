import torch
import math
import triton
import triton.language as tl
import numpy as np
import os

DEVICE = "cuda"

# Set print options for better readability
torch.set_printoptions(profile="full", linewidth=200, precision=4, sci_mode=False)

@triton.jit
def ffn_kernel(
    a_ptr,
    b_ptrt,
    output_ptr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    bs1: tl.constexpr,
    bs2: tl.constexpr,
    bs3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_d1 = tl.program_id(0)
    pid_d3 = tl.program_id(1)

    offsets_d1 = tl.arange(0, bs1) + bs1 * pid_d1
    offsets_d3 = tl.arange(0, bs3) + bs3 * pid_d3
    t_val = tl.zeros((bs1, bs3), dtype=tl.float32)
    
    # Pointers to matrices A and B
    a_ptrs = a_ptr + (offsets_d1[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    b_ptrs = b_ptrt + (offsets_d3[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    
    for index in range(0, tl.cdiv(d2, BLOCK_SIZE)):
        mask_a = (offsets_d1[:, None] < d1) & (
            tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE
        )
        a_val = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_b = (offsets_d3[:, None] < d3) & (
            tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE
        )
        b_val = tl.load(b_ptrs, mask=mask_b, other=0.0)
        b_val = b_val.trans(1, 0)

        t_val += tl.dot(a_val, b_val, input_precision="ieee")

        a_ptrs += BLOCK_SIZE
        b_ptrs += BLOCK_SIZE

    output_ptrs = output_ptr + (
        offsets_d1[:, None] * d3 + tl.arange(0, bs3)[None, :] + pid_d3 * bs3
    )
    tl.store(
        output_ptrs,
        t_val,
        mask=(offsets_d1[:, None] < d1) & (tl.arange(0, bs3) < d3 - pid_d3 * bs3),
    )

def matrix_multiplication(a, b):
    block_size = 16
    bs1 = block_size
    bs2 = block_size
    bs3 = block_size
    bs_hidden = block_size
    (d1, d2) = a.shape
    d3 = b.shape[0]
    output = torch.zeros((d1, d3), dtype=a.dtype, device=a.device)
    grid = lambda META: (triton.cdiv(d1, bs1), triton.cdiv(d3, bs3))
    ffn_kernel[grid](a, b, output, d1, d2, d3, bs1, bs2, bs3, BLOCK_SIZE=bs_hidden)
    return output

reference_path = "reference.pt"

print(f"Loading data from {reference_path}...")
if not os.path.exists(reference_path):
    raise FileNotFoundError(f"File not found: {reference_path}")

# 1. Load data
data = torch.load(reference_path, map_location=DEVICE)

if "a" in data and "b" in data:
    a = data["a"]
    b = data["b"]
elif "args" in data:
    a = data["args"][0]
    b = data["args"][1]
else:
    print("Warning: Input tensors not found in reference.pt, trying inputs.pt...")
    inputs = torch.load("inputs.pt", weights_only=True)
    a = inputs["a"]
    b = inputs["b"]

output_torch = data["torch_output"]
output_triton_from_file = data["triton_output"]

print("\n" + "="*50)
print("1. Input Matrices")
print("="*50)
print(f"Matrix A (Shape {a.shape}):\n{a}")
print(f"\nMatrix B (Shape {b.shape}):\n{b}")

print("\n" + "="*50)
print("2. Run Triton Kernel to Generate Perturbed Values")
print("="*50)
output_perturb = matrix_multiplication(a, b) 
print(f"Triton Runtime Output (Perturb):\n{output_perturb}")
print(f"\nTriton File Output (Ref):\n{output_triton_from_file}")

diff_triton = torch.max(torch.abs(output_triton_from_file - output_perturb)).item()
print(f"\n[Diff] Max Difference between File-Triton and Runtime-Triton: {diff_triton}")

if diff_triton == 0:
    print("⚠️ WARNING: The runtime Triton result is identical to the file result. The interval width will be 0.")

# Calculate the error bounds
bound_endpoint_1 = output_perturb
bound_endpoint_2 = 2 * output_triton_from_file - output_perturb

lower_bound = torch.min(bound_endpoint_1, bound_endpoint_2)
upper_bound = torch.max(bound_endpoint_1, bound_endpoint_2)

# Check if the PyTorch reference is within the computed Triton bounds
is_within_range = (output_torch >= lower_bound) & (output_torch <= upper_bound)

print("\n" + "="*80)
print("3. Detailed Analysis by Index")
print("="*80)
print(f"{'Index':<12} | {'State':<6} | {'Torch (Target)':<15} | {'Lower Bound':<15} | {'Upper Bound':<15} | {'Triton(File)':<15} | {'Triton(Run)':<15}")
print("-" * 110)

rows, cols = output_torch.shape
for r in range(rows):
    for c in range(cols):
        val_torch = output_torch[r, c].item()
        val_lower = lower_bound[r, c].item()
        val_upper = upper_bound[r, c].item()
        val_ref = output_triton_from_file[r, c].item()
        val_run = output_perturb[r, c].item()
        status = is_within_range[r, c].item()
        
        state_str = "✅ PASS" if status else "❌ FAIL"
        
        print(f"({r}, {c:<3})   | {state_str} | {val_torch:<15.10f} | {val_lower:<15.10f} | {val_upper:<15.10f} | {val_ref:<15.10f} | {val_run:<15.10f}")

print("\n" + "="*50)
print("4. Final Summary")
print("="*50)
total_elements = output_torch.numel()
within_count = is_within_range.sum().item()
percentage = (within_count / total_elements) * 100

print(f"Points within range: {within_count} / {total_elements} ({percentage:.2f}%)")

if within_count == total_elements:
    print("SUCCESS: Torch output is fully within the range.")
else:
    print("INFO: Some points are outside the range.")