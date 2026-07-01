import triton
import triton.language as tl
import torch
import numpy as np
import os


@triton.jit
def apply_fp32_perturbation(val):
    val_i32 = val.to(tl.int32, bitcast=True)

    step = 1
    
    is_pos = val >= 0.0
    delta = tl.where(is_pos, -step, step)
    
    res_i32 = val_i32 + delta
    return res_i32.to(tl.float32, bitcast=True)


@triton.jit
def test_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_size,
    STORE_FLAG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_PERTURB: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_size
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    
    out = x * y
    if ENABLE_PERTURB:
        out = apply_fp32_perturbation(out)

    if STORE_FLAG:
        tl.store(out_ptr + offset, out, mask=mask)
        out = tl.load(out_ptr + offset, mask=mask)
    out = out + x

    if ENABLE_PERTURB:
        out = apply_fp32_perturbation(out)

    tl.store(out_ptr + offset, out, mask=mask)

if os.path.exists("inputs.pt"):
    inputs = torch.load("inputs.pt", weights_only=True)
    x = inputs["x"]
    y = inputs["y"]
else:
    print("Warning: inputs.pt not found, using random data.")
    torch.manual_seed(0)
    x = torch.randn((100, 100), dtype=torch.float32, device="cuda")
    y = torch.randn((100, 100), dtype=torch.float32, device="cuda")

n_size = x.numel()

ref = x * y + x

out2 = torch.zeros_like(x)

grid = lambda meta: (triton.cdiv(n_size, meta["BLOCK_SIZE"]),)
print("Running Triton Kernel (Result Perturbation, FP32 1 ULP)...")
test_kernel[grid](
    x, y, out2, n_size, 
    STORE_FLAG=False, 
    BLOCK_SIZE=1024,
    ENABLE_PERTURB=True  
)

max_diff = (ref - out2).abs().max().item()
mean_diff = (ref - out2).abs().mean().item()
allclose = torch.allclose(ref, out2)

print(f"Max Diff : {max_diff}")
print(f"Mean Diff: {mean_diff}")
print(f"All Close: {allclose}")
