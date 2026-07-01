import torch
import time  

# Seed set for reproducibility
torch.manual_seed(44)

prep_start = time.perf_counter()

# Initialize high-dimensional vector
v1 = torch.randn(35429888)

prep_duration = time.perf_counter() - prep_start

print(f"Original FP32 norm:   {v1.norm().item()}")
print("-" * 30)

def perturb_down_ulp(t):
    """Perturbs the tensor by 1 ULP toward negative infinity."""
    target = torch.full_like(t, float('-inf'))
    return torch.nextafter(t, target)

def custom_norm_mm_fp32_only(x):
    """Calculates L2 norm using matrix multiplication with ULP perturbations."""
    x_row = x.view(1, -1)
    x_col = x.view(-1, 1)

    # Compute dot product via MM
    s = torch.mm(x_row, x_col)
    s = perturb_down_ulp(s)
    
    # Compute Square Root
    res = torch.sqrt(s)
    res = perturb_down_ulp(res)

    return res.squeeze()

# Synchronize if using GPU to ensure accurate timing
if v1.is_cuda:
    torch.cuda.synchronize()

print("Starting custom Norm calculation (with ULP perturbation)...")
calc_start = time.perf_counter()

res_fp32 = custom_norm_mm_fp32_only(v1)

if v1.is_cuda:
    torch.cuda.synchronize()
calc_duration = time.perf_counter() - calc_start

print(f"FP32 (Perturbed):     {res_fp32.item()}")
print("-" * 30)
print(f"Core Calculation Duration: {calc_duration:.6f} seconds")
print(f"Throughput: {v1.numel() / calc_duration / 1e6:.2f} M elements/s")