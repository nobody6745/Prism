import torch
import time
import platform

script_start = time.perf_counter()

print(f"PyTorch Version: {torch.__version__}")
print(f"Platform: {platform.platform()}")
print("-" * 30)

# Initialize vector
t0 = time.perf_counter()
v1 = torch.randn(35429888)
if torch.cuda.is_available(): torch.cuda.synchronize()

# --- FP32 Precision Benchmark ---
t1 = time.perf_counter()
res_fp32 = v1.norm()
if torch.cuda.is_available(): torch.cuda.synchronize()
print(f"FP32 Norm: {res_fp32.item():.4f} | Duration: {time.perf_counter() - t1:.4f} s")

# --- FP64 Precision Benchmark ---
t2 = time.perf_counter()
res_fp64 = v1.double().norm()
if torch.cuda.is_available(): torch.cuda.synchronize()
print(f"FP64 Norm: {res_fp64.item():.4f} | Duration: {time.perf_counter() - t2:.4f} s")

# --- FP16 Precision Benchmark ---
t3 = time.perf_counter()
res_fp16 = v1.half().norm()
if torch.cuda.is_available(): torch.cuda.synchronize()
print(f"FP16 Norm: {res_fp16.item():.4f} | Duration: {time.perf_counter() - t3:.4f} s")

print("-" * 30)
print(f"Total script execution time: {time.perf_counter() - script_start:.4f} s")