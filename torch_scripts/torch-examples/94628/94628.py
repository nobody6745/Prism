import torch
import time


torch.manual_seed(42)
save_path = "distributive_law_fp16.pt"
save_data = {}

script_real_start = time.perf_counter()

configurations = [
    ('cuda:0', torch.float16), 
    ('cuda:0', torch.float32), 
    ('cpu', torch.float32)
]

for device_str, dtype in configurations:
    if 'cuda' in device_str and not torch.cuda.is_available():
        print(f" Skipping {device_str} (CUDA not available)")
        continue

    gen_start = time.perf_counter()
    torch.manual_seed(42)
    if 'cuda' in device_str:
        torch.cuda.manual_seed(42)

    A = torch.randn(10_000, 10_000, device=device_str, dtype=dtype)
    B = torch.randn(10_000, device=device_str, dtype=dtype)
    C = torch.randn(10_000, device=device_str, dtype=dtype)
    
    if 'cuda' in device_str:
        torch.cuda.synchronize()
    gen_duration = time.perf_counter() - gen_start

    if 'cuda' in device_str:
        torch.cuda.synchronize()
    
    calc_start = time.perf_counter()
 
    result_1 = A @ B + A @ C 
    result_2 = A @ (B + C)

    if 'cuda' in device_str:
        torch.cuda.synchronize()
    
    calc_duration = time.perf_counter() - calc_start
    diff = result_1 - result_2
    diff_norm = diff.norm().item()
    max_diff = diff.abs().max().item()
    is_close = torch.allclose(result_1, result_2, atol=1e-3)

    print(f' Device: {device_str} | Dtype: {dtype}')
    print(f'   - Data Generation Time: {gen_duration:.4f} s')
    print(f'   - Core Computation Time: {calc_duration:.4f} s')
    print(f'   - Satisfies Distributive Law (atol=1e-3): {is_close}')
    print(f'   - Difference Norm: {diff_norm:.4f}')
    print(f'   - Max Element-wise Difference: {max_diff:.4f}')

    if dtype == torch.float16 and 'cuda' in device_str:
        save_data = {
            "result_AB_plus_AC": result_1.cpu(),
            "result_A_BplusC": result_2.cpu(),
            "diff": diff.cpu(),
            "calc_time": calc_duration
        }

    print('=' * 40)

save_start = time.perf_counter()
if save_data:
    torch.save(save_data, save_path)
    save_duration = time.perf_counter() - save_start
    print(f"\n FP16 results saved to {save_path} (Save Time: {save_duration:.4f} s)")

script_total_duration = time.perf_counter() - script_real_start
print(f" Total Script Runtime: {script_total_duration:.4f} s")