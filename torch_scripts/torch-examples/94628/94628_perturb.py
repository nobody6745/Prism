import torch
import time  

torch.manual_seed(42)
save_path = "distributive_perturb.pt"
save_data = {}

script_start = time.perf_counter()

configurations = [
    ('cuda:0', torch.float16), 
    ('cuda:0', torch.float32), 
    ('cpu', torch.float32)
]

for device_str, dtype in configurations:
    if 'cuda' in device_str and not torch.cuda.is_available():
        print(f"Skipping {device_str} (CUDA not available)")
        continue

    torch.manual_seed(42)
    if 'cuda' in device_str:
        torch.cuda.manual_seed(42)

    A = torch.randn(10_000, 10_000, device=device_str, dtype=dtype)
    B = torch.randn(10_000, device=device_str, dtype=dtype)
    C = torch.randn(10_000, device=device_str, dtype=dtype)
    if 'cuda' in device_str:
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    result_1 = A @ B + A @ C 
    
    result_2 = A @ (B + C)
    if 'cuda' in device_str:
        torch.cuda.synchronize()
        
    duration = time.perf_counter() - t0

    diff = result_1 - result_2
    diff_norm = diff.norm().item()
    is_close = torch.allclose(result_1, result_2, atol=1e-3)

    print(f'Device: {device_str}, dtype: {dtype}')
    print(f'Execution Time: {duration:.6f} seconds')
    print(f'Are all values close (atol=1e-3): {is_close}')
    print(f'Norm of difference: {diff_norm:.4f}')
    
    if dtype == torch.float16 and 'cuda' in device_str:
        save_data = {
            "result_AB_plus_AC_perturb": result_1.cpu(),
            "time_duration": duration
        }
        max_diff = diff.abs().max().item()
        print(f"  >>> FP16 Max Element-wise Diff: {max_diff:.4f}")

    print('=' * 30)


torch.save(save_data, save_path)
print(f"\n time: {time.perf_counter() - script_start:.4f}s")
