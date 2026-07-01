import torch
import numpy as np
import time


script_start = time.perf_counter()


a = torch.tensor(0.00615, dtype=torch.float16)


t_cpu_start = time.perf_counter()
out_cpu = torch.digamma(a)


target_neg = torch.full_like(out_cpu, float('-inf'))
out_cpu_perturbed = torch.nextafter(out_cpu, target_neg)
duration_cpu = time.perf_counter() - t_cpu_start


if torch.cuda.is_available():
    a_cuda = a.cuda()
    torch.cuda.synchronize()
    
    t_gpu_start = time.perf_counter()
    out_gpu = torch.digamma(a_cuda)
    torch.cuda.synchronize()  
    duration_gpu = time.perf_counter() - t_gpu_start
-
    print(f"  CPU+Perturb Time: {duration_cpu:.8f} s")
    print(f"  GPU Compute Time: {duration_gpu:.8f} s")
    print(f"CPU (Original):  {out_cpu.item()}")
    print(f"CPU (Perturbed): {out_cpu_perturbed.item()}")
    print(f"GPU:             {out_gpu.item()}")

    try:
        np.testing.assert_allclose(out_cpu_perturbed.numpy(), out_gpu.cpu().numpy(), atol=0.01)
        print(" Assertion Passed! (CPU perturbed 1 ULP matches GPU)")
    except AssertionError as e:
        print(f" Assertion Failed:\n{e}")
else:
    print("need cuda")

script_total = time.perf_counter() - script_start
print(f"\n time: {script_total:.4f} s")