import torch
import numpy as np
import time  

np_array = np.array([[[6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5]],
              
                     [[6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5]],
            
                     [[6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5],
                      [6.5, 6.5, 6.5, 6.5, 6.5]]], dtype=np.float16)

script_start = time.perf_counter()

input_cpu = torch.from_numpy(np_array)

input_gpu = input_cpu.cuda()
dim = 1

t0 = time.perf_counter()
out_cpu = torch.cumprod(input_cpu, dim)
duration_cpu = time.perf_counter() - t0
_ = torch.cumprod(input_gpu, dim)
torch.cuda.synchronize()

t1 = time.perf_counter()
out_gpu = torch.cumprod(input_gpu, dim)
torch.cuda.synchronize()  
duration_gpu = time.perf_counter() - t1

max_diff = (out_cpu - out_gpu.cpu()).abs().max()
print(f"max diff: {max_diff.item():.10e}")

results = {
    "input_np": np_array,
    "dim": dim,
    "out_cpu": out_cpu,
    "out_gpu": out_gpu.cpu(),
    "timings": {"cpu": duration_cpu, "gpu": duration_gpu}
}

output_file = "cumprod_results.pt"
torch.save(results, output_file)
print(f"\ntime: {time.perf_counter() - script_start:.4f} s")