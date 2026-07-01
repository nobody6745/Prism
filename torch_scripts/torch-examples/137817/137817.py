import torch
import numpy as np
import time

a = torch.tensor(0.00615, dtype=torch.float16)

t0 = time.perf_counter()
out_cpu = torch.digamma(a)
duration_cpu = time.perf_counter() - t0

a_cuda = a.cuda()
torch.cuda.synchronize() 

t1 = time.perf_counter()
out_gpu = torch.digamma(a_cuda)
torch.cuda.synchronize() 
duration_gpu = time.perf_counter() - t1


print(f"  CPU 执行耗时: {duration_cpu:.8f} s")
print(f"  GPU 执行耗时: {duration_gpu:.8f} s")


np.testing.assert_allclose(out_cpu.numpy(), out_gpu.cpu().numpy(), atol=0.01)