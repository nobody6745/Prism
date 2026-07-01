import torch
import time

torch.manual_seed(0)

def f(input_data, device='cpu'):
    if device == 'cuda':
        torch.cuda.synchronize()  
    start_time = time.perf_counter()
    
    model = torch.nn.InstanceNorm3d(3, device=device)
    out = model(input_data)
    result = out.sum()
    
    if device == 'cuda':
        torch.cuda.synchronize() 
    end_time = time.perf_counter()
    
    return result, end_time - start_time

input_data = torch.rand(100, 3, 1, 10, 10)

inp_cpu = input_data.clone().cpu()
inp_cuda = input_data.clone().cuda()

res_cpu, time_cpu = f(inp_cpu)
print(f"CPU Result: {res_cpu:.6f} | Time: {time_cpu:.6f} s")

res_cuda, time_cuda = f(inp_cuda, 'cuda')
print(f"GPU Result: {res_cuda:.6f} | Time: {time_cuda:.6f} s")
print("-" * 30)
print(f"Speedup: {time_cpu / time_cuda:.2f}x")