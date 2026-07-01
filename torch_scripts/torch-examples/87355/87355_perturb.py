import torch
import random
import torch.nn as nn
import time 

seed = 1234
dtype = torch.float16

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = False

net = nn.Linear(768, 768).to('cuda:0', dtype=dtype)
net.eval()

K = 10
N = 10
x = torch.randn(K, 768, device='cuda:0', dtype=dtype)
y = torch.randn(N, 768, device='cuda:0', dtype=dtype)
z = torch.cat([x, y], dim=0)

script_start = time.perf_counter()

print('\n nn.Linear')
with torch.no_grad():
    torch.cuda.synchronize()
    start_linear = time.perf_counter()
    
    a = net(x)       
    b = net(z)          
    
    torch.cuda.synchronize()
    end_linear = time.perf_counter()
    duration_linear = end_linear - start_linear

diff_ab = a - b[:K]
print(f"Time (nn.Linear): {duration_linear:.6f} s")
print(f"sum(|a - b[:K]|) = {diff_ab.abs().sum().item()}")

print('\n Manual GEMM (@ + bias)')
with torch.no_grad():
    torch.cuda.synchronize()
    start_gemm = time.perf_counter()
    
    c = x @ net.weight.data.t() + net.bias.data.view(1, -1)
    d = z @ net.weight.data.t() + net.bias.data.view(1, -1)
    
    torch.cuda.synchronize()
    end_gemm = time.perf_counter()
    duration_gemm = end_gemm - start_gemm

diff_cd = c - d[:K]
print(f"Time (Manual GEMM): {duration_gemm:.6f} s")
print(f"sum(|c - d[:K]|) = {diff_cd.abs().sum().item()}")

script_total_time = time.perf_counter() - script_start

save_dict = {
    "a_perturb": a.cpu(),
    "c_perturb": c.cpu(),
    "time_linear": duration_linear,
    "time_gemm": duration_gemm,
    "total_script_time": script_total_time
}

torch.save(save_dict, "linear_perturb.pt")
print(f"\nTotal script execution time: {script_total_time:.4f} s")
print("Saved to linear_perturb.pt")