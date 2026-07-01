import torch
import random
import torch.nn as nn

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

x_f32 = x.float()
z_f32 = z.float()

weight_f32 = net.weight.data.float()
bias_f32 = net.bias.data.float()


net_f32 = nn.Linear(768, 768).to('cuda:0', dtype=torch.float32)
net_f32.weight.data = weight_f32
net_f32.bias.data = bias_f32
net_f32.eval()

print('\n nn.Linear Comparisons')
with torch.no_grad():
    a = net(x)          
    b = net_f32(z_f32)  
diff_ab = a - b[:K]

print(f"a (FP16) shape: {a.shape}")
print(f"b (FP32) [:K] shape: {b[:K].shape}")
print(f"sum(|a - b[:K]|) = {diff_ab.abs().sum().item()}")

print("\n--- a[0, :10] (FP16) ---")
print(a[0, :10])
print("\n--- b[0, :10] (FP32 Truth) ---")
print(b[0, :10])
print("\n--- (a - b[:K])[0, :10] (Error) ---")
print(diff_ab[0, :10])

print('\n Manual GEMM Comparisons')
with torch.no_grad():
    c = x @ net.weight.data.t() + net.bias.data.view(1, -1)
    d = z_f32 @ weight_f32.t() + bias_f32.view(1, -1)

diff_cd = c - d[:K]

print(f"c (FP16) shape: {c.shape}")
print(f"d (FP32) [:K] shape: {d[:K].shape}")
print(f"sum(|c - d[:K]|) = {diff_cd.abs().sum().item()}")

print("\n--- c[0, :10] (FP16) ---")
print(c[0, :10])
print("\n--- d[0, :10] (FP32 Truth) ---")
print(d[0, :10])
print("\n--- (c - d[:K])[0, :10] (Error) ---")
print(diff_cd[0, :10])

save_dict = {
    "a": a.cpu(), 
    "b": b.cpu(), 
    "c": c.cpu(), 
    "d": d.cpu(),
    "diff_ab": diff_ab.cpu(),
    "diff_cd": diff_cd.cpu(),
}

torch.save(save_dict, "linear_vs_gemm_debug.pt")
print("\nSaved to linear_vs_gemm_debug.pt")