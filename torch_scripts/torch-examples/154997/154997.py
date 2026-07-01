import torch
from torch import nn
import os
import time

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "linear_slice_data.pt")

script_start = time.perf_counter()
torch.random.manual_seed(200)

dtype = torch.float16
batch, seq, embed_dim, delay = 2, 16, 4, 2
device = "cuda"

x1 = torch.randn(batch, seq, embed_dim, device=device, dtype=dtype)
x2 = x1[:, delay:]

net = nn.Linear(embed_dim, embed_dim).to(device).to(dtype=dtype)

_ = net(x1)
torch.cuda.synchronize()
torch.cuda.synchronize()
t0 = time.perf_counter()

out1 = net(x1)
out1_slice = out1[:, delay:]

torch.cuda.synchronize()
duration_a = time.perf_counter() - t0

torch.cuda.synchronize()
t1 = time.perf_counter()

x2_f32 = x2.float()
weight_f32 = net.weight.float()
bias_f32 = net.bias.float() if net.bias is not None else None

out2_f32 = torch.nn.functional.linear(x2_f32, weight_f32, bias_f32)

out2 = out2_f32.to(dtype)

torch.cuda.synchronize()
duration_b = time.perf_counter() - t1
diff_matrix = (out1_slice.float() - out2.float()).abs()
max_diff = diff_matrix.max()

print(f"  Path A (FP16 Full) Time:   {duration_a:.8f} s")
print(f"  Path B (FP32 Direct) Time: {duration_b:.8f} s")
print(f" Max Difference (FP16 vs FP32_ref): {max_diff.item():.10e}")

is_close = torch.allclose(out1_slice, out2, atol=1e-3)
print(f" Result Match (atol=1e-3): {is_close}")
saved_data = {
    "outputs": {
        "out1_slice": out1_slice.detach().cpu(),
        "out2_direct": out2.detach().cpu()
    },
    "timings": {
        "duration_a": duration_a,
        "duration_b": duration_b
    },
    "diff": {
        "max_diff": max_diff.item()
    }
}

torch.save(saved_data, DATA_FILE)
print(f"time: {time.perf_counter() - script_start:.4f} s")