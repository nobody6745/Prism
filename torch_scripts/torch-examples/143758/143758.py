import torch
from torch import nn
from torch.nn import functional as F
import time  


torch.manual_seed(0)

print(" Starting Linear Layer vs Manual Matmul performance test...")

lin_layer = nn.Linear(200, 300)
input_tensor = torch.randn(1, 400, 200)


t0 = time.perf_counter()
out_lin_layer = lin_layer(input_tensor)
duration_layer = time.perf_counter() - t0

t1 = time.perf_counter()

out_manual = input_tensor @ lin_layer.weight.data.t() + lin_layer.bias.data
duration_manual = time.perf_counter() - t1

print(f" Linear Layer Time: {duration_layer:.8f} s")
print(f" Manual Matmul Time: {duration_manual:.8f} s")

diff = out_lin_layer - out_manual
abs_diff = diff.abs()

print("\n Numeric difference stats (layer vs manual)")
print("Max absolute difference:  ", abs_diff.max().item())
print("Mean absolute difference: ", abs_diff.mean().item())

save_path = "linear_vs_manual.pt"
save_dict = {
    "out_lin_layer": out_lin_layer,
    "out_manual": out_manual,
    "timings": {
        "layer_duration": duration_layer,
        "manual_duration": duration_manual
    }
}

torch.save(save_dict, save_path)