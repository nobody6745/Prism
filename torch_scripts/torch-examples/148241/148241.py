import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config
import time
import os

config.fallback_random = True
torch.set_grad_enabled(False)
torch.manual_seed(0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.shrink = nn.Tanhshrink()

    def forward(self, x):
        x = self.shrink(x)
        x = torch.atan2(x, x)
        return x


model = Model()

x = torch.randn(1, 3, 64, 64)
inputs = [x]


def run_test(model, inputs, backend):
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    torch.manual_seed(0)
    output = model(*inputs)
    return output


t0 = time.perf_counter()
output = run_test(model, inputs, 'eager')
t1 = time.perf_counter()
t2 = time.perf_counter()
c_output = run_test(model, inputs, 'inductor')
t3 = time.perf_counter()

print(torch.allclose(output, c_output, 1e-3, 1e-3, equal_nan=True))
print(torch.max(torch.abs(output - c_output)))

print("\n====== Runtime ======")
print(f"Eager runtime:    {t1 - t0:.6f} s")
print(f"Inductor runtime: {t3 - t2:.6f} s")
print(f"Total runtime:    {t3 - t0:.6f} s")