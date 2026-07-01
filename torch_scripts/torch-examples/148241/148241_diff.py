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

output_eager = run_test(model, inputs, 'eager')
output_inductor = run_test(model, inputs, 'inductor')

atol = 1e-3
rtol = 1e-3


abs_diff = torch.abs(output_eager - output_inductor)

mismatch_mask = abs_diff > (atol + rtol * torch.abs(output_inductor))

num_mismatches = torch.sum(mismatch_mask).item()
total_elements = output_eager.numel()
mismatch_ratio = (num_mismatches / total_elements) * 100

print(f"====== Numerical Consistency Analysis ======")
print(f"Total Elements:  {total_elements}")
print(f"Mismatched Pts:  {num_mismatches}")
print(f"Mismatch Ratio:  {mismatch_ratio:.4f}%")
print(f"Max Abs Diff:    {torch.max(abs_diff):.6e}")

if num_mismatches > 0:
    print(f"\nExample of Mismatched Values:")
    print(f"Eager Sample:    {output_eager[mismatch_mask][0]}")
    print(f"Inductor Sample: {output_inductor[mismatch_mask][0]}")