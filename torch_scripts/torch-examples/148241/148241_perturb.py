import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config
import time

# Inductor configuration for numerical consistency
config.fallback_random = True
torch.set_grad_enabled(False)
torch.manual_seed(0)

def move_ulp(tensor, toward_pos_inf=True):
    """Perturbs the tensor by 1 ULP (Unit in the Last Place) in the specified direction."""
    target = float('inf') if toward_pos_inf else float('-inf')
    return torch.nextafter(tensor, torch.tensor(target, device=tensor.device, dtype=tensor.dtype))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Original computational path
        tanh_x_orig = torch.tanh(x)
        shrink_out_orig = x - tanh_x_orig
        
        pos_mask = x > 0
        neg_mask = x < 0
        
        # Construct perturbed input (x_pert) by moving 1 ULP toward zero
        x_pert = torch.clone(x)
        x_pert[pos_mask] = move_ulp(x[pos_mask], toward_pos_inf=False)
        x_pert[neg_mask] = move_ulp(x[neg_mask], toward_pos_inf=True)
        
        # Construct perturbed intermediate (tanh_x_pert) by moving 1 ULP away from zero
        tanh_x_pert = torch.tanh(x)
        tanh_x_pert[pos_mask] = move_ulp(tanh_x_pert[pos_mask], toward_pos_inf=True)
        tanh_x_pert[neg_mask] = move_ulp(tanh_x_pert[neg_mask], toward_pos_inf=False)
        
        # Perturbed computational path
        shrink_out_pert = x_pert - tanh_x_pert
        
        # Check for zero-crossings (sign flips) within the 1 ULP safety envelope
        zero_crossing = (shrink_out_orig * shrink_out_pert <= 0)
        num_crossings = torch.sum(zero_crossing).item()
        
        if num_crossings > 0:
            print(f"[PRISM Alert] Numerical instability detected: {num_crossings} elements "
                  f"experienced a sign flip under 1 ULP perturbation!")
        
        # Final operation (Atan2 is sensitive to inputs near zero)
        x_final = torch.atan2(shrink_out_pert, shrink_out_pert)
        return x_final

model = Model()
x = torch.randn(1, 3, 64, 64)
inputs = [x]

def run_test(model, inputs, backend):
    """Executes the model using either the Eager or Compiled backend."""
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    torch.manual_seed(0)
    return model(*inputs)

# Baseline execution
output_eager = run_test(model, inputs, 'eager')