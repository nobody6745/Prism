import torch
import triton.language as tl
import triton
import numpy as np

torch.manual_seed(0)

N = 64
inputs = torch.load("inputs.pt", weights_only=True)
A = inputs["A"]
B = inputs["B"]

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)[:, None] * N + tl.arange(0, N)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = tl.dot(a, b)
    tl.store(c_ptr + offs, c)

def matmul(a, b):
    N = a.shape[0]
    c = torch.empty((N, N), dtype=a.dtype, device=a.device)
    grid = lambda META: (1, 1)
    matmul_kernel[grid](a, b, c, N)
    return c


C_triton_tf32 = matmul(A, B)

triton_output_np = C_triton_tf32.cpu().numpy()
np.save("perturb.npy", triton_output_np)
print("\nTriton output has been successfully saved to 'perturb.npy'")