import random
from typing import Any
import os

import torch
from torch import nn
import numpy as np

import triton
import triton.language as tl
from triton.runtime import driver

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(42)

def cdiv(x: int, y: int):
    return (x + y - 1) // y

@triton.jit
def perturb_k_ulp_neg(x, k):
    i_x = x.to(tl.int32, bitcast=True)
    delta = tl.where(i_x > 0, -k, k)
    res_int = i_x + delta
    neg_tiny_bits = -2147483648 + k 
    res_int = tl.where(i_x == 0, neg_tiny_bits, res_int)
    return res_int.to(tl.float32, bitcast=True)

@triton.jit
def _flash_ffn_second_half_kernel(
    X2_ptr, Y2_ptr, W3_ptr, Out_ptr,
    # [NEW] Debug Pointers
    X3_DBG_ptr, SIG_X2_DBG_ptr,               
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offset_m < M
    offset_k = tl.arange(0, K) 

    acc = tl.zeros([BLOCK_M, K], dtype=tl.float32)

    num_n_blocks = tl.cdiv(N, BLOCK_N)
    
    for block_idx in range(num_n_blocks):
        offset_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        
        x2_ptrs = X2_ptr + offset_m[:, None] * N + offset_n[None, :]
        y2_ptrs = Y2_ptr + offset_m[:, None] * N + offset_n[None, :]
        
        load_mask = mask_m[:, None]
        
        x2 = tl.load(x2_ptrs, mask=load_mask, other=0.0)
        y2 = tl.load(y2_ptrs, mask=load_mask, other=0.0)

        sig_x2 = tl.sigmoid(x2)
        sig_x2 = perturb_k_ulp_neg(sig_x2, 6)          
        x3 = x2 * sig_x2 * y2
        
        debug_ptrs = offset_m[:, None] * N + offset_n[None, :]
        tl.store(SIG_X2_DBG_ptr + debug_ptrs, sig_x2.to(SIG_X2_DBG_ptr.dtype.element_ty), mask=load_mask)
        tl.store(X3_DBG_ptr + debug_ptrs, x3.to(X3_DBG_ptr.dtype.element_ty), mask=load_mask)

        w3_ptrs = W3_ptr + offset_n[:, None] * K + offset_k[None, :]
        w3 = tl.load(w3_ptrs)

        dot_update = tl.dot(x3, w3)
        acc += dot_update

    out_ptrs = Out_ptr + offset_m[:, None] * K + offset_k[None, :]
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])

class FlashFFNSecondHalfFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, y2, w3):
        M, N = x2.shape
        K = w3.shape[1] 
        
        BLOCK_M = 16
        BLOCK_N = 32
        
        Out = torch.zeros((M, K), dtype=x2.dtype, device=x2.device)
        X3_DBG = torch.empty((M, N), dtype=x2.dtype, device=x2.device)
        SIG_X2_DBG = torch.empty((M, N), dtype=x2.dtype, device=x2.device)   

        grid = lambda META: (cdiv(M, META['BLOCK_M']),)
        
        _flash_ffn_second_half_kernel[grid](
            X2_ptr=x2, Y2_ptr=y2, W3_ptr=w3, Out_ptr=Out,
            X3_DBG_ptr=X3_DBG, SIG_X2_DBG_ptr=SIG_X2_DBG,           
            M=M, N=N, K=K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )
        
        return Out, X3_DBG, SIG_X2_DBG

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return (None,) * 3

flash_ffn_second_half = FlashFFNSecondHalfFunction.apply

class FlashFFN(torch.nn.Module):
    def __init__(self, d_model, d_inner, dtype=torch.float32, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        
        self.weight0 = torch.nn.Parameter(torch.ones(d_model, dtype=dtype)) 
        self.weight1 = torch.nn.Parameter(torch.empty(d_model, d_inner, dtype=dtype))
        self.weight2 = torch.nn.Parameter(torch.empty(d_model, d_inner, dtype=dtype))
        self.weight3 = torch.nn.Parameter(torch.empty(d_inner, d_model, dtype=dtype))
        
        torch.nn.init.normal_(self.weight1, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight2, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight3, mean=0.0, std=0.02)

# ==========================================
#  4. 主程序
# ==========================================
device = torch.device("cuda")
dmodel = 128
dinner = dmodel // 2

inputs = torch.randn(
    (10000, dmodel), dtype=torch.float32, device=device, requires_grad=True
)

ffn = FlashFFN(d_model=dmodel, d_inner=dinner).to(device)

w0 = ffn.weight0.data
w1 = ffn.weight1.data
w2 = ffn.weight2.data
w3 = ffn.weight3.data

print(f"Testing Shapes: M={inputs.shape[0]}, K={dmodel}, N={dinner}")
print("-" * 60)

with torch.no_grad():
    L_ref = torch.rsqrt(torch.sum(inputs * inputs, dim=-1) / dmodel)
    x1_torch = inputs * L_ref.view(-1, 1) * w0.view(1, -1)
    
    x2_torch = torch.matmul(x1_torch, w1)
    y2_torch = torch.matmul(x1_torch, w2)
    
    sig_x2_ref = torch.sigmoid(x2_torch)
    x3_ref = x2_torch * sig_x2_ref * y2_torch         
    out_ref = torch.matmul(x3_ref, w3)

print("Running Triton Kernel (Inputs: x2_torch, y2_torch)...")
out_tri, x3_tri, sig_x2_tri = flash_ffn_second_half(x2_torch, y2_torch, w3)   

def check_diff(name, t_ref, t_tri):
    diff = (t_ref.double() - t_tri.double()).abs()
    max_diff = diff.max().item()
    print(f"{name:<8} | Max Diff: {max_diff:.6e}")

check_diff("SIG_X2", sig_x2_ref, sig_x2_tri)  
check_diff("X3",     x3_ref,     x3_tri)
check_diff("Out",    out_ref,    out_tri)

print("-" * 60)

save_path = "perturb.npy"
np.save("perturb.npy", out_tri.detach().cpu())
'''save_data = {
    "inputs_from_torch": {
        "x2": x2_torch.detach().cpu(),
        "y2": y2_torch.detach().cpu(),
        "sig_x2": sig_x2_ref.detach().cpu(),           # <--- PyTorch 版
    },
    "triton_results": {
        "sig_x2": sig_x2_tri.detach().cpu(),           # <--- Triton 版
        "x3": x3_tri.detach().cpu(),
        "out": out_tri.detach().cpu(),
    }
}
'''
#torch.save(save_data, save_path)
#print(f"All tensors including SIG_X2 saved to {save_path}")