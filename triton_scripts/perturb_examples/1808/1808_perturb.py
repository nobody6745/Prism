import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def _kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K * SPLIT_K):
        a = tl.load(A, mask=rk[None, :] < k, other=0.0)
        b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # -----------------------------------------------------------
    # 核心修改：在 Kernel 内部向负无穷方向扰动 1 ULP (针对 fp16)
    # -----------------------------------------------------------
    # 对于 float16，其尾数有 10 位，ulp 约等于当前量级的 2^-11
    # 我们在 float32 空间计算偏移，确保减法操作能跨越 fp16 的最小表示位
    eps = 4*0.00048828125  # 即 2^-11
    
    # 计算当前 acc 在 fp16 下的量级
    acc_fp16_val = acc.to(tl.float16).to(tl.float32)
    # 构造扰动：当前值的 2^-11 加上一个微小的常数防止零值附近失效
    perturbation = tl.abs(acc_fp16_val) * eps + 1e-8
    
    # 向负无穷方向移动（即减去偏移）
    acc = (acc - perturbation).to(C.dtype.element_ty)
    # -----------------------------------------------------------

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    @staticmethod
    def _call(a, b):
        device = a.device
        if a.stride(0) > 1 and a.stride(1) > 1: a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1: b = b.contiguous()
        
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )
        
        _kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            GROUP_M=8,       
            ACC_TYPE=ACC_TYPE,
            BLOCK_M=128,      # 减小 Block Size 增加运算/调度次数
            BLOCK_N=128,      
            BLOCK_K=32,      
            SPLIT_K=4,       
        )
        return c

    @staticmethod
    def forward(ctx, a, b):
        return _matmul._call(a, b)

matmul = _matmul.apply

# ------------------------------------------------------------
# 测试与保存
# ------------------------------------------------------------
M, N, K = 2048, 2048, 2048
# 假设 inputs.pt 存在
try:
    inputs = torch.load("inputs.pt", weights_only=True)
    a = inputs["a"].cuda()
    b = inputs["b"].cuda()
except:
    print("Warning: inputs.pt not found, using random data.")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

# compute torch
torch_output = torch.matmul(a, b)
# compute triton (with kernel-side perturbation)
triton_output = matmul(a, b)

# compare
print(f"Triton Sample: {triton_output[16][219]:.6e}")
print(f"Torch  Sample: {torch_output[16][219]:.6e}")
print("max diff:", (torch_output - triton_output).abs().max())

if torch.allclose(triton_output, torch_output, atol=1e-5, rtol=0):
    print("✅ Triton and Torch match (Perturbation might be too small to break allclose)")
else:
    print("❌ Triton and Torch differ (Perturbation applied)")
    triton_output_np = triton_output.cpu().numpy()
    np.save("perturb.npy", triton_output_np)
    print("\nTriton output with kernel-side -1 ULP has been saved to 'perturb.npy'")