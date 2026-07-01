import torch
import math
import triton
import triton.language as tl
import os

# 确保使用 GPU
DEVICE = "cuda"

@triton.jit
def ffn_kernel(
    a_ptr,
    b_ptrt,
    output_ptr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    bs1: tl.constexpr,
    bs2: tl.constexpr,
    bs3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    pid_d1 = tl.program_id(0)
    pid_d3 = tl.program_id(1)

    offsets_d1 = tl.arange(0, bs1) + bs1 * pid_d1
    offsets_d3 = tl.arange(0, bs3) + bs3 * pid_d3
    t_val = tl.zeros((bs1, bs3), dtype=tl.float32)
    a_ptrs = a_ptr + (offsets_d1[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    b_ptrs = b_ptrt + (offsets_d3[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    for index in range(0, tl.cdiv(d2, BLOCK_SIZE)):
        mask_a = (offsets_d1[:, None] < d1) & (
            tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE
        )
        a_val = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_b = (offsets_d3[:, None] < d3) & (
            tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE
        )
        b_val = tl.load(b_ptrs, mask=mask_b, other=0.0)
        b_val = b_val.trans(1, 0)

        # a_val = tl.inline_asm_elementwise(ASM, "=r, r", [a_val], dtype=tl.float32, is_pure=True, pack=1)
        # b_val = tl.inline_asm_elementwise(ASM, "=r, r", [b_val], dtype=tl.float32, is_pure=True, pack=1)

        t_val += tl.dot(a_val, b_val)

        a_ptrs += BLOCK_SIZE
        b_ptrs += BLOCK_SIZE

    output_ptrs = output_ptr + (
        offsets_d1[:, None] * d3 + tl.arange(0, bs3)[None, :] + pid_d3 * bs3
    )
    tl.store(
        output_ptrs,
        t_val,
        mask=(offsets_d1[:, None] < d1) & (tl.arange(0, bs3) < d3 - pid_d3 * bs3),
    )


def matrix_multiplication(a, b):
    """
    a: d1 x d2
    b: d3 x d2
    """
    block_size = 16
    bs1 = block_size
    bs2 = block_size
    bs3 = block_size
    bs_hidden = block_size
    (d1, d2) = a.shape
    d3 = b.shape[0]
    output = torch.zeros((d1, d3), dtype=a.dtype, device=a.device)
    grid = lambda META: (triton.cdiv(d1, bs1), triton.cdiv(d3, bs3))
    ffn_kernel[grid](a, b, output, d1, d2, d3, bs1, bs2, bs3, BLOCK_SIZE=bs_hidden)
    return output


def torch_func(a, b):
    output = torch.mm(a, torch.t(b))
    return output

# ==========================================
# 辅助函数：模拟 TF32 截断
# ==========================================
def truncate_to_tf32(tensor):
    """
    将 FP32 Tensor 截断为 TF32 精度。
    原理：保留高 19 位 (1 符号 + 8 指数 + 10 尾数)，清零低 13 位。
    Mask: 0xFFFFE000
    """
    # 确保 tensor 是 float32 类型
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # 转换为 int32 视图以便进行位操作
    tensor_int = tensor.view(torch.int32)
    
    # 创建掩码：1111 1111 1111 1111 1110 0000 0000 0000 (Hex: FFFFE000)
    mask = 0xFFFFE000
    
    # 应用掩码
    tensor_truncated_int = tensor_int & mask
    
    # 转回 float32 视图
    return tensor_truncated_int.view(torch.float32)

# ==========================================
# 主程序
# ==========================================

M = 3
N = 2
dim_hidden = 4
torch.manual_seed(91356303)

# 1. 加载数据
if os.path.exists("inputs.pt"):
    inputs = torch.load("inputs.pt", weights_only=True)
    a = inputs["a"].to(DEVICE)
    b = inputs["b"].to(DEVICE)
else:
    print("Warning: inputs.pt not found, generating random data.")
    a = torch.randn((M, N), dtype=torch.float32, device=DEVICE)
    b = torch.randn((dim_hidden, N), dtype=torch.float32, device=DEVICE)

print("-" * 40)
print("Applying TF32 Truncation Simulation...")
print("-" * 40)

# 2. 应用 TF32 截断
a = truncate_to_tf32(a)
b = truncate_to_tf32(b)

# 3. 运行计算
output_torch = torch_func(a, b)
output_triton = matrix_multiplication(a, b)

print(f"a (TF32 Truncated): \n", a)
print(f"b (TF32 Truncated): \n", b)
print(f"output_torch: \n", output_torch)
print(f"output_triton: \n", output_triton)

diff = torch.max(torch.abs(output_triton - output_torch)).item()
print(f"\nThe torch difference is {diff}")

# 4. 保存结果
with open("reference.pt", "wb") as f:
    output = {
        "torch_output": output_torch.detach(),
        "triton_output": output_triton,
        "a_truncated": a,
        "b_truncated": b
    }
    torch.save(output, f)