import torch
import triton.language as tl
import triton
import os

# 设置随机种子以保证可复现性
torch.manual_seed(0)

# 加载数据或生成随机数据
if os.path.exists("inputs.pt"):
    inputs = torch.load("inputs.pt", weights_only=True)
    A = inputs["A"]
    B = inputs["B"]
else:
    print("Warning: inputs.pt not found, generating random data.")
    N = 64
    A = torch.randn((N, N), device='cuda', dtype=torch.float32)
    B = torch.randn((N, N), device='cuda', dtype=torch.float32)

# ==========================================
# 新增：TF32 截断模拟函数
# ==========================================
def truncate_tf32(input_tensor):
    """
    模拟 TF32 截断行为：
    保留 FP32 的高 19 位 (1 符号位 + 8 指数位 + 10 尾数位)。
    将低 13 位清零。
    Mask: 0xFFFFE000
    """
    # 1. 确保是 float32
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.float()
        
    # 2. 转换为 int32 视图进行位操作
    # 注意：使用 .view() 不会复制数据，只是改变解释方式
    input_int = input_tensor.view(torch.int32)
    
    # 3. 创建掩码：保留高19位
    # Hex: FFFF E000 -> Binary: 1111 1111 1111 1111 1110 0000 0000 0000
    mask = 0xFFFFE000
    
    # 4. 应用掩码
    truncated_int = input_int & mask
    
    # 5. 转回 float32 视图
    return truncated_int.view(torch.float32)

# ==========================================
# 应用截断
# ==========================================
print("Applying TF32 Truncation to inputs A and B...")
A_truncated = truncate_tf32(A)
B_truncated = truncate_tf32(B)

# ==========================================
# Triton Kernel 定义
# ==========================================
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr, allow_tf32: tl.constexpr):
    offs = tl.arange(0, N)[:, None] * N + tl.arange(0, N)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    # allow_tf32=True 会允许 TensorCore 使用 TF32 累加
    c = tl.dot(a, b, allow_tf32=allow_tf32)
    tl.store(c_ptr + offs, c)

def matmul(a, b, allow_tf32):
    N = a.shape[0]
    c = torch.empty((N, N), dtype=a.dtype, device=a.device)
    grid = lambda META: (1, 1)
    matmul_kernel[grid](a, b, c, N, allow_tf32)
    return c

# ==========================================
# 计算与对比
# ==========================================

# 1. PyTorch 参考值 (使用截断后的输入)
# allow_tf32 控制的是内部计算是否再次使用 TF32 格式，
# 但由于我们已经手动截断了输入，这里的 allow_tf32=True 更符合全程 TF32 的模拟。
torch.backends.cuda.matmul.allow_tf32 = True
C_ref_tf32 = A_truncated @ B_truncated

# 2. Triton 计算 (使用截断后的输入)
# 开启 allow_tf32 以启用 Tensor Core
C_triton_tf32 = matmul(A_truncated, B_truncated, allow_tf32=True)

# 3. 结果分析
mean_diff = torch.abs(C_ref_tf32 - C_triton_tf32).mean()
max_diff = torch.abs(C_ref_tf32 - C_triton_tf32).max()

print(f"{mean_diff = }")
print(f"{max_diff = }")

# 4. 保存结果
with open("reference.pt", "wb") as f:
    output = {
        "torch_output": C_ref_tf32,
        "triton_output": C_triton_tf32,
        "A_truncated": A_truncated,
        "B_truncated": B_truncated
    }
    torch.save(output, f)
    print("Results saved to reference.pt")