import torch
import time
import random
import numpy as np
import torch.nn.functional as F

input_size = 1024
output_size = 512
feature_size = 100
batch_size = 2
seed = 1234

script_start = time.perf_counter()

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

saved_results = {}
x = torch.randn(batch_size, feature_size, input_size, device="cuda")

def test_precision(dtype, tag_prefix):
    linear.to(dtype)
    x_dtype = x.to(dtype)

    _ = linear(x_dtype)
    torch.cuda.synchronize()

    t_start = time.perf_counter()

    a1 = linear(x_dtype)[:1]
    
    weight_f32 = linear.weight.float()
    bias_f32 = linear.bias.float() if linear.bias is not None else None
    input_f32 = x_dtype[:1].float()
    a2 = F.linear(input_f32, weight_f32, bias_f32)

    torch.cuda.synchronize() 
    t_end = time.perf_counter()
    duration = t_end - t_start
    diff_tensor = (a1.float() - a2).abs()
    max_diff = diff_tensor.max().item()
    sum_diff = diff_tensor.sum().item()
    
    print(f"[{dtype}] Max Diff: {max_diff:.6e} | Sum Diff: {sum_diff:.6e}")

    key_name = f"{tag_prefix}_{str(dtype)}"
    saved_results[key_name] = {
        "dtype": str(dtype),
        "a1_perturb": a1.detach().cpu(),
        "max_diff": max_diff,
        "duration": duration
    }

    return duration

linear = torch.nn.Linear(input_size, output_size, bias=False).cuda()
print(f"\n Linear WITHOUT Bias")
times_no_bias = {
    "bf16": test_precision(torch.bfloat16, "no_bias"),
    "fp16": test_precision(torch.float16, "no_bias"),
    "fp32": test_precision(torch.float32, "no_bias")
}


linear = torch.nn.Linear(input_size, output_size, bias=True).cuda()
print(f"\n Linear WITH Bias")
times_with_bias = {
    "bf16": test_precision(torch.bfloat16, "with_bias"),
    "fp16": test_precision(torch.float16, "with_bias"),
    "fp32": test_precision(torch.float32, "with_bias")
}

output_filename = "outputs_perturb.pt"
torch.save(saved_results, output_filename)

script_total = time.perf_counter() - script_start
print(f"time: {script_total:.4f} 秒")