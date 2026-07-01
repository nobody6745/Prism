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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(seed)
saved_results = {}

x = torch.randn(batch_size, feature_size, input_size, device="cuda")

def test_precision(dtype, tag_prefix, linear_layer):
    linear_layer.to(dtype)
    x_dtype = x.to(dtype)

    _ = linear_layer(x_dtype)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    a1 = linear_layer(x_dtype)[:1]
    weight_f32 = linear_layer.weight.float()
    bias_f32 = linear_layer.bias.float() if linear_layer.bias is not None else None
    input_f32 = x_dtype[:1].float()
    a2 = F.linear(input_f32, weight_f32, bias_f32)

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    diff_tensor = (a1.float() - a2).abs()
    max_diff = diff_tensor.max().item()
    avg_diff = diff_tensor.mean().item()
    
    print(f"[{dtype}] Max Diff: {max_diff:.6e}, Avg Diff: {avg_diff:.6e}")
    key_name = f"{tag_prefix}_{str(dtype)}"
    saved_results[key_name] = {
        "dtype": str(dtype),
        "a1": a1.detach().cpu(),
        "a2": a2.detach().cpu(),
        "max_diff": max_diff,
        "avg_diff": avg_diff,
        "time_s": t_end - t_start
    }

    return t_end - t_start

for has_bias in [False, True]:
    tag = "with_bias" if has_bias else "no_bias"
    print(f"\n Running experiment: Linear {tag}")
    
    linear = torch.nn.Linear(input_size, output_size, bias=has_bias).cuda()
    
    for dt in [torch.bfloat16, torch.float16, torch.float32]:
        duration = test_precision(dt, tag, linear)
        print(f"Time - {str(dt):<15}: {duration:.6f} s")
output_filename = "outputs.pt"
torch.save(saved_results, output_filename)

script_total = time.perf_counter() - script_start

print(f"time : {script_total:.4f} 秒")
