import torch
import torch._dynamo
import datetime
import time

# Suppress errors in TorchDynamo to prevent script interruption during compilation
torch._dynamo.config.suppress_errors = True
torch.set_default_device('cuda')
torch.set_printoptions(precision=10)

scale = torch.tensor(0.180336877703666687)
x = torch.tensor(1134139801600.000000)

def f(x, scale):
    # Testing numerical consistency: theoretically, (a - b) where a == b should be 0
    max_scaled = x * scale
    return torch.exp(max_scaled - x * scale)

# --- Eager Mode Execution ---
torch.cuda.synchronize()
t0 = time.perf_counter()
res_eager = f(x, scale)
torch.cuda.synchronize()
time_eager = time.perf_counter() - t0

# --- Torch.compile Setup ---
f_compiled = torch.compile(f)

# Warmup run (includes compilation time)
torch.cuda.synchronize()
t1 = time.perf_counter()
res_compiled_warmup = f_compiled(x, scale)
torch.cuda.synchronize()
time_compile_warmup = time.perf_counter() - t1

# Steady State run (actual execution performance)
torch.cuda.synchronize()
t2 = time.perf_counter()
res_compiled = f_compiled(x, scale)
torch.cuda.synchronize()
time_compile_steady = time.perf_counter() - t2

# --- Report Output ---
print("="*60)
print(f"Precision and Performance Benchmark ({res_eager.dtype})")
print("="*60)
print(f"Eager Mode Result      : {res_eager.item():.10f} | Duration: {time_eager:.6f}s")
print(f"Compile Mode (Warmup)  : {res_compiled_warmup.item():.10f} | Duration: {time_compile_warmup:.6f}s")
print(f"Compile Mode (Steady)  : {res_compiled.item():.10f} | Duration: {time_compile_steady:.6f}s")
print("-" * 60)
print(f"Speedup Ratio (Steady State): {time_eager / time_compile_steady:.2f}x")
print("="*60)

# --- Data Persistence ---
results_dict = {
    "eager_output": res_eager.cpu(),
    "compiled_output": res_compiled.cpu(),
    "timings": {
        "eager": time_eager,
        "compile_warmup": time_compile_warmup,
        "compile_steady": time_compile_steady
    },
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
torch.save(results_dict, "precision_test_results.pt")
print("Results and timing data saved to precision_test_results.pt")