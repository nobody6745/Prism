import torch
import time  

script_start = time.perf_counter()

SEED = 1234

torch.manual_seed(SEED)

x = torch.randn(1000)
y = torch.randn(1000)

t0 = time.perf_counter()
result = x * y
diff = torch.addcmul(-result, x, y)
f32_duration = time.perf_counter() - t0
print(f"diff_cpu: {diff.abs().max().item()}")

t1 = time.perf_counter()
xDouble = x.double()
yDouble = y.double()

diffDouble = xDouble * yDouble - result.double()
f64_duration = time.perf_counter() - t1
print(f"diff_double (Max Abs Error): {diffDouble.abs().max().item()}")


print("-" * 40)
print(f"Float32 Path Time: {f32_duration:.6f} s")
print(f"Float64 Path Time: {f64_duration:.6f} s")
print(f"Precision Overhead: {f64_duration / f32_duration:.2f}x")
print("-" * 40)

save_dict = {
    "x": x,
    "y": y,
    "result_f32": result,
    "diff_double": diffDouble,
    "timings": {
        "f32": f32_duration,
        "f64": f64_duration
    }
}

torch.save(save_dict, "diff_analysis_data.pt")
print(f"Saved to diff_analysis_data.pt")

total_duration = time.perf_counter() - script_start
print(f"Total script execution time: {total_duration:.4f} s")