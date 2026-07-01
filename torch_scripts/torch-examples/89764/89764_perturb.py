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
duration_f32 = time.perf_counter() - t0
print(f"diff_cpu: {diff.abs().max().item()}")

t1 = time.perf_counter()
xDouble = x.double()
yDouble = y.double()

def perturb_down_fp32(val_double):
    val_float = val_double.float()
    val_next_float = torch.nextafter(val_float, torch.tensor(-float('inf'), device=val_float.device))
    ulp_gap = (val_float - val_next_float).abs().double()
    return val_double - ulp_gap
prod_double = xDouble * yDouble
prod_perturbed = perturb_down_fp32(prod_double)

raw_diff = prod_perturbed - result.double()
diffDouble_perturbed = perturb_down_fp32(raw_diff)
duration_f64_perturb = time.perf_counter() - t1

print(f"diff_double (perturbed): {diffDouble_perturbed.abs().max().item()}")

t2 = time.perf_counter()
save_dict = {
    "x": x,
    "y": y,
    "result_f32": result,
    "perturb_double": diffDouble_perturbed
}
torch.save(save_dict, "perturb_data.pt")
duration_save = time.perf_counter() - t2

print("-" * 45)
print(f"{'Step':<25} | {'Time (s)':<15}")
print("-" * 45)
print(f"{'FP32 Calculation':<25} | {duration_f32:.6f}")
print(f"{'FP64 Perturb Calculation':<25} | {duration_f64_perturb:.6f}")
print(f"{'Data Saving':<25} | {duration_save:.6f}")
print("-" * 45)

script_total = time.perf_counter() - script_start
print(f"Total script execution time: {script_total:.4f} s")