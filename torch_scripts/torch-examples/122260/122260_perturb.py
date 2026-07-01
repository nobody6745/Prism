import torch
import datetime
import time 

# Set default device and print precision for high-fidelity analysis
torch.set_default_device('cuda')
torch.set_printoptions(precision=10)

scale = torch.tensor(0.180336877703666687)
x = torch.tensor(1134139801600.000000)

def perturb_neg_inf(val):
    """Perturbs the value by 1 ULP towards negative infinity."""
    neg_inf = torch.tensor(float('-inf'), dtype=val.dtype, device=val.device)
    return torch.nextafter(val, neg_inf)

def f(x, scale):
    torch.cuda.synchronize()
    t_f_start = time.perf_counter()

    # Step 1: Standard multiplication
    mul1 = x * scale
    
    # Step 2: Multiplication with 1 ULP perturbation
    mul2 = x * scale
    mul2_p = perturb_neg_inf(mul2)
    
    # Step 3: Compute difference and apply another perturbation
    diff = mul1 - mul2_p
    diff_p = perturb_neg_inf(diff)
    
    # Step 4: Exponential result with final perturbation
    res = torch.exp(diff_p)
    final_res = perturb_neg_inf(res)

    torch.cuda.synchronize()
    t_f_duration = time.perf_counter() - t_f_start
    return final_res, t_f_duration

script_start = time.perf_counter()

# Execute core computation
res_eager, calc_duration = f(x, scale)

print("="*50)
print("ULP Perturbation Experiment Results")
print("="*50)
print(f"Eager Perturbed Value: {res_eager.item():.10f}")
print(f"Core Calculation Duration: {calc_duration:.6f} seconds")
print("="*50)

# Prepare metadata and save results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

results_dict = {
    "x_input": x.cpu(),
    "scale_input": scale.cpu(),
    "perturbed_output": res_eager.cpu(),
    "calc_time": calc_duration,
    "timestamp": timestamp
}
torch.save(results_dict, "perturbation_results.pt")

script_total_duration = time.perf_counter() - script_start
print(f"\nResults saved. Total script execution time: {script_total_duration:.4f} seconds")