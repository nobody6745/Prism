import torch
import numpy as np
import math
import os
import time 

def perturb_down_ulp(t):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    target = torch.full_like(t, float('-inf'))
    return torch.nextafter(t, target)

def local_perturbed_arange(start, end, step, dtype=torch.float32):
    t_start = time.perf_counter()
    
    span = end - start
    num_steps = math.ceil(span / step)
    res = torch.empty(num_steps, dtype=dtype)
    start_t = torch.tensor(start, dtype=dtype)
    step_t = torch.tensor(step, dtype=dtype)
    
    for i in range(num_steps):
        i_t = torch.tensor(i, dtype=dtype)
        offset = perturb_down_ulp(i_t * step_t)
        val = perturb_down_ulp(start_t + offset)
        res[i] = val
        
    duration = time.perf_counter() - t_start
    return res, duration

def test_arange_accuracy():
    SAVE_DIR = "./output_results"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    DATA_FILE = os.path.join(SAVE_DIR, "arange_perturbed.pt")
    LOG_FILE = os.path.join(SAVE_DIR, "arange_perturbed_log.txt")
    script_real_start = time.perf_counter()

    with open(LOG_FILE, "w", encoding="utf-8") as log_f:
        def log_print(text):
            print(text)
            log_f.write(text + "\n")

        saved_data = {}
        input_shapes = [(2.555599, 50, 1, torch.strided, torch.float)]

        count = 0
        for params in input_shapes:
            count += 1
            start, end, step, layout, dtype = params
-
            t0 = time.perf_counter()
            pytorch_arange_output = torch.arange(start=start, end=end, step=step, device="cpu", dtype=dtype)
            t_torch = time.perf_counter() - t0

            t1 = time.perf_counter()
            numpy_arange_output = np.arange(start, end, step, dtype=np.float32)
            t_numpy = time.perf_counter() - t1

            local_perturbed_output, t_local = local_perturbed_arange(start, end, step, dtype=dtype)

            log_print(f"Test {count}: start={start}, end={end}, step={step}")
            log_print(f" PyTorch Time: {t_torch:.8f} s")
            log_print(f" NumPy Time:   {t_numpy:.8f} s")
            log_print(f" Local Time:   {t_local:.8f} s (Perturbed)")
            log_print("-" * 50)
            is_smaller = torch.all(local_perturbed_output <= pytorch_arange_output)
            log_print(f"All Perturbed <= PyTorch? {is_smaller.item()}")
            log_print("=" * 50 + "\n")

            saved_data[f"test_{count}"] = {
                "params": params,
                "timings": {"torch": t_torch, "numpy": t_numpy, "local": t_local},
                "local_perturbed": local_perturbed_output
            }

    torch.save(saved_data, DATA_FILE)
    script_total = time.perf_counter() - script_real_start
    print(f"time : {script_total:.4f} s")

if __name__ == "__main__":
    test_arange_accuracy()