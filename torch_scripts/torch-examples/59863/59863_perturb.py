import torch
import numpy as np
import os
import time 

def run_and_record():
    script_start = time.perf_counter()
    
    print(f"PyTorch Version: {torch.__version__}")
    
    data_start = time.perf_counter()
    N = 8 * 80 * 80 * 80  # 4,096,000
    print(f"Data Size: {N}")
    
    data_np = np.arange(N) - (N / 2) + 0.5
    t = torch.from_numpy(data_np).to(torch.float32)
    data_end = time.perf_counter()
    
    print("-" * 50)

    print("Running Single-Thread Summation...")
    torch.set_num_threads(1)
    compute_start = time.perf_counter()
    sum_1thread = t.sum().item()
    compute_end = time.perf_counter()
    
    print(f"Sum (Threads=1):        {sum_1thread:.6f}")

    filename = "sum_thread_perturb.npz"
    print(f"\n Saving results to {filename}...")
    
    np.savez(filename, 
        val_1thread=sum_1thread,
    )
    
    script_end = time.perf_counter()

    print("-" * 50)
    print(f"Data Preparation Time:  {data_end - data_start:.6f} s")
    print(f"Compute Time (Sum):     {compute_end - compute_start:.6f} s")
    print(f"Total Execution Time:   {script_end - script_start:.6f} s")
    print("Done.")

if __name__ == "__main__":
    run_and_record()