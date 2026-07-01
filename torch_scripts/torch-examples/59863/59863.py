import torch
import numpy as np
import os
import time 

def run_and_record():
    total_start = time.perf_counter()  
    print(f"PyTorch Version: {torch.__version__}")
    
    N = 8 * 80 * 80 * 80  # 4,096,000
    print(f"Data Size: {N}")
    
    data_np = np.arange(N) - (N / 2) + 0.5
    t = torch.from_numpy(data_np).to(torch.float32)
    
    print("-" * 50)

    print("Running Single-Thread Summation...")
    torch.set_num_threads(1)
    
    s1_start = time.perf_counter()
    sum_1thread = t.sum().item()
    s1_end = time.perf_counter()
    
    duration_1 = s1_end - s1_start
    print(f"Sum (Threads=1):        {sum_1thread:.6f}")
    print(f"Time (Threads=1):       {duration_1:.8f} s")

    print("Running Multi-Thread Summation...")
    torch.set_num_threads(2)
    
    s2_start = time.perf_counter()
    sum_2threads = t.sum().item()
    s2_end = time.perf_counter()
    
    duration_2 = s2_end - s2_start
    print(f"Sum (Threads=2):        {sum_2threads:.6f}")
    print(f"Time (Threads=2):       {duration_2:.8f} s")
    
    print("-" * 50)
    
    diff = sum_1thread - sum_2threads
    speedup = duration_1 / duration_2 if duration_2 > 0 else 0
    
    print(f"Diff (1 vs 2 threads):  {diff:.6e}")
    print(f"Speedup:                {speedup:.2f}x")
    
    filename = "sum_thread_comparison.npz"
    print(f"\nSaving results to {filename}...")
    
    np.savez(filename, 
        val_1thread=sum_1thread,
        val_2threads=sum_2threads,
        duration_1=duration_1,
        duration_2=duration_2,
        diff=diff,
        config={'N': N}
    )
    
    total_end = time.perf_counter()
    print(f"Total Execution Time:   {total_end - total_start:.4f} s")
    print("Done.")

if __name__ == "__main__":
    run_and_record()