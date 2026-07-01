import torch
import platform
import time  

def print_system_info():
    print(f"OS: {platform.system()}")
    print(f"Architecture: {platform.machine()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Num threads: {torch.get_num_threads()}")

def test():
    script_start = time.perf_counter()

    N = 128
    x = torch.linspace(0, 1, steps=N*N, dtype=torch.float32).reshape(N, N)
    y = torch.linspace(0, 1, steps=N*N, dtype=torch.float32).reshape(N, N).mT

    t0 = time.perf_counter()
    
    result = torch.matmul(x, y)
    final_sum = result.sum()
    
    duration_compute = time.perf_counter() - t0

    val_as_float = final_sum.item()
    val_as_hex = hex(final_sum.view(torch.int32).item())

    print(f"\n Compute Time (matmul+sum): {duration_compute:.8f} s")
    print(f"Decimal Result: {val_as_float:.20f}")
    print(f"Hex Representation: {val_as_hex}")
    
    script_total = time.perf_counter() - script_start
    print(f" Total Script Execution Time: {script_total:.4f} s")

if __name__ == "__main__":
    print_system_info()
    test()