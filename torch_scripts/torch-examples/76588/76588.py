import torch
import time

def run_case(x_val, case_id):
    start_time = time.perf_counter()
    
    a = torch.tensor([x_val, x_val, x_val], dtype=torch.float32)
    b = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)

    c = torch.cross(a, b)
    
    end_time = time.perf_counter()
    duration = (end_time - start_time) * 1e6 

    print(f"--- Case {case_id} ---")
    print(f"x_val  : {x_val:e}")
    print(f"Result : {c}")
    print(f"Time   : {duration:.2f} us")
    print()

script_start = time.perf_counter()

run_case(4.294968e+09, 1)


run_case(4.294968e+05, 2)

script_end = time.perf_counter()
print(f"Total script execution time: {(script_end - script_start):.4f} seconds")