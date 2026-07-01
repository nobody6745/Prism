import torch
import numpy as np
import os
import time 

target_ds = [5, 6]

script_start = time.perf_counter()

print(f"{'d':<3} | {'Input File':<18} | {'Diff (Old vs New)':<20} | {'Loop Time (ms)':<15} | {'Output File'}")
print("-" * 90)

for d in target_ds:
    loop_start = time.perf_counter() 
    
    input_filename = f"mismatch_d{d}.npz"
    output_filename = f"perturb__d{d}.npz"
    
    if not os.path.exists(input_filename):
        print(f"{d:<3} | {input_filename:<18} | {'[File Not Found]':<20} | {'-':<15} | -")
        continue

    data = np.load(input_filename)
    a = torch.from_numpy(data['a'])
    b = torch.from_numpy(data['b'])
    y1_old = data['y1']
    y2_old = data['y2']
    y1_perturb = torch.einsum("bij,bj->b", a, b)
    y2_perturb = torch.einsum("bij,bj->bi", a, b).sum(dim=1)
    
    y1_perturb_np = y1_perturb.detach().numpy()
    y2_perturb_np = y2_perturb.detach().numpy()

    diff_check = np.linalg.norm(y1_old - y1_perturb_np)

    save_dict = {
        "a": data['a'],
        "b": data['b'],
        "y1_old": y1_old,
        "y2_old": y2_old,
        "y1_perturb": y1_perturb_np,
        "y2_perturb": y2_perturb_np
    }
    np.savez(output_filename, **save_dict)
    
    loop_duration = (time.perf_counter() - loop_start) * 1000  
    
    print(f"{d:<3} | {input_filename:<18} | {diff_check:<20.4e} | {loop_duration:<15.2f} | {output_filename}")

total_duration = time.perf_counter() - script_start

print("-" * 90)
print(f"Total execution time: {total_duration:.4f} seconds")
print("Done. Use numpy to inspect the 'perturb__d*.npz' files.")