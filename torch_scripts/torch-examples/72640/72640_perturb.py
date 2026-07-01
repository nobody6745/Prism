import torch
import random
import numpy as np
import time

def run_and_save_batch1():
    script_start = time.perf_counter()
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    li = torch.nn.Linear(1000, 2000)
    li.eval()
    input_data = torch.rand(1, 1000)
    
    output = li(input_data)

    filename = "linear_batch_perturb.npz"
    np.savez(filename,
        output=output.detach().cpu().numpy() 
    )
    
    script_end = time.perf_counter()
    print(f"Total Time: {script_end - script_start:.4f} s")

if __name__ == '__main__':
    run_and_save_batch1()