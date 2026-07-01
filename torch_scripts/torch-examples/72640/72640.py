import torch
import random
import numpy as np
import time

def run_and_save():
    script_start = time.perf_counter()
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    li = torch.nn.Linear(1000, 2000)
    li.eval()

    ex = torch.rand(2, 1000)
    ex3 = ex[:1]

    out = li(ex)      # Batch=2
    out3 = li(ex3)    # Batch=1
    out_slice = out[0:1] 

    filename = "linear_batch_diff.npz"

    np.savez(filename,
        out_batch1=out3.detach().cpu().numpy(),      # B1_Clean
        out_batch2_slice=out_slice.detach().cpu().numpy(), # B2_Clean_Slice
        out_batch2_full=out.detach().cpu().numpy()
    )
    
    script_end = time.perf_counter()
    print(f"Total Time: {script_end - script_start:.4f} s")

if __name__ == '__main__':
    run_and_save()