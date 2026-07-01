import torch
import numpy as np
import time  


torch.manual_seed(2)
torch.set_printoptions(precision=20, sci_mode=False)

dim = 90


q = torch.rand(size=(128, dim), device='cpu', dtype=torch.double)
one_array_batch1 = torch.ones(size=(1, dim), device='cpu', dtype=torch.double)
one_array_batch2 = torch.ones(size=(2, dim), device='cpu', dtype=torch.double)

start_time_1 = time.perf_counter()
res1 = one_array_batch1.matmul(q.t())
end_time_1 = time.perf_counter()
duration1 = end_time_1 - start_time_1

start_time_2 = time.perf_counter()
res2 = one_array_batch2.matmul(q.t())
end_time_2 = time.perf_counter()
duration2 = end_time_2 - start_time_2


res1_comp = res1[0:1]
res2_comp = res2[0:1]


identity = (res1_comp == res2_comp)


print(f"Path 1 (Batch=1) Time: {duration1:.8f} seconds")
print(f"Path 2 (Batch=2) Time: {duration2:.8f} seconds")
print("-" * 30)
print("Identity Check:", identity)
print(f"All match: {identity.all()}")

filename = "matmul_comparison_with_time.npz"
print(f"\nSaving results to {filename}...")

save_dict = {
    "q": q.numpy(),
    "res1_comp": res1_comp.numpy(),
    "res2_comp": res2_comp.numpy(),
    "identity": identity.numpy(),
    "duration1": np.array(duration1), 
    "duration2": np.array(duration2)  
}

np.savez(filename, **save_dict)
print("Done.")