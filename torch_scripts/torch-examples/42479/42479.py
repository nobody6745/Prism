import torch
import time

start = time.perf_counter()

print(torch.__version__)

# Normal behavior
a = torch.tensor([[5000]]).float()
b = torch.tensor([[6500]]).float()
d1 = torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

print(a)
print(b)
print(d1)
print(d2)

print()

# Buggy behavior
a = torch.tensor([[512695]]).float()
b = torch.tensor([[512804]]).float()
d1 = torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

print(a)
print(b)
print(d1)
print(d2)

end = time.perf_counter()

print(f"\nTotal runtime: {end - start:.6f} seconds")