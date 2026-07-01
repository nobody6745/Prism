import torch
import time
import math

start = time.perf_counter()

print(f"PyTorch Version: {torch.__version__}")


a = torch.tensor([[5000]]).float()
b = torch.tensor([[6500]]).float()

a_sq = torch.sum(a * a, dim=-1, keepdim=True)  # a^2
b_sq = torch.sum(b * b, dim=-1, keepdim=True)  # b^2
ab = torch.mm(a, b.t())                        # ab


a_sq_pert = torch.nextafter(a_sq, torch.full_like(a_sq, float('inf')))
b_sq_pert = torch.nextafter(b_sq, torch.full_like(b_sq, float('inf')))

ab_pert = torch.nextafter(ab, torch.full_like(ab, float('-inf')))

d1_sq = a_sq_pert - 2 * ab_pert + b_sq_pert.t()
d1 = torch.sqrt(torch.clamp(d1_sq, min=0.0))
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

print("\n--- Normal Behavior (With 1 ULP Perturbation) ---")
print(f"a: {a.item()}, b: {b.item()}")
print(f"d1 (Perturbed): {d1.item()}  <-- 小数值下，1 ULP 扰动几乎无影响")
print(f"d2 (Correct):   {d2.item()}")


a = torch.tensor([[512695]]).float()
b = torch.tensor([[512804]]).float()

a_sq = torch.sum(a * a, dim=-1, keepdim=True)  
b_sq = torch.sum(b * b, dim=-1, keepdim=True)  
ab = torch.mm(a, b.t())                        

a_sq_pert = torch.nextafter(a_sq, torch.full_like(a_sq, float('inf')))
b_sq_pert = torch.nextafter(b_sq, torch.full_like(b_sq, float('inf')))

ab_pert = torch.nextafter(ab, torch.full_like(ab, float('-inf')))


ulp_a_sq = a_sq_pert.item() - a_sq.item()
print("\n--- ULP Size Check ---")
print(f"1 ULP at {a_sq.item():.2e} is exactly: {ulp_a_sq}")

d1_sq = a_sq_pert - 2 * ab_pert + b_sq_pert.t()
d1 = torch.sqrt(torch.clamp(d1_sq, min=0.0))
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

print("\n--- Buggy Behavior (With 1 ULP Perturbation) ---")
print(f"a: {a.item()}, b: {b.item()}")
print(f"d1 (Perturbed): {d1.item()}  <-- 误差爆炸！")
print(f"d2 (Correct):   {d2.item()}")

end = time.perf_counter()
print(f"\nTotal runtime: {end - start:.6f} seconds")