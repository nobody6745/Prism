import torch
import time

start_time = time.perf_counter()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

index = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

while True:
    s = torch.randn(4, 5, dtype=torch.float32, device=device)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    blocks = [s[batch == i] for i in batch.unique()]
    s_block = torch.block_diag(*blocks).to(dtype=s.dtype, device=s.device)
    sum_s = s.sum(dim=-1, dtype=s.dtype)
    sum_s_block = s_block.sum(dim=-1, dtype=s_block.dtype)

    mean_s = sum_s.mean()
    mean_s_block = sum_s_block.mean()
    if torch.not_equal(mean_s, mean_s_block).any():
        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        ms = mean_s.item()
        msb = mean_s_block.item()
        diff = ms - msb

        print(f"\n Error detected at iteration {index}!")
        print(f"mean_s:       {ms:.20f}")
        print(f"mean_s_block: {msb:.20f}")
        print(f"Difference:   {diff:.20f}")

        save_path = "block_diag_error.pt"
        save_dict = {
            "iteration": index,
            "s_raw": s.cpu(),
            "s_block": s_block.cpu(),
            "sum_s": sum_s.cpu(),
            "sum_s_block": sum_s_block.cpu(),
            "mean_s": mean_s.cpu(),
            "mean_s_block": mean_s_block.cpu(),
            "diff_val": diff,
        }
        torch.save(save_dict, save_path)
        print(f" Total runtime: {end_time - start_time:.6f} seconds")

        break

    if index % 100 == 0:
        print(f"Test {index}: Success (Means match: {mean_s.item():.6f})")

    index += 1

    if index > 10000:
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        print("Reached max iterations without error.")
        print(f" Total runtime: {end_time - start_time:.6f} seconds")
        break