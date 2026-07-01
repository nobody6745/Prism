import torch
import time

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

index = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

script_start = time.perf_counter()

while True:
    # Initialize random data
    s = torch.randn(4, 5, dtype=torch.float32, device=device)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    
    # --- Path A: Standard Summation ---
    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    sum_s = s.sum(dim=-1, dtype=s.dtype)
    mean_s = sum_s.mean()
    
    if device.type == 'cuda': torch.cuda.synchronize()
    duration_a = time.perf_counter() - t0
    
    # --- Path B: Block Diagonal Summation ---
    if device.type == 'cuda': torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    # Split by batch index and create a block diagonal matrix
    blocks = [s[batch == i] for i in batch.unique()]
    s_block = torch.block_diag(*blocks).to(dtype=s.dtype, device=s.device)
    sum_s_block = s_block.sum(dim=-1, dtype=s_block.dtype)
    mean_s_block = sum_s_block.mean()
    
    if device.type == 'cuda': torch.cuda.synchronize()
    duration_b = time.perf_counter() - t1

    # --- Strict Numerical Comparison ---
    if torch.not_equal(mean_s, mean_s_block).any():
        ms = mean_s.item()
        msb = mean_s_block.item()
        diff = ms - msb

        print(f"\n Numerical drift detected at iteration {index}!")
        print(f"  Path A Time: {duration_a:.8f} s")
        print(f"  Path B Time: {duration_b:.8f} s (Overhead: {duration_b/duration_a:.2f}x)")
        print(f"mean_s:       {ms:.20f}")
        print(f"mean_s_block: {msb:.20f}")
        print(f"Difference:   {diff:.20e}")
        
        # Save state for debugging
        save_path = "perturb.pt"
        save_dict = {
            "iteration": index,
            "s_raw": s.cpu(),
            "s_block": s_block.cpu(),
            "sum_s_perturb": sum_s.cpu(),
            "mean_s_perturb": mean_s.cpu(),
            "timings": {"path_a": duration_a, "path_b": duration_b}
        }
        
        torch.save(save_dict, save_path)
        print(f"Snapshot data saved to: {save_path}")
        break 

    if index % 500 == 0:
        print(f"Test {index}: Success (mean: {mean_s.item():.6f})")

    index += 1
    if index > 10000:
        print("Reached max iterations without error.")
        break

print(f"\n Total script execution time: {time.perf_counter() - script_start:.4f} s")