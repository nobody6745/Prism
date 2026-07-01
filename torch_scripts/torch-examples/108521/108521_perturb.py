import torch
import time  # Import timing module

# Used to store all mismatch cases
mismatch_records = []

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def test(m, n, k, dtype, rtol, atol):
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    # 'c' is the reference result for the full 'm' rows
    c = torch.mm(a, b)

    for i in range(1, m + 1):
        # 'd' is the independent calculation result for the first 'i' rows
        d = torch.mm(a[:i, :], b)
        
        slice_c = c[:i, :]
        
        # Check if the independent slice matches the sliced reference
        if not torch.allclose(d, slice_c, rtol=rtol, atol=atol):
            diff = (d - slice_c).abs()
            max_diff = diff.max().item()
            
            print(f"Mismatch found: i={i}, m={m}, n={n}, k={k}, dtype={dtype}")
            print(f"   Max Diff: {max_diff:.6e}")
            
            record = {
                "meta": {
                    "i": i, "m": m, "n": n, "k": k, 
                    "dtype": str(dtype),
                    "rtol": rtol, "atol": atol
                },
                "input_a": a.cpu(),
                "input_b": b.cpu(),
                "result_full": c.cpu(),
                "result_slice": d.cpu(),
                "diff": diff.cpu(),
                "max_diff": max_diff
            }

            # For FP16, compute an FP32 reference to analyze numerical drift
            if dtype == torch.float16:
                a_f32 = a.float()
                b_f32 = b.float()
                c_f32 = torch.mm(a_f32, b_f32)
                record["result_full"] = c_f32.cpu()

            mismatch_records.append(record)
            break

set_seed(0)

# Define precision thresholds and dtypes
dtypes = [
    (1e-3, 1e-5, torch.float16),
    (1e-5, 1e-8, torch.float32),
]

print("Starting Matrix Multiplication Consistency Test...")

# Start recording total execution time
torch.cuda.synchronize()
start_total = time.perf_counter()

# Grid search across different dimensions
for rtol, atol, dtype in dtypes:
    for m in [4, 8, 16]:
        for n in [256, 512]:
            for k in [256, 512]:
                test(m, n, k, dtype, rtol, atol)

# End recording total execution time
torch.cuda.synchronize()
end_total = time.perf_counter()
total_duration = end_total - start_total

# Save results
save_filename = "matmul_mismatch.pt"
if len(mismatch_records) > 0:
    torch.save(mismatch_records, save_filename)
    print("\n" + "="*60)
    print(f"Test complete. Found {len(mismatch_records)} mismatch cases.")
    print(f"Total execution time: {total_duration:.4f} seconds")
    print(f"Detailed data saved to: {save_filename}")
    print("="*60)
else:
    print("\n" + "="*60)
    print("Test PASSED!")
    print(f"Total execution time: {total_duration:.4f} seconds")
    print("="*60)