import torch
import time 

mismatch_records = []

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def test(m, n, k, dtype, rtol, atol):
    # Initialize random matrices on CUDA
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    # Compute full matrix multiplication (GEMM)
    c = torch.mm(a, b)

    # Iteratively slice the rows of A and compare results
    for i in range(1, m + 1):
        d = torch.mm(a[:i, :], b)
        slice_c = c[:i, :]
        
        # Check numerical consistency between sliced GEMM and full GEMM
        if not torch.allclose(d, slice_c, rtol=rtol, atol=atol):
            diff = (d - slice_c).abs()
            max_diff = diff.max().item()
            
            print(f" Mismatch found: i={i}, m={m}, n={n}, k={k}, dtype={dtype}")
            print(f"   Max Diff: {max_diff:.6e}")
            
            record = {
                "meta": {"i": i, "m": m, "n": n, "k": k, "dtype": str(dtype)},
                "result_full": c.cpu(),
                "result_slice": d.cpu(),
                "max_diff": max_diff
            }

            # For FP16, provide FP32 reference for better analysis
            if dtype == torch.float16:
                c_f32 = torch.mm(a.float(), b.float())
                record["result_full"] = c_f32.cpu()

            mismatch_records.append(record)
            break

set_seed(0)

# Define precision thresholds and dtypes for testing
dtypes = [
    (1e-3, 1e-5, torch.float16),
    (1e-5, 1e-8, torch.float32),
]

print("Starting Matrix Multiplication Consistency Test...")

torch.cuda.synchronize()  
script_start = time.perf_counter()

# Grid search across different matrix dimensions
for rtol, atol, dtype in dtypes:
    for m in [4, 8, 16]:
        for n in [256, 512]:
            for k in [256, 512]:
                test(m, n, k, dtype, rtol, atol)

torch.cuda.synchronize() 
script_end = time.perf_counter()
total_duration = script_end - script_start

save_filename = "matmul_mismatch.pt"
if len(mismatch_records) > 0:
    torch.save(mismatch_records, save_filename)
    print("\n" + "="*60)
    print(f"Test complete. Found {len(mismatch_records)} mismatch cases.")
    print(f"Total Execution Time: {total_duration:.4f} seconds")
    print(f"Detailed data saved to: {save_filename}")
    print("="*60)
else:
    print("\n" + "="*60)
    print(f"Test PASSED! Total Execution Time: {total_duration:.4f} seconds")
    print("="*60)