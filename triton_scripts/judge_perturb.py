import os
import torch
import numpy as np
from tabulate import tabulate
from dataclasses import dataclass


@dataclass
class GroundTruth:
    reference_in: bool
    triton_in: bool

cases = {
    "1924": GroundTruth(True, True), "5990": GroundTruth(True, True),
    "3017": GroundTruth(True, False), "5895": GroundTruth(True, True),
    "4551": GroundTruth(True, True), "1190": GroundTruth(True, True),
    "2843": GroundTruth(True, True), "1960": GroundTruth(True, True),
    "2680b": GroundTruth(True, True), "376": GroundTruth(True, True),
    "5065": GroundTruth(True, True), "1840": GroundTruth(True, True),
    "1666": GroundTruth(True, False), "1937": GroundTruth(True, True),
    "1671": GroundTruth(True, True), "1821": GroundTruth(True, False),
    "4701": GroundTruth(True, False), "1808": GroundTruth(True, True),
    "1578": GroundTruth(True, True), "6227": GroundTruth(True, True),
}


table_data = []
headers = [
    "Case ID",
    "Max Absolute Error (Baseline vs Target) & Key Point Analysis",
    "Target in Range (Count/Total | %)"
]


for case in cases.keys():
    reference_file = f"./perturb_examples/{case}/reference.pt"
    perturbed_file = f"./perturb_examples/{case}/perturb.npy"

    if not os.path.exists(reference_file) or not os.path.exists(perturbed_file):
        table_data.append([case, "File missing", "N/A"])
        continue

    # --- Data Loading ---
    reference_data = torch.load(reference_file, map_location="cpu")
    
    torch_raw = reference_data["torch_output"].cpu()
    triton_raw = reference_data["triton_output"].cpu()

    # Determine baseline and target
    if case == "3017":
        baseline_tensor = torch_raw
        target_tensor = triton_raw
        baseline_name = "Torch (Ref)"
        target_name = "Triton (Target)"
    else:
        baseline_tensor = triton_raw
        target_tensor = torch_raw
        baseline_name = "Triton (Ref)"
        target_name = "Torch (Target)"

    # Convert to Numpy
    baseline_np = baseline_tensor.numpy()
    target_np = target_tensor.numpy()

    # Load perturbation data
    perturbed_flat = np.load(perturbed_file)

    if baseline_np.size != perturbed_flat.size:
        table_data.append([case, "Data size mismatch", "N/A"])
        continue

    perturbed_np = perturbed_flat.reshape(baseline_np.shape)

    abs_error = np.abs(baseline_np - target_np)
    max_absolute_error = np.max(abs_error)
    max_err_idx = np.argmax(abs_error)

    target_np_64 = target_np.astype(np.float64)
    perturbed_np_64 = perturbed_np.astype(np.float64)
    baseline_np_64 = baseline_np.astype(np.float64)

    # Calculate perturbation interval boundaries
    lower_bound = np.minimum(perturbed_np_64, 2 * baseline_np_64 - perturbed_np_64)
    upper_bound = np.maximum(perturbed_np_64, 2 * baseline_np_64 - perturbed_np_64)
    
    # === Key Logic: If float16, add one fp16 ULP compensation to the upper bound ===
    if baseline_tensor.dtype == torch.float16:
        # np.spacing provides the minimum interval for the value at float16 precision
        ulp_f16 = np.spacing(upper_bound.astype(np.float16)).astype(np.float64)
        upper_bound += ulp_f16
    
    # Check if Target falls within the interval
    in_range = (target_np_64 >= lower_bound) & (target_np_64 <= upper_bound)
    
    count_in_range = np.sum(in_range)
    total_count = in_range.size
    percent_in_range = 100.0 * (count_in_range / total_count)

    baseline_val_at_max = baseline_np.flatten()[max_err_idx]
    perturbed_val_at_max = perturbed_np.flatten()[max_err_idx]
    target_val_at_max = target_np.flatten()[max_err_idx]
    lower_bound_at_max = lower_bound.flatten()[max_err_idx]
    upper_bound_at_max = upper_bound.flatten()[max_err_idx]

    max_abs_err_details_str = (
        f"{max_absolute_error:.6e}\n"
        f"---------- Key Point Analysis ----------\n"
        f"{target_name}:     {target_val_at_max:.6e}\n"
        f"Lower Bound:         {lower_bound_at_max:.6e}\n"
        f"Upper Bound:         {upper_bound_at_max:.6e}\n"
        f"{baseline_name}:     {baseline_val_at_max:.6e}\n"
        f"Perturbed:           {perturbed_val_at_max:.6e}"
    )

    range_info_str = (
        f"{percent_in_range:.2f}%\n"
        f"({count_in_range} / {total_count})"
    )

    table_data.append([
        case,
        max_abs_err_details_str,
        range_info_str
    ])

print( "Perturbation Result Analysis (Includes FP16 ULP Compensation)")
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))