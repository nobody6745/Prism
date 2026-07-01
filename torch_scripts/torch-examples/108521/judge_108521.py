import torch
import os
import time  # 导入计时模块

def check_interval_coverage_inverted():
    # 记录脚本开始时间
    script_start = time.perf_counter()

    file_clean = "matmul_mismatch.pt"
    file_perturb = "perturb.pt"

    if not (os.path.exists(file_clean) and os.path.exists(file_perturb)):
        print(f"❌ 错误: 找不到文件。")
        return

    print(f"📂 加载数据中...")
    data_clean = torch.load(file_clean)
    data_perturb = torch.load(file_perturb)

    print("\n🚀 开始反向区间覆盖测试...")
    
    # 初始化全局统计变量
    total_radius_sum = 0.0
    max_radius_global = 0.0
    total_elements_global = 0
    passed_cases = 0
    total_cases = len(data_clean)

    print(f"{'ID':<4} | {'Shape':<15} | {'In Range %':<12} | {'Avg Radius':<12} | {'Max Radius':<12} | {'Status'}")
    print("-" * 100)

    for idx, (rec_clean, rec_perturb) in enumerate(zip(data_clean, data_perturb)):
        meta = rec_clean["meta"]
        i = meta["i"]
        
        # 1. 提取计算变量
        raw_slice = rec_clean["result_slice"].float().cuda()
        val_slice = raw_slice[:i, :]

        if "result_full_f32" in rec_clean and rec_clean["result_full_f32"] is not None:
            raw_full = rec_clean["result_full_f32"].float().cuda()
        else:
            raw_full = rec_clean["result_full"].float().cuda()
        val_full = raw_full[:i, :] 

        raw_perturb = rec_perturb["result_perturb"].float().cuda()
        val_perturb = raw_perturb[:i, :] 

        # 2. 计算区间与半径 (Radius = |Slice - Perturb|)
        # 这里的半径代表了由于非确定性导致的“容错半宽”
        radius_tensor = (val_slice - val_perturb).abs()
        
        # 统计当前 case 的半径
        current_avg_radius = radius_tensor.mean().item()
        current_max_radius = radius_tensor.max().item()
        
        # 更新全局统计
        total_radius_sum += radius_tensor.sum().item()
        total_elements_global += radius_tensor.numel()
        if current_max_radius > max_radius_global:
            max_radius_global = current_max_radius

        # 3. 判定判定
        bound_a = val_perturb
        bound_b = 2 * val_slice - val_perturb
        lower, upper = torch.min(bound_a, bound_b), torch.max(bound_a, bound_b)
        
        epsilon = 1e-12
        is_inside = (val_full >= (lower - epsilon)) & (val_full <= (upper + epsilon))
        ratio = is_inside.sum().item() / val_full.numel() * 100
        
        if ratio == 100.0: passed_cases += 1
        status = "✅ PASS" if ratio == 100.0 else "❌ FAIL"
        
        shape_str = f"{meta['m']},{meta['n']},{meta['k']}"
        print(f"{idx:<4} | {shape_str:<15} | {ratio:.2f}%{'':<6} | {current_avg_radius:.4e} | {current_max_radius:.4e} | {status}")

    # 4. 汇总统计
    global_avg_radius = total_radius_sum / total_elements_global if total_elements_global > 0 else 0
    
    torch.cuda.synchronize()
    script_end = time.perf_counter()
    duration = script_end - script_start

    print("\n" + "=" * 60)
    print(f"📊 最终统计报告 (Final Statistics)")
    print("-" * 60)
    print(f"📏 全局平均区间半径 (Avg Radius): {global_avg_radius:.6e}")
    print(f"📏 全局最大区间半径 (Max Radius): {max_radius_global:.6e}")
    print(f"⏱️  脚本总执行时间: {duration:.4f} 秒")
    print(f"✅ 测试通过率: {passed_cases}/{total_cases}")
    print("=" * 60)

if __name__ == "__main__":
    if torch.cuda.is_available():
        check_interval_coverage_inverted()
    else:
        print("Need CUDA to run.")