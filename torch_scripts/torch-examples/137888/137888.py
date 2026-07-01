import numpy as np
import torch
import tensorflow as tf
import os
import time  

SAVE_DIR = "./output_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
DATA_FILE = os.path.join(SAVE_DIR, "digamma_diff.pt")

def test_digamma():
    script_start = time.perf_counter()
    
    print(f"=== Comparing digamma (PyTorch vs TensorFlow) ===")
    raw_val = 0.10000000149011612
    input_data = np.array([raw_val], dtype=np.float32)
    
    print(f"Input (float32): {input_data[0]:.20e}")
    print("-" * 60)
    pt_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    t0 = time.perf_counter()
    pt_result_tensor = torch.digamma(pt_tensor)
    pt_duration = time.perf_counter() - t0
    
    pt_result = pt_result_tensor.item()
    print(f"PyTorch Result:    {pt_result:.20e} | Time: {pt_duration:.8f} s")

    with tf.device('/CPU:0'):
        tf_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        
        t1 = time.perf_counter()
        tf_result_tensor = tf.math.digamma(tf_tensor)
        tf_duration = time.perf_counter() - t1
        
        tf_result = tf_result_tensor.numpy()[0]
    print(f"TensorFlow Result: {tf_result:.20e} | Time: {tf_duration:.8f} s")

    diff = pt_result - tf_result
    abs_diff = abs(diff)
    
    print("-" * 60)
    print(f"Difference:        {diff:.20e}")
    
    if abs_diff == 0:
        print("Bit-exact match")
    else:
        print("Implementation divergence")

    saved_data = {
        "outputs": {
            "pytorch": pt_result,
            "tensorflow": float(tf_result)
        },
        "timings": {
            "pytorch": pt_duration,
            "tensorflow": tf_duration
        },
        "analysis": {
            "diff": diff,
            "is_equal": (abs_diff == 0)
        }
    }

    torch.save(saved_data, DATA_FILE)
    script_total = time.perf_counter() - script_start
    print(f"\n time: {script_total:.4f} s")

if __name__ == "__main__":
    test_digamma()