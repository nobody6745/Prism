import torch
import numpy as np
import os
import time 

try:
    import tensorflow as tf
except ImportError:
    tf = None
    print(" Warning: TensorFlow not found. TF tests will be skipped.")

def test_arange_accuracy():
    SAVE_DIR = "./output_results"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    TEXT_LOG_FILE = os.path.join(SAVE_DIR, "arange_record.txt")
    DATA_FILE = os.path.join(SAVE_DIR, "arange_data.pt")
    script_real_start = time.perf_counter()

    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)

    input_shapes = [
        (2.555599, 50, 1, torch.strided, torch.float),
    ]

    all_results_data = {}

    with open(TEXT_LOG_FILE, "w", encoding="utf-8") as log_f:
        def log_print(text):
            print(text)
            log_f.write(text + "\n")
            log_f.flush()

        count = 0
        for input_params in input_shapes:
            count += 1
            start, end, step, layout, dtype = input_params

            log_print(f"Test {count} : Params: start={start}, end={end}, step={step}")
            log_print("-" * 50)

            t0 = time.perf_counter()
            pytorch_arange_output = torch.arange(start=start, step=step, end=end, device="cpu", dtype=dtype, layout=layout)
            t_torch = time.perf_counter() - t0
            log_print(f"PyTorch Time: {t_torch:.8f} s")

            t1 = time.perf_counter()
            numpy_arange_output = np.arange(start, end, step, dtype=np.float32)
            t_numpy = time.perf_counter() - t1
            log_print(f"NumPy Time:   {t_numpy:.8f} s")

            tf_as_numpy = None
            t_tf = None
            if tf is not None:
                try:
                    t2 = time.perf_counter()
                    tf_arange_output = tf.experimental.numpy.arange(start, end, step, dtype=tf.float32)
                    tf_as_numpy = tf_arange_output.numpy()
                    t_tf = time.perf_counter() - t2
                    log_print(f"TF Time:      {t_tf:.8f} s")
                except Exception as e:
                    log_print(f"TF failed: {e}")
            else:
                log_print("TF Time:      Skipped")

            log_print("-" * 50)

            all_results_data[f"test_{count}"] = {
                "params": input_params,
                "torch": pytorch_arange_output,
                "numpy": numpy_arange_output,
                "tensorflow": tf_as_numpy,
                "timings": {"torch": t_torch, "numpy": t_numpy, "tf": t_tf}
            }

    torch.save(all_results_data, DATA_FILE)
    script_total = time.perf_counter() - script_real_start
    print(f"\n time: {script_total:.4f} s")

if __name__ == "__main__":
    test_arange_accuracy()