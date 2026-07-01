import torch
import time

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


dim1_indices = torch.arange(4, dtype=torch.int64).view(1, 4, 1, 1)
indice = dim1_indices.expand((2, 4, 256, 256))
vals = torch.randn((2, 4, 256, 256), dtype=torch.float16)
grad = torch.zeros((4,), dtype=torch.float16)
accumulate = True

indice = indice.to(device)
vals = vals.to(device)
grad = grad.to(device)
_ = torch.ops.aten.index_put(grad, [indice], vals, accumulate=accumulate)
if device == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
result = torch.ops.aten.index_put(grad, [indice], vals, accumulate=accumulate)

if device == 'cuda': torch.cuda.synchronize()
duration = time.perf_counter() - t0
print(f"time: {duration:.8f} s")

print(f"result (First 4):\n{result}")
