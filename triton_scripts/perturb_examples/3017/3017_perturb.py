import triton
import triton.language as tl
import torch
import numpy as np

@triton.jit
def test_kernel(
        # Inputs
        k, k_stride_i, k_stride_d,
        v, v_stride_i, v_stride_e,
        # Outputs
        output, out_stride_i, out_stride_e,
):
    i = tl.arange(0, 16)

    k_loaded = tl.load(
        k
        + i[:, None] * k_stride_i
        + i[None, :] * k_stride_d,
    )
    v_loaded = tl.load(
        v
        + i[:, None] * v_stride_i
        + i[None, :] * v_stride_e,
    )
    tl.debug_barrier()

    kv = k_loaded[:, :, None] * v_loaded[:, None, :]
    tl.debug_barrier()

    context = tl.cumsum(kv, axis=0)
    tl.debug_barrier()

    out = tl.sum(context, axis=1)
    tl.debug_barrier()

    tl.store(
        output
        + i[:, None] * out_stride_i
        + i[None, :] * out_stride_e,
        out
    )


def test_case():

    T = 16
    inputs = torch.load("inputs.pt", weights_only=True)
    k = inputs["k"]
    v = inputs["v"]
    out = torch.zeros((T, T), device='cuda')

    test_kernel[(1,)](
        k, *k.stride(),
        v, *v.stride(),
        out, *out.stride(),
    )

    kv = k[:, :, None] * v[:, None, :]
    context = torch.cumsum(kv, dim=0)
    out_cmp = torch.sum(context, dim=1)
    print(out_cmp)
    print(out)
    print('max diff:', (out - out_cmp).abs().max())
    triton_output_np = out.cpu().numpy()
    np.save("perturb.npy", triton_output_np)
    print("\nTriton output has been successfully saved to 'perturb.npy'")


if __name__ == "__main__":
    test_case()

