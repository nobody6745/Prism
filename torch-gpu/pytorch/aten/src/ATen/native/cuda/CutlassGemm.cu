

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cutlass/gemm/device/gemm.h>  

namespace at {
namespace native {


using CutlassGemmF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,   // A
    float, cutlass::layout::RowMajor,   // B
    float, cutlass::layout::RowMajor,   // C/D
    float                               // accumulator
>;

static void run_cutlass_gemm_f32(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha,
    float beta) {

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2,
              "CUTLASS GEMM: only support 2D tensors for now");

  TORCH_CHECK(
      A.size(1) == B.size(0),
      "CUTLASS GEMM: A.size(1) must equal B.size(0) (K mismatch)");

  TORCH_CHECK(
      A.size(0) == C.size(0) && B.size(1) == C.size(1),
      "CUTLASS GEMM: C shape mismatch");

  // 简化处理：要求 row-major contiguous，非 contiguous 就 copy
  Tensor A_ = A.contiguous();
  Tensor B_ = B.contiguous();
  Tensor C_ = C.contiguous();

  int64_t m = A_.size(0);
  int64_t k = A_.size(1);
  int64_t n = B_.size(1);

  CutlassGemmF32 gemm_op;

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  CutlassGemmF32::Arguments args{
      problem_size,
      {A_.data_ptr<float>(), k},  // lda = K
      {B_.data_ptr<float>(), n},  // ldb = N
      {C_.data_ptr<float>(), n},  // ldc = N
      {C_.data_ptr<float>(), n},  // ldd = N
      {alpha, beta}
  };

  cutlass::Status status = gemm_op(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM failed, status = ", int(status));

  if (!C_.is_alias_of(C)) {
    C.copy_(C_);
  }
}


Tensor& cutlass_mm_out_cuda(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {

  TORCH_CHECK(self.device().is_cuda() && mat2.device().is_cuda(),
              "cutlass_mm_out_cuda: only CUDA tensors supported");

  TORCH_CHECK(self.scalar_type() == kFloat &&
              mat2.scalar_type() == kFloat,
              "cutlass_mm_out_cuda: only float32 for now");

  TORCH_CHECK(self.size(1) == mat2.size(0),
              "mm: self.size(1) must equal mat2.size(0)");

  out.resize_({self.size(0), mat2.size(1)});

  // C = A @ B  => alpha = 1, beta = 0
  run_cutlass_gemm_f32(self, mat2, out, /*alpha=*/1.f, /*beta=*/0.f);
  return out;
}


Tensor& cutlass_addmm_out_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {

  TORCH_CHECK(self.device().is_cuda() &&
              mat1.device().is_cuda() &&
              mat2.device().is_cuda(),
              "cutlass_addmm_out_cuda: only CUDA tensors supported");

  TORCH_CHECK(self.scalar_type() == kFloat &&
              mat1.scalar_type() == kFloat &&
              mat2.scalar_type() == kFloat,
              "cutlass_addmm_out_cuda: only float32 for now");

  TORCH_CHECK(mat1.size(1) == mat2.size(0), "addmm: K mismatch");
  TORCH_CHECK(self.size(0) == mat1.size(0) &&
              self.size(1) == mat2.size(1),
              "addmm: self shape must match mat1 @ mat2");

  out.resize_as_(self);
  out.copy_(self);

  float alpha_f = alpha.toFloat();
  float beta_f  = beta.toFloat();

  run_cutlass_gemm_f32(mat1, mat2, out, /*alpha=*/alpha_f, /*beta=*/beta_f);
  return out;
}

} // namespace native
} // namespace at


TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("mm.out",     TORCH_FN(at::native::cutlass_mm_out_cuda));
  m.impl("addmm.out",  TORCH_FN(at::native::cutlass_addmm_out_cuda));
}

