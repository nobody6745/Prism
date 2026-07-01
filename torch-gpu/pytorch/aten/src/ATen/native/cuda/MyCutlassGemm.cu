// aten/src/ATen/native/cuda/MyCutlassGemm.cu

#include <iostream>
#include <type_traits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/array.h" 
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/thread/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

#include "MyCutlassGemm.h"

namespace cutlass {

CUTLASS_HOST_DEVICE void print_debug_once() {
#if defined(__CUDA_ARCH__)
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      // printf("[DEBUG] PerturbedMixed Operator (16 ULP) HIT!\n");
  }
#endif
}

CUTLASS_HOST_DEVICE
static float ulp_down_float(float x) {
#if defined(__CUDA_ARCH__)
  union { float f; uint32_t u; } v;
  v.f = x;
  uint32_t u = v.u;
  
  if ((u & 0x7fffffffU) > 0x7f800000U) return x; // NaN
  if (u == 0xff800000U) return x;                // -inf
  
  if ((u & 0x7fffffffU) == 0) { v.u = 0x80000002U; return v.f; } // +/-0 -> 2nd -min
  
  if (u == 0x7f800000U) { v.u = 0x7f7ffffeU; return v.f; } // +inf -> 2nd max
  
  if ((u & 0x80000000U) == 0) v.u = u - 2;
  else                        v.u = u + 2;
  return v.f;
#else
  return std::nextafterf(std::nextafterf(x, -INFINITY), -INFINITY);
#endif
}

CUTLASS_HOST_DEVICE
static cutlass::half_t ulp_down_half(cutlass::half_t x) {
#if defined(__CUDA_ARCH__)
  __half h = reinterpret_cast<__half&>(x);
  union { __half h; uint16_t u; } v;
  v.h = h;
  uint16_t u = v.u;
  const uint16_t N = 2; 


  if ((u & 0x7c00U) == 0x7c00U && (u & 0x03ffU) != 0) return x;
  
  if (u == 0xfc00U) return x;
  
  if ((u & 0x8000U) == 0) {
      if (u >= N) {
          v.u = u - N; 
      } else {
          v.u = 0x8000U + (N - u);
      }
  }

  else {

      if ((u & 0x7fffU) + N >= 0x7c00U) {
          v.u = 0xfc00U;
      } else {
          v.u = u + N;
      }
  }

  return reinterpret_cast<cutlass::half_t&>(v.h);
#else
 
  float f = static_cast<float>(x);
  for(int i=0; i<16; ++i) f = std::nextafter(f, -INFINITY);
  return static_cast<cutlass::half_t>(f);
#endif
}


struct PerturbedFloat {
    float val;
    CUTLASS_HOST_DEVICE PerturbedFloat() : val(0.0f) {}
    CUTLASS_HOST_DEVICE PerturbedFloat(float v) : val(v) {}
    CUTLASS_HOST_DEVICE PerturbedFloat(cutlass::half_t v) : val(static_cast<float>(v)) {}
    
    CUTLASS_HOST_DEVICE operator float() const { return val; }
    
    CUTLASS_HOST_DEVICE bool operator==(const PerturbedFloat& rhs) const { return val == rhs.val; }
    CUTLASS_HOST_DEVICE bool operator!=(const PerturbedFloat& rhs) const { return val != rhs.val; }

    CUTLASS_HOST_DEVICE friend PerturbedFloat operator+(PerturbedFloat const &a, PerturbedFloat const &b) {
        return PerturbedFloat(cutlass::ulp_down_float(a.val + b.val));
    }
    CUTLASS_HOST_DEVICE friend PerturbedFloat operator*(PerturbedFloat const &a, PerturbedFloat const &b) {
        return PerturbedFloat(cutlass::ulp_down_float(a.val * b.val));
    }
    CUTLASS_HOST_DEVICE PerturbedFloat &operator+=(PerturbedFloat const &b) {
        this->val = cutlass::ulp_down_float(this->val + b.val); return *this;
    }
    CUTLASS_HOST_DEVICE PerturbedFloat &operator*=(PerturbedFloat const &b) {
        this->val = cutlass::ulp_down_float(this->val * b.val); return *this;
    }
};

struct PerturbedMixed {
    float val; 

    CUTLASS_HOST_DEVICE PerturbedMixed() : val(0.0f) {}
    CUTLASS_HOST_DEVICE PerturbedMixed(float v) : val(v) {}
    CUTLASS_HOST_DEVICE PerturbedMixed(double v) : val(static_cast<float>(v)) {}
    CUTLASS_HOST_DEVICE PerturbedMixed(int v) : val(static_cast<float>(v)) {}
    CUTLASS_HOST_DEVICE PerturbedMixed(cutlass::half_t v) : val(static_cast<float>(v)) {}

    CUTLASS_HOST_DEVICE explicit operator float() const { return val; }

    CUTLASS_HOST_DEVICE explicit operator cutlass::half_t() const { 
        cutlass::half_t res = static_cast<cutlass::half_t>(val);
        return ulp_down_half(res); 
    }
    
    CUTLASS_HOST_DEVICE explicit operator int() const {
        return static_cast<int>(val);
    }

    CUTLASS_HOST_DEVICE bool operator==(const PerturbedMixed& rhs) const { return val == rhs.val; }
    CUTLASS_HOST_DEVICE bool operator!=(const PerturbedMixed& rhs) const { return val != rhs.val; }

    CUTLASS_HOST_DEVICE friend PerturbedMixed operator+(PerturbedMixed const &a, PerturbedMixed const &b) {
        return PerturbedMixed(cutlass::ulp_down_float(a.val + b.val));
    }
    
    CUTLASS_HOST_DEVICE PerturbedMixed &operator+=(PerturbedMixed const &b) {
        this->val = cutlass::ulp_down_float(this->val + b.val); 
        return *this;
    }

    CUTLASS_HOST_DEVICE friend PerturbedMixed operator*(PerturbedMixed const &a, PerturbedMixed const &b) {
        return PerturbedMixed(a.val * b.val);
    }

    CUTLASS_HOST_DEVICE PerturbedMixed &operator*=(PerturbedMixed const &b) {
        this->val = this->val * b.val;
        return *this;
    }
    
    CUTLASS_HOST_DEVICE PerturbedMixed &operator-=(PerturbedMixed const &b) {
        this->val = cutlass::ulp_down_float(this->val - b.val); return *this;
    }
    CUTLASS_HOST_DEVICE PerturbedMixed &operator/=(PerturbedMixed const &b) {
        this->val = cutlass::ulp_down_float(this->val / b.val); return *this;
    }
};

CUTLASS_HOST_DEVICE PerturbedFloat conj(PerturbedFloat const &x) { return x; }
CUTLASS_HOST_DEVICE PerturbedMixed conj(PerturbedMixed const &x) { return x; }

template <>
struct multiply_add<PerturbedMixed, PerturbedMixed, PerturbedMixed> {
  CUTLASS_HOST_DEVICE
  PerturbedMixed operator()(
      PerturbedMixed const &a,
      PerturbedMixed const &b,
      PerturbedMixed const &c) const {
    return (a * b) + c;
  }
};

template <>
struct multiply_add<PerturbedMixed, cutlass::half_t, cutlass::half_t> {
  CUTLASS_HOST_DEVICE
  PerturbedMixed operator()(
      PerturbedMixed const &a, // Accumulator
      cutlass::half_t const &b, // A
      cutlass::half_t const &c  // B
      ) const {
    PerturbedMixed mb(b);
    PerturbedMixed mc(c);
    return (mb * mc) + a; 
  }
};

} // namespace cutlass

namespace std {

template <> struct numeric_limits<cutlass::PerturbedFloat> {
    static const bool is_specialized = true;
    static const bool is_signed = true;
    static const bool is_integer = false;
    static const bool is_exact = false;
    static const bool has_infinity = true;
    static const bool has_quiet_NaN = true;
    static const bool has_signaling_NaN = true;
    static const float_denorm_style has_denorm = denorm_present;
    static const bool has_denorm_loss = true;
    static const std::float_round_style round_style = round_to_nearest;
    static const bool is_iec559 = true;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const int digits = 24;
    CUTLASS_HOST_DEVICE static cutlass::PerturbedFloat min() { return cutlass::PerturbedFloat(1.17549435e-38f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedFloat max() { return cutlass::PerturbedFloat(3.40282347e+38f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedFloat lowest() { return cutlass::PerturbedFloat(-3.40282347e+38f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedFloat epsilon() { return cutlass::PerturbedFloat(1.19209290e-07f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedFloat round_error() { return cutlass::PerturbedFloat(0.5f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedFloat infinity() { return cutlass::PerturbedFloat(HUGE_VALF); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedFloat quiet_NaN() { return cutlass::PerturbedFloat(NAN); }
};

template <> struct numeric_limits<cutlass::PerturbedMixed> {
    static const bool is_specialized = true;
    static const bool is_signed = true;
    static const bool is_integer = false;
    static const bool is_exact = false;
    static const bool has_infinity = true;
    static const bool has_quiet_NaN = true;
    static const bool has_signaling_NaN = true;
    static const float_denorm_style has_denorm = denorm_present;
    static const bool has_denorm_loss = true;
    static const std::float_round_style round_style = round_to_nearest;
    static const bool is_iec559 = true;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const int digits = 24; 
    CUTLASS_HOST_DEVICE static cutlass::PerturbedMixed min() { return cutlass::PerturbedMixed(1.17549435e-38f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedMixed max() { return cutlass::PerturbedMixed(3.40282347e+38f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedMixed lowest() { return cutlass::PerturbedMixed(-3.40282347e+38f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedMixed epsilon() { return cutlass::PerturbedMixed(1.19209290e-07f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedMixed round_error() { return cutlass::PerturbedMixed(0.5f); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedMixed infinity() { return cutlass::PerturbedMixed(HUGE_VALF); }
    CUTLASS_HOST_DEVICE static cutlass::PerturbedMixed quiet_NaN() { return cutlass::PerturbedMixed(NAN); }
};
}

__global__ void my_cutlass_post_bump_f32(float* D) { if (blockIdx.x == 0 && threadIdx.x == 0) { } }
__global__ void my_cutlass_post_bump_f16(cutlass::half_t* D) { if (blockIdx.x == 0 && threadIdx.x == 0) { } }

namespace my_ops {

namespace detail {
inline void sync_for_debug() {
    if (std::getenv("CUTLASS_DEBUG_SYNC")) cudaDeviceSynchronize();
}
inline void print_host_debug_once() {
    static bool printed = false;
    if (!printed) {
        std::cout << "\n[MyCutlassGemm] >>> New Binary: Mixed Precision (Cast 16-ULP Down) <<<\n" << std::endl;
        printed = true;
    }
}
inline bool post_bump_enabled() { return (std::getenv("CUTLASS_POST_BUMP") != nullptr); }
} 

template <typename T> struct CutlassGemmConfig;

template <> struct CutlassGemmConfig<float> {
  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = cutlass::PerturbedFloat;
  using ElementCompute = cutlass::PerturbedFloat;

  using OpClass = cutlass::arch::OpClassSimt;
  using ArchTag = cutlass::arch::Sm80; 
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  static int const kAlignment = 1;
};

template <> struct CutlassGemmConfig<cutlass::half_t> {
  using ElementInput = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::PerturbedMixed; 
  using ElementCompute = cutlass::PerturbedMixed;

  using OpClass = cutlass::arch::OpClassSimt;
  using ArchTag = cutlass::arch::Sm50; 
  
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  static int const kAlignment = 1;
};


template <typename ElemT>
void run_cutlass_gemm_rr_ptr(
    ElemT* D, const ElemT* A, const ElemT* B,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta,
    cudaStream_t stream) {

  detail::print_host_debug_once();

  using Config = CutlassGemmConfig<ElemT>;
  using GemmRR = cutlass::gemm::device::Gemm<
      typename Config::ElementInput,  cutlass::layout::RowMajor,
      typename Config::ElementInput,  cutlass::layout::RowMajor,
      typename Config::ElementOutput, cutlass::layout::RowMajor,
      typename Config::ElementAccumulator, 
      typename Config::OpClass,
      typename Config::ArchTag,
      typename Config::ThreadblockShape,
      typename Config::WarpShape,
      typename Config::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          typename Config::ElementOutput,
          Config::kAlignment,
          typename Config::ElementAccumulator,
          typename Config::ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      2>;

  typename Config::ElementCompute alpha_val(alpha);
  typename Config::ElementCompute beta_val(beta);

  typename GemmRR::Arguments args(
      {M, N, K}, {A, lda}, {B, ldb}, {D, ldc}, {D, ldc},
      {alpha_val, beta_val});

  GemmRR gemm_op;
  cutlass::Status status = gemm_op(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM RR Failed");
  
  if (detail::post_bump_enabled()) {
    if constexpr (std::is_same_v<ElemT, float>) my_cutlass_post_bump_f32<<<1, 1, 0, stream>>>((float*)D);
    else my_cutlass_post_bump_f16<<<1, 1, 0, stream>>>((cutlass::half_t*)D);
  }
}

template <typename ElemT>
void run_cutlass_gemm_rc_ptr(
    ElemT* D, const ElemT* A, const ElemT* B,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta,
    cudaStream_t stream) {

  using Config = CutlassGemmConfig<ElemT>;
  using GemmRC = cutlass::gemm::device::Gemm<
      typename Config::ElementInput,  cutlass::layout::RowMajor,
      typename Config::ElementInput,  cutlass::layout::ColumnMajor,
      typename Config::ElementOutput, cutlass::layout::RowMajor,
      typename Config::ElementAccumulator,
      typename Config::OpClass,
      typename Config::ArchTag,
      typename Config::ThreadblockShape,
      typename Config::WarpShape,
      typename Config::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          typename Config::ElementOutput,
          Config::kAlignment,
          typename Config::ElementAccumulator,
          typename Config::ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      2>;

  typename Config::ElementCompute alpha_val(alpha);
  typename Config::ElementCompute beta_val(beta);

  typename GemmRC::Arguments args(
      {M, N, K}, {A, lda}, {B, ldb}, {D, ldc}, {D, ldc},
      {alpha_val, beta_val});

  GemmRC gemm_op;
  cutlass::Status status = gemm_op(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM RC Failed");
  if (detail::post_bump_enabled()) {
    if constexpr (std::is_same_v<ElemT, float>) my_cutlass_post_bump_f32<<<1, 1, 0, stream>>>((float*)D);
    else my_cutlass_post_bump_f16<<<1, 1, 0, stream>>>((cutlass::half_t*)D);
  }
}

template <typename ElemT>
void run_cutlass_bgemm_rr_ptr(
    ElemT* D, const ElemT* A, const ElemT* B,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    long long int batch_stride_a,
    long long int batch_stride_b,
    long long int batch_stride_c,
    int batch_count,
    float alpha, float beta,
    cudaStream_t stream) {

  using Config = CutlassGemmConfig<ElemT>;
  using GemmBatchedRR = cutlass::gemm::device::GemmBatched<
      typename Config::ElementInput,  cutlass::layout::RowMajor,
      typename Config::ElementInput,  cutlass::layout::RowMajor,
      typename Config::ElementOutput, cutlass::layout::RowMajor,
      typename Config::ElementAccumulator,
      typename Config::OpClass,
      typename Config::ArchTag,
      typename Config::ThreadblockShape,
      typename Config::WarpShape,
      typename Config::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          typename Config::ElementOutput,
          Config::kAlignment,
          typename Config::ElementAccumulator,
          typename Config::ElementCompute>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      2>;

  typename Config::ElementCompute alpha_val(alpha);
  typename Config::ElementCompute beta_val(beta);

  typename GemmBatchedRR::Arguments args(
      {M, N, K},
      {A, lda}, batch_stride_a,
      {B, ldb}, batch_stride_b,
      {D, ldc}, batch_stride_c,
      {D, ldc}, batch_stride_c,
      {alpha_val, beta_val},
      batch_count);

  GemmBatchedRR gemm_op;
  cutlass::Status status = gemm_op(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS Batched GEMM Failed: ", cutlassGetStatusString(status));
  
  if (detail::post_bump_enabled()) {
    if constexpr (std::is_same_v<ElemT, float>) my_cutlass_post_bump_f32<<<1, 1, 0, stream>>>((float*)D);
    else my_cutlass_post_bump_f16<<<1, 1, 0, stream>>>((cutlass::half_t*)D);
  }
}



void run_cutlass_gemm_raw(
    void* D, const void* A, const void* B, const void* C,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    char transa, char transb,
    float alpha, float beta,
    MyGemmDType dtype,
    c10::cuda::CUDAStream stream) {

  TORCH_CHECK(transa == 'n' || transa == 'N', "Only support transA='N'");
  bool is_b_trans = (transb == 't' || transb == 'T');
  TORCH_CHECK(D == C, "Expect C and D to be the same ptr");
  cudaStream_t cuda_stream = stream.stream();

  if (dtype == MyGemmDType::F32) {
    if (is_b_trans) run_cutlass_gemm_rc_ptr<float>((float*)D, (const float*)A, (const float*)B, m, n, k, lda, ldb, ldc, alpha, beta, cuda_stream);
    else            run_cutlass_gemm_rr_ptr<float>((float*)D, (const float*)A, (const float*)B, m, n, k, lda, ldb, ldc, alpha, beta, cuda_stream);
  } 
  else if (dtype == MyGemmDType::F16) {
    if (is_b_trans) run_cutlass_gemm_rc_ptr<cutlass::half_t>((cutlass::half_t*)D, (const cutlass::half_t*)A, (const cutlass::half_t*)B, m, n, k, lda, ldb, ldc, alpha, beta, cuda_stream);
    else            run_cutlass_gemm_rr_ptr<cutlass::half_t>((cutlass::half_t*)D, (const cutlass::half_t*)A, (const cutlass::half_t*)B, m, n, k, lda, ldb, ldc, alpha, beta, cuda_stream);
  } 
  else {
    TORCH_CHECK(false, "Unsupported DType");
  }
  detail::sync_for_debug();
}

void run_cutlass_bgemm_raw(
    void* D, const void* A, const void* B, const void* C,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    long long int batch_stride_a,
    long long int batch_stride_b,
    long long int batch_stride_c,
    int64_t batch_count,
    char transa, char transb,
    float alpha, float beta,
    MyGemmDType dtype,
    c10::cuda::CUDAStream stream) {

  TORCH_CHECK(transa == 'n' || transa == 'N', "Batched: Only support transA='N'");
  TORCH_CHECK(transb == 'n' || transb == 'N', "Batched: Only support transB='N' for now");
  cudaStream_t cuda_stream = stream.stream();

  if (dtype == MyGemmDType::F32) {
      run_cutlass_bgemm_rr_ptr<float>((float*)D, (const float*)A, (const float*)B, (int)m, (int)n, (int)k, (int)lda, (int)ldb, (int)ldc, batch_stride_a, batch_stride_b, batch_stride_c, (int)batch_count, alpha, beta, cuda_stream);
  } 
  else if (dtype == MyGemmDType::F16) {
      run_cutlass_bgemm_rr_ptr<cutlass::half_t>((cutlass::half_t*)D, (const cutlass::half_t*)A, (const cutlass::half_t*)B, (int)m, (int)n, (int)k, (int)lda, (int)ldb, (int)ldc, batch_stride_a, batch_stride_b, batch_stride_c, (int)batch_count, alpha, beta, cuda_stream);
  } 
  else {
    TORCH_CHECK(false, "Batched: Unsupported DType");
  }
  detail::sync_for_debug();
}

} // namespace my_ops
