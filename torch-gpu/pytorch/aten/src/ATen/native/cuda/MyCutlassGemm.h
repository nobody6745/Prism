#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/Scalar.h>
#include <c10/cuda/CUDAStream.h>

namespace my_ops {

void run_cutlass_gemm_dispatch(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Scalar& beta);

void run_cutlass_gemm_dispatch_linear(
    at::Tensor& result,
    const at::Tensor& x,     // [M,K]
    const at::Tensor& w_t,   // view: [K,N] = w.t()
    const at::Tensor& w,     // original weight: [N,K] row-major contiguous
    const at::Scalar& alpha,
    const at::Scalar& beta);

enum class MyGemmDType : int {
  F16 = 0,
  F32 = 1,
};

void run_cutlass_gemm_raw(
    void* D,
    const void* A,
    const void* B,
    const void* C,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    char transa, char transb,   // 'n' or 't'
    float alpha, float beta,
    MyGemmDType dtype,
    c10::cuda::CUDAStream stream);

void run_cutlass_bgemm_raw(
    void* D,
    const void* A,
    const void* B,
    const void* C,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    long long int batch_stride_a,
    long long int batch_stride_b,
    long long int batch_stride_c,
    int64_t batch_count,
    char transa, char transb,
    float alpha, float beta,
    MyGemmDType dtype,
    c10::cuda::CUDAStream stream);

} // namespace my_ops
