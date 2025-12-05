/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutlass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <cublas_v2.h>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////


// 内部实现：支持任意 SmArch
template <typename T, typename SmArch,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
cudaError_t CutlassSgemmNN_impl(
  int M, int N, int K,
  float alpha,
  T const *A, int lda,
  T const *B, int ldb,
  float beta,
  float *C, int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using CutlassType = std::conditional_t<std::is_same_v<T, half>, cutlass::half_t, T>;
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // Tile shapes tuned for each architecture
  using ShapeMMAThreadBlock = std::conditional_t<
    std::is_same_v<SmArch, cutlass::arch::Sm80>,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<128, 128, 32>  // Sm70 uses K=32 for Tensor Core
  >;

  using ShapeMMAWarp = std::conditional_t<
    std::is_same_v<SmArch, cutlass::arch::Sm80>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<64, 64, 32>
  >;

  using ShapeMMAOp = std::conditional_t<
    std::is_same_v<SmArch, cutlass::arch::Sm80>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::gemm::GemmShape<8, 8, 4>   // V100: MMA 8x8x4 for fp16
  >;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      float,
      128 / (sizeof(float) * 8),
      float,
      float>;

  constexpr int NumStages = std::is_same_v<SmArch, cutlass::arch::Sm80> ? 4 : 2;

  using CutlassGemmOpt = cutlass::gemm::device::Gemm<
      CutlassType, ColumnMajor,
      CutlassType, ColumnMajor,
      float, ColumnMajor,
      float,
      MMAOp,
      SmArch,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>;

  typename CutlassGemmOpt::Arguments args(
      {M, N, K},
      {reinterpret_cast<CutlassType const*>(A), lda},
      {reinterpret_cast<CutlassType const*>(B), ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta});

  CutlassGemmOpt gemm_operator;
  cutlass::Status status = gemm_operator(args);

  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
// 仅当T是half或float时启用
template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  int sm_major, int sm_minor) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`
  if (sm_major == 7 && sm_minor == 0) {
    return CutlassSgemmNN_impl<T, cutlass::arch::Sm70>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (sm_major == 8 && sm_minor == 0) {
    return CutlassSgemmNN_impl<T, cutlass::arch::Sm80>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    std::cerr << "Unsupported SM version: " << sm_major << "." << sm_minor << std::endl;
    return cudaErrorNotSupported;
  }



//   using ColumnMajor = cutlass::layout::ColumnMajor;

//   using CutlassType = std::conditional_t<std::is_same_v<T, half>, cutlass::half_t, T>;

//   // using CutlassGemm = cutlass::gemm::device::Gemm<CutlassType,        // Data-type of A matrix
//   //                                                 ColumnMajor,        // Layout of A matrix
//   //                                                 CutlassType,        // Data-type of B matrix
//   //                                                 ColumnMajor,  // Layout of B matrix
//   //                                                 float,        // Data-type of C matrix
//   //                                                 ColumnMajor>; // Layout of C matrix

//   // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
//   using MMAOp = cutlass::arch::OpClassTensorOp;

//   // This code section describes CUDA SM architecture number
//   using SmArch = cutlass::arch::Sm80;

//   // This code section describes the tile size a thread block will compute
//   using ShapeMMAThreadBlock =
//       cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 32
//   // This code section describes tile size a warp will compute
//   using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 32 
//   // This code section describes the size of MMA op
//   using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 8, N = 8, K = 4

//   // This code section describes how threadblocks are scheduled on GPU
//   using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

//   // This code section describes ?
//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
//       float,                                     // <- data type of output matrix
//       128 / (sizeof(float)*8),  // <- this is the number of elements per
//                                                         // vectorized memory access. For half
//                                                         // precision, it's 8 elements. This becomes
//                                                         // the vector width of math instructions in
//                                                         // epilogue too
//       float,                                // <- data type of accumulator
//       float>;  // <- data type for alpha/beta in linear combination function <- data type of epilogue operations

//   // Number of pipelines you want to use
//   constexpr int NumStages = 4;

//   using CutlassGemmOpt = cutlass::gemm::device::Gemm<CutlassType,
//                                           ColumnMajor,
//                                           CutlassType,
//                                           ColumnMajor,
//                                           float,
//                                           ColumnMajor,
//                                           float,
//                                           MMAOp,
//                                           SmArch,
//                                           ShapeMMAThreadBlock,
//                                           ShapeMMAWarp,
//                                           ShapeMMAOp,
//                                           EpilogueOp,
//                                           SwizzleThreadBlock,
//                                           NumStages>;

//   // Define a CUTLASS GEMM type
//   CutlassGemmOpt gemm_operator;

//   // Construct the CUTLASS GEMM arguments object.
//   //
//   // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
//   // in host code and passed to kernels by value. These may include pointers, strides, scalars,
//   // and other arguments needed by Gemm and its components.
//   //
//   // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
//   // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
//   // 关键：用typename声明依赖类型Arguments
//   // 对CutlassGemm::Arguments添加typename关键字，解决编译器无法识别依赖类型的问题
//   // typename CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
//   //                             {reinterpret_cast<CutlassType const*>(A), lda},    // Tensor-ref for source matrix A
//   //                             {reinterpret_cast<CutlassType const*>(B), ldb},    // Tensor-ref for source matrix B
//   //                             {C, ldc},    // Tensor-ref for source matrix C
//   //                             {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
//   //                             {alpha, beta}); // Scalars used in the Epilogue

//   // Create a tuple of problem size for matrix multiplication
//   // cutlass::gemm::GemmCoord problem_size(M, N, K);

//   // // Initialize tensors using CUTLASS helper functions
//   // cutlass::HostTensor<CutlassType, ColumnMajor> tensor_a(
//   //     problem_size.mk());  // <- Create matrix A with dimensions M x K
//   // cutlass::HostTensor<CutlassType, ColumnMajor> tensor_b(
//   //     problem_size.kn());  // <- Create matrix B with dimensions K x N
//   // cutlass::HostTensor<float, ColumnMajor> tensor_c(
//   //     problem_size.mn());  // <- Create matrix C with dimensions M x N
//   // cutlass::HostTensor<float, ColumnMajor> tensor_d(
//   //     problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
//   //                          // CUTLASS kernel     

//   typename CutlassGemmOpt::Arguments args({M, N, K},  // Gemm Problem dimensions
//                             {reinterpret_cast<CutlassType const*>(A), lda},    // Tensor-ref for source matrix A
//                             {reinterpret_cast<CutlassType const*>(B), ldb},    // Tensor-ref for source matrix B
//                             {C, ldc},    // Tensor-ref for source matrix C
//                             {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
//                             {alpha, beta}); // Scalars used in the Epilogue                         

//   //
//   // Launch the CUTLASS GEMM kernel.
//   //
  
//   cutlass::Status status = gemm_operator(args);

//   //
//   // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
//   //

//   if (status != cutlass::Status::kSuccess) {
//     return cudaErrorUnknown;
//   }

//   // Return success, if no errors were encountered.
//   return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    // 根据T的类型赋值（编译期分支）
    if constexpr (std::is_same_v<T, float>) {
      matrix[offset] = value;  // float直接赋值
    }
    else if constexpr (std::is_same_v<T, half>) {
      matrix[offset] = __float2half(value);  // half需转换
    }

    // matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
cudaError_t InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
cudaError_t AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      T a_val = A[i + k * lda];
      T b_val = B[k + j * ldb];

      // 编译期分支：half直接计算，float直接计算
      accumulator += static_cast<float>(A[i + k * lda]) * static_cast<float>(B[k + j * ldb]);
    }
    // 将累加结果转为float，统一赋值给C矩阵（float类型）
    // float final_accum = static_cast<float>(accumulator);
    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
cudaError_t cublasGemm(cublasHandle_t handle,
  int M,
  int N,
  int K,
  float alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  float beta,
  float *C,
  int ldc){

  // 根据类型选择数据类型和标量
  cudaDataType_t data_type = std::is_same_v<T, half> ? CUDA_R_16F : CUDA_R_32F;
  void const *alpha_ptr = &alpha;
  void const *beta_ptr = &beta;

  // if constexpr (std::is_same_v<T, half>) {
  //   half alpha_h = __float2half(alpha);
  //   half beta_h = __float2half(beta);
  //   alpha_ptr = &alpha_h;
  //   beta_ptr = &beta_h;
  // } else {
    // alpha_ptr = &alpha;
    // beta_ptr = &beta;
  // }
  // 新版cublasGemmEx需补充computeType和algo参数
  cudaDataType_t compute_type = CUDA_R_32F;  // 计算精度（用float累加）
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;  // 默认算法
  // cuBLAS GEMM参数：(opB, opA, n, m, k, alpha, B, typeB, ldb, A, typeA, lda, beta, C, typeC, ldc)
  // 对应：C(m×n) = alpha * A(m×k) * B(k×n) + beta * C(m×n)
  cublasStatus_t cublas_status = cublasGemmEx(
    handle,
    CUBLAS_OP_N,      // A矩阵不转置
    CUBLAS_OP_N,      // A矩阵不转置
    M,                // A的行数m
    N,                // B的列数n
    K,                // A的列数/B的行数k
    alpha_ptr,        // 标量alpha
    A,                // 矩阵A (mxk)
    data_type,        // A的数据类型
    lda,              // A的leading dimension（列主序为m）
    B,                // 矩阵B (kxn)
    data_type,        // B的数据类型
    ldb,              // B的leading dimension（列主序为k）
    beta_ptr,         // 标量beta
    C,                // 输出矩阵C (m×n)
    CUDA_R_32F,       // C的数据类型（float）
    ldc,              // C的leading dimension（列主序为m）
    compute_type,
    algo
  );


  // 错误处理
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS GEMM failed: " << cublas_status << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
template <typename T, std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta, int sm_major, int sm_minor) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  // float *A;
  T *A;
  // float *B;
  T *B;
  float *C_cutlass;
  float *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Warm-up
  std::cout << "Running warm-up..." << std::endl;
  for (int i = 0; i < 5; ++i) {
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, sm_major, sm_minor);
    if (result != cudaSuccess) { /* 释放内存 */ return result; }
  }
  cudaDeviceSynchronize();

  // Time Statistic
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, sm_major, sm_minor);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  std::cout << "Cutlass GEMM time: " << elapsed_ms << " ms" << std::endl;

  // 性能计算
  double flops = 2.0 * M * N * K;
  double gflops = flops / (elapsed_ms * 1e6);
  std::cout << "Cutlass GEMM Performance: " << gflops << " GFLOPS" << std::endl;

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Verify.
  //



  // Launch reference blas GEMM
  // 创建cuBLAS句柄
  cublasHandle_t handle;
  cublasStatus_t cublas_status = cublasCreate_v2(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS handle creation failed: " << cublas_status << std::endl;
    return cudaErrorUnknown;
  }
  // result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
  cudaEvent_t start_blas, stop_blas;
  cudaEventCreate(&start_blas);
  cudaEventCreate(&stop_blas);
  cudaEventRecord(start_blas);

  result = cublasGemm(handle, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  cudaEventRecord(stop_blas);
  cudaEventSynchronize(stop_blas);
  float elapsed_ms_blas;
  cudaEventElapsedTime(&elapsed_ms_blas, start_blas, stop_blas);
  std::cout << "Cublas GEMM time: " << elapsed_ms_blas << " ms" << std::endl;

  // 性能计算
  double flops_blas = 2.0 * M * N * K;
  double gflops_blas = flops / (elapsed_ms_blas * 1e6);
  std::cout << "Cublas GEMM Performance: " << gflops_blas << " GFLOPS" << std::endl;

  
  // 销毁句柄
  cublasDestroy_v2(handle);

  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<float> host_cutlass(ldc * N, 0);
  std::vector<float> host_reference(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  // -------------------------- 精度内一致性验证 --------------------------
  bool is_correct = true;
  const float abs_error_threshold = std::is_same_v<T, half> ? 1e-3 : 1e-5;  // 绝对误差阈值（float用1e-5，half用1e-3）
  const float rel_error_threshold = std::is_same_v<T, half> ? 1e-2 : 1e-4;  // 相对误差阈值（可选）
  int error_count = 0;
  const int max_error_print = 5;  // 最多打印前5个误差元素

  for (size_t idx = 0; idx < host_cutlass.size(); ++idx) {
      float cutlass_val = host_cutlass[idx];
      float ref_val = host_reference[idx];
      
      // 计算绝对误差和相对误差
      float abs_error = fabs(cutlass_val - ref_val);
      float rel_error = (ref_val != 0) ? (abs_error / fabs(ref_val)) : abs_error;

      // 判断是否超过阈值（满足任一阈值即可）
      if (abs_error > abs_error_threshold || rel_error > rel_error_threshold) {
          is_correct = false;
          error_count++;
          
          // 打印前几个误差元素（便于调试）
          if (error_count <= max_error_print) {
              std::cerr << "Error at index " << idx 
                        << ": Cutlass=" << cutlass_val 
                        << ", Reference=" << ref_val 
                        << ", AbsError=" << abs_error 
                        << ", RelError=" << rel_error << std::endl;
          }
      }
  }

  // 输出验证结果
  if (!is_correct) {
      std::cerr << "CUTLASS results incorrect! Total errors: " << error_count << std::endl;
      return cudaErrorUnknown;
  } else {
      std::cout << "All results match within tolerance (abs=" << abs_error_threshold 
                << ", rel=" << rel_error_threshold << ")" << std::endl;
  }

  //
  // Test for bit equivalence of results.
  //

  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  // Get device properties
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  int sm_major = deviceProp.major;
  int sm_minor = deviceProp.minor;

  std::cout << "Detected GPU: " << deviceProp.name
            << " (SM " << sm_major << "." << sm_minor << ")" << std::endl;

  // Only support V100 (7.0) and A100 (8.0)
  if (!((sm_major == 7 && sm_minor == 0) || (sm_major == 8 && sm_minor == 0))) {
    std::cerr << "This example only supports V100 (SM 7.0) or A100 (SM 8.0)." << std::endl;
    return -1;
  }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm<half>(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],      // beta
    sm_major,
    sm_minor
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
