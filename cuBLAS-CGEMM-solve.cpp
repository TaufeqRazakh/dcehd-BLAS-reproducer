#include "cublas_v2.h"
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#define M 32
#define N 32
#define K 42875

#define cudaErrorCheck(ans, cause)                                             \
  { cudaAssert((ans), cause, __FILE__, __LINE__); }

#define cublasErrorCheck(ans, cause)                                           \
  { cublasAssert((ans), cause, __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const std::string &cause,
                       const char *filename, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::ostringstream err;
    err << "cudaAssert: " << cudaGetErrorName(code) << " "
        << cudaGetErrorString(code) << ", file " << filename << ", line "
        << line << std::endl
        << cause << std::endl;
    std::cerr << err.str();
    if (abort)
      throw std::runtime_error(cause);
  }
}

inline void cublasAssert(cublasStatus_t code, const std::string &cause,
                         const char *file, int line, bool abort = true) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream err;
    err << "cublasAssert: from file " << file << " , line " << line << std::endl
        << cause << std::endl;
    std::cerr << err.str();
    // if (abort) exit(code);
    throw std::runtime_error(cause);
  }
}

int main(int argc, char **argv) {
  std::cout << "Testing complex float matmul with matrices A (42,875 x 32) and "
               "B a(42,875 x 32) to produce a matrix C (16 x 16)"
            << std::endl;
  int i, j;
  int m = 16;
  int n = 16;
  int k = 42875;
  const std::complex<float> alpha = std::complex<float>(3);
  const std::complex<float> beta = std::complex<float>(7);
  int lda = 32;
  int ldb = 32;
  int ldc = 16;

  // Set and define cuda linear algebra handles
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cudaStream_t hstream;
  cublasHandle_t h_cublas;

  cudaErrorCheck(cudaStreamCreate(&hstream), "cudaStreamCreate failed!");
  cublasErrorCheck(cublasCreate(&h_cublas), "cublasCreate failed!");
  cublasErrorCheck(cublasSetStream(h_cublas, hstream),
                   "cublasSetStream failed!");

  // Define A matrix
  std::complex<float> *devPtrA;
  std::complex<float> *A = 0;
  // Assign A on host
  A = (std::complex<float> *)malloc(M * K * sizeof(*A));
  if (!A) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  // Initialize A
  for (j = 0; j < K * M; j++) {
    A[j] = std::complex<double>(2, j);
  }
  // Create memory for A on device
  cudaStat = cudaMalloc((void **)&devPtrA, M * K * sizeof(*A));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  // Move date from host to device
  stat = cublasSetMatrix(M, K, sizeof(*A), A, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cublasDestroy(h_cublas);
    return EXIT_FAILURE;
  }

  // Define B matrix
  std::complex<float> *devPtrB;
  std::complex<float> *B = 0;
  B = (std::complex<float> *)malloc(N * K * sizeof(*B));
  if (!B) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  for (j = 0; j < K * M; j++) {
    B[j] = std::complex<double>(2, j);
  }
  cudaStat = cudaMalloc((void **)&devPtrB, N * K * sizeof(*B));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(N, K, sizeof(*B), B, N, devPtrB, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrB);
    cublasDestroy(h_cublas);
    return EXIT_FAILURE;
  }

  // Define resulting C matrix
  std::complex<float> *devPtrC;
  std::complex<float> *C = 0;
  C = (std::complex<float> *)malloc(m * n * sizeof(*C));

  cudaStat = cudaMalloc((void **)&devPtrC, m * n * sizeof(*C));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }

  cublasCgemm(h_cublas, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k,
                     reinterpret_cast<const cuComplex *>(&alpha),
                     reinterpret_cast<const cuComplex *>(A), lda,
                     reinterpret_cast<const cuComplex *>(B + 16), ldb,
                     reinterpret_cast<const cuComplex *>(&beta),
                     reinterpret_cast<cuComplex *>(C), ldc);

  // wait for stream to complete
  cudaErrorCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed!");
  std::cout << "Blas call has finished" << std::endl;

  // Destroy CUDA handles
  cublasErrorCheck(cublasDestroy(h_cublas), "cublasDestroy failed!");
  cudaErrorCheck(cudaStreamDestroy(hstream), "cudaStreamDestroy failed!");
}
