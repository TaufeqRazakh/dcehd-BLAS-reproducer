#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <complex>
#define M 32
#define N 42875
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define cudaErrorCheck(ans, cause)              \
{                                               \
  cudaAssert((ans), cause, __FILE__, __LINE__); \
}

#define cublasErrorCheck(ans, cause)              \
{                                                 \
  cublasAssert((ans), cause, __FILE__, __LINE__); \
}

inline void cudaAssert(cudaError_t code, const std::string& cause, const char* filename, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    std::ostringstream err;
    err << "cudaAssert: " << cudaGetErrorName(code) << " " << cudaGetErrorString(code) << ", file " << filename
        << ", line " << line << std::endl
        << cause << std::endl;
    std::cerr << err.str();
    if (abort)
      throw std::runtime_error(cause);
  }
}

inline void cublasAssert(cublasStatus_t code, const std::string& cause, const char* file, int line, bool abort = true)
{
  if (code != CUBLAS_STATUS_SUCCESS)
  {
    std::ostringstream err;
    err << "cublasAssert: from file " << file << " , line " << line << std::endl
        << cause << std::endl;
    std::cerr << err.str();
    //if (abort) exit(code);
    throw std::runtime_error(cause);
  }
}

int main(int argc, char **argv)
{
  std::cout << "Starting complex double matmul of size lXn and nXm with cuBlas";
  int i, j;  
  int m = 16;
  int n = 16;
  int k = 42875;
  const std::complex<double> cDvol = std::complex<double>(3);
 
  // Set and define cuda linear algebra handles
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cudaStream_t hstream;
  cublasHandle_t h_cublas;

  cudaErrorCheck(cudaStreamCreate(&hstream), "cudaStreamCreate failed!");
  cublasErrorCheck(cublasCreate(&h_cublas), "cublasCreate failed!");
  cublasErrorCheck(cublasSetStream(h_cublas, hstream), "cublasSetStream failed!");

  // Define A matrix
  std::complex<double>* devPtrA;
  std::complex<double>* A = 0;
  // Assign A on host
  A = (std::complex<double>*)malloc (32 * 42875 * sizeof (*A));
  if (!A) {
      printf ("host memory allocation failed");
      return EXIT_FAILURE;
  }
  // Initialize A
  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          A[IDX2C(i,j,M)] = (float)(i * N + j + 1);
      }
  }
  // Create memory for A on device
  cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*A));
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  // Move date from host to device
  stat = cublasSetMatrix (M, N, sizeof(*A), A, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (devPtrA);
      cublasDestroy(h_cublas);
      return EXIT_FAILURE;
  }

  // Define B matrix
  std::complex<double>* devPtrB;
  std::complex<double>* B = 0;
  B = (std::complex<double>*)malloc (32 * 42875 * sizeof (*B));
  if (!B) {
      printf ("host memory allocation failed");
      return EXIT_FAILURE;
  }
  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          B[IDX2C(i,j,M)] = (float)(i * N + j + 1);
      }
  }
  cudaStat = cudaMalloc ((void**)&devPtrB, M*N*sizeof(*B));
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (M, N, sizeof(*B), B, M, devPtrB, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (devPtrB);
      cublasDestroy(h_cublas);
      return EXIT_FAILURE;
  }

  // Define resulting C matrix
  std::complex<double>* devPtrC;
  std::complex<double>* C = 0;
  C = (std::complex<double>*)malloc (16 * 42875 * sizeof (*B));
  if (!B) {
      printf ("host memory allocation failed");
      return EXIT_FAILURE;
  }
  for (j = 1; j <= 16; j++) {
      for (i = 1; i <= M; i++) {
          B[IDX2C(i,j,M)] = (float)(i * N + j + 1);
      }
  }
  cudaStat = cudaMalloc ((void**)&devPtrC, M*N*sizeof(*C));
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (M, N, sizeof(*C), C, M, devPtrB, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (devPtrC);
      cublasDestroy(h_cublas);
      return EXIT_FAILURE;
  }

  // Use CUDA handles here
  // the original blas function was
  //cublasErrorCheck(qmcplusplus::cuBLAS::gemm(cuda_linear_algebra_handles.h_cublas, CUBLAS_OP_N, CUBLAS_OP_C, nlumo,
  //                                             Next, Nxyz, &cDvol, psi_core_ptr, Norb, psi0_core_ptr + nlumo, Norb,
  //                                             &czero, ovlp_m_ptr, nlumo),
  // The problem BLAS call
  return cublasZgemm(h_cublas, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&cDvol), reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B + nlumo), ldb,
                     reinterpret_cast<const cuDoubleComplex*>(cDvol), reinterpret_cast<const cuDoubleComplex*>(C), ldc);
  
  // wait for stream to complete
  cudaDeviceSynchronize();

  // Destroy CUDA handles

  cublasErrorCheck(cublasDestroy(h_cublas), "cublasDestroy failed!");
  cudaErrorCheck(cudaStreamDestroy(hstream), "cudaStreamDestroy failed!");
}
