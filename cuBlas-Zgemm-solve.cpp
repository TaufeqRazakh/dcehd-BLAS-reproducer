#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define cublasErrorCheck(ans, cause)                \
  {                                                 \
    cublasAssert((ans), cause, __FILE__, __LINE__); \
  }

int main(int argc, char **argv)
{
	std::cout << "Starting complex double matmul of size lXn and nXm with cuBlas";
  
  int m = 16;
  int n = 16;
  int k = 42875;
  const std::complex<RealType> cDvol = std::complex<RealType>(Dvol);

  // Define A matrix
  // TODO: A must be of type complex double
  float* devPtrA;
  float* A = 0;
  A = (float *)malloc (32 * 42875 * sizeof (*a));
  if (!A) {
      printf ("host memory allocation failed");
      return EXIT_FAILURE;
  }
  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          A[IDX2F(i,j,M)] = (float)((i-1) * N + j);
      }
  }
  cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (M, N, sizeof(*A), A, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (devPtrA);
      ublasDestroy(handle);
      return EXIT_FAILURE;
  }

  // Define B matrix
  // TODO: A must be of type complex double
  float* devPtrA;
  float* A = 0;
  A = (float *)malloc (32 * 42875 * sizeof (*a));
  if (!A) {
      printf ("host memory allocation failed");
      return EXIT_FAILURE;
  }
  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          A[IDX2F(i,j,M)] = (float)((i-1) * N + j);
      }
  }
  cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (M, N, sizeof(*A), A, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (devPtrA);
      ublasDestroy(handle);
      return EXIT_FAILURE;
  }

  // Set and define cuda linear algebra handles
  cudaStream_t hstream;
  cublasHandle_t h_cublas;

  cudaErrorCheck(cudaStreamCreate(&hstream), "cudaStreamCreate failed!");
  cublasErrorCheck(cublasCreate(&h_cublas), "cublasCreate failed!");
  cublasErrorCheck(cublasSetStream(h_cublas, hstream), "cublasSetStream failed!");

  // Use CUDA handles here
  // the original blas function was
  // cublasErrircheck(cublasZgeam(cuda_linear_algebra_handles.h_cublas, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, castNativeType(alpha), castNativeType(A), lda, castNativeType(B), ldb,
  //                 castNativeType(beta), castNativeType(C), ldc);
  cublasErrorCheck(qmcplusplus::cuBLAS::gemm(cuda_linear_algebra_handles.h_cublas, CUBLAS_OP_N, CUBLAS_OP_C, nlumo,
                                               Next, Nxyz, &cDvol, psi_core_ptr, Norb, psi0_core_ptr + nlumo, Norb,
                                               &czero, ovlp_m_ptr, nlumo),


  // Destroy CUDA handles

  cublasErrorCheck(cublasDestroy(h_cublas), "cublasDestroy failed!");
  cudaErrorCheck(cudaStreamDestroy(hstream), "cudaStreamDestroy failed!");
}