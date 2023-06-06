#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

typedef cuComplex cuComplexFloat;  // Define cuComplexFloat as cuComplex for simplicity

int main() {
    // Set the matrix dimensions
    const int N = 2;
    const int M = 2;
    const int K = 2;

    // Initialize input matrices
    float A_real[N * K] = {1.0f, 2.0f, 3.0f, 4.0f};
    float A_imag[N * K] = {0.0f, 0.0f, 0.0f, 0.0f};
    float B_real[K * M] = {1.0f, 0.0f, 0.0f, 1.0f};
    float B_imag[K * M] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Cast input matrices to cuComplex
    cuComplexFloat* d_A;
    cuComplexFloat* d_B;
    cudaMalloc((void**)&d_A, N * K * sizeof(cuComplexFloat));
    cudaMalloc((void**)&d_B, K * M * sizeof(cuComplexFloat));

    // Copy input matrices to device
    cudaMemcpy(d_A, reinterpret_cast<cuComplexFloat*>(A_real), N * K * sizeof(cuComplexFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<cuComplexFloat*>(A_imag), reinterpret_cast<cuComplexFloat*>(A_imag), N * K * sizeof(cuComplexFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, reinterpret_cast<cuComplexFloat*>(B_real), K * M * sizeof(cuComplexFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<cuComplexFloat*>(B_imag), reinterpret_cast<cuComplexFloat*>(B_imag), K * M * sizeof(cuComplexFloat), cudaMemcpyHostToDevice);

    // Allocate memory for the output matrix C
    cuComplexFloat* d_C;
    cudaMalloc((void**)&d_C, N * M * sizeof(cuComplexFloat));

    // Create a handle for the cuBLAS library
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define alpha and beta values for the matrix multiplication
    cuComplexFloat alpha = make_cuComplex(1.0f, 0.0f);
    cuComplexFloat beta = make_cuComplex(0.0f, 0.0f);

    // Perform the matrix multiplication
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_A, N, d_B, K, &beta, d_C, N);

    // Wait for the GPU to finish executing the matrix multiplication
    cudaDeviceSynchronize();

    // Copy the result matrix C back to the host
    cuComplexFloat C[N * M];
    cudaMemcpy(C, d_C, N * M * sizeof(cuComplexFloat), cudaMemcpyDeviceToHost);

    // Print the result matrix C
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < N * M; i++) {
        std::cout << "Real: " << cuCrealf(C[i]) << " Imaginary: " << cuCimagf(C[i]) << std::endl;
    }
}
