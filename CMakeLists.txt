cmake_minimum_required (VERSION 3.17)

project (cublas-reproducer)

find_package(CUDAToolkit 11.0 REQUIRED)
if(NOT TARGET CUDA::cublas)
    message(
      FATAL_ERROR
        "Target CUDA::cublas not found.")
  endif()

 add_executable (dcehd-BLAS-reproducer cuBLAS-Zgemm-solve.cpp)
 target_link_libraries (dcehd-BLAS-reproducer LINK_PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)