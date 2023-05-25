cmake_minimum_required (VERSION 3.17)

project (cublas-reproducer
      LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(CUDAToolkit 11.0 REQUIRED)
if(NOT TARGET CUDA::cublas)
    message(
      FATAL_ERROR
        "Target CUDA::cublas not found.")
  endif()

add_executable (dcehd-BLAS-reproducer cuBLAS-ZGEMM-solve.cpp)
target_link_libraries (dcehd-BLAS-reproducer LINK_PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)
