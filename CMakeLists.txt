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

add_library(cuda-essential INTERFACE)
target_link_libraries (cuda-essential INTERFACE CUDA::cudart CUDA::cublas CUDA::cusolver) 

add_executable (zgemm-test cuBLAS-ZGEMM-solve.cpp)
target_link_libraries (zgemm-test cuda-essential) 

add_executable (cgemm-test cuBLAS-CGEMM-solve.cpp)
target_link_libraries (cgemm-test cuda-essential)

add_executable (simplified-cgemm-test simplified-cuBLAS-CGEMM-solve.cpp)
target_link_libraries (simplified-cgemm-test cuda-essential)
