#### Build Steps
```
mkdir build_artifact
cd build_artifact
cmake -DCMAKE_CXX_COMPILER=clang++ ..
make
```
#### Run Steps
```
./zgemm-test 
./cgemm-test 
```

#### Issues

+ `zgemm-test` <br/> cudaErrorIllegalAddress an illegal memory access was encountered, file /home/razakh/dcehd-BLAS-reproducer/cuBLAS-ZGEMM-solve.cpp, line 140  <br/>
cudaDeviceSynchronize failed!
+ `cgemm-test`  <br/> cudaAssert: cudaErrorIllegalAddress an illegal memory access was encountered, file /home/razakh/dcehd-BLAS-reproducer/cuBLAS-CGEMM-solve.cpp, line 140  <br/>
cudaDeviceSynchronize failed!

