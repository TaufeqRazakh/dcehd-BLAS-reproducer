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

+ `zgemm-test` does not print message after BLAS call
+ `cgemm-test` fails with malloc failure 
