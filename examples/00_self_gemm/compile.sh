nvcc -std=c++17 -I$HOME/cuda-code/cutlass/include -I$HOME/cuda-code/cutlass/examples/common -I$HOME/cuda-code/cutlass/tools/util/include  -arch=sm_70 self_gemm.cu -o self_gemm.out -lcublas -lcudart
