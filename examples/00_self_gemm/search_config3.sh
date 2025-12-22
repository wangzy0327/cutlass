#!/bin/bash

# 定义参数范围：256的整数倍，从256到6144
start=256
end=6144
step=256

# 固定alpha和beta
alpha=1
beta=0

# 输出结果文件
result_file="gemm_performance_comparison3.csv"

# 写入CSV表头
echo "m,n,k,cutlass_time_ms,cutlass_gflops,cublas_time_ms,cublas_gflops" > $result_file

# 遍历m/n/k的组合（可根据需求调整为m/n/k独立遍历，或固定其中两个）
for m in $(seq $start $step $end); do
    for n in $(seq $start $step $end); do
        for k in $(seq $start $step $end); do
            echo "========================================"
            echo "Testing shape: m=$m, n=$n, k=$k"
            echo "========================================"
            
            # 执行测试程序，捕获输出
            output=$(./self_gemm.out $m $n $k $alpha $beta 2>&1)
            
            # 提取Cutlass时间和GFLOPS
            cutlass_time=$(echo "$output" | grep "Cutlass GEMM time:" | awk '{print $4}')
            cutlass_gflops=$(echo "$output" | grep "Cutlass GEMM Performance:" | awk '{print $4}')
            
            # 提取cuBLAS时间和GFLOPS
            cublas_time=$(echo "$output" | grep "Cublas GEMM time:" | awk '{print $4}')
            cublas_gflops=$(echo "$output" | grep "Cublas GEMM Performance:" | awk '{print $4}')
            
            # 输出当前结果
            echo "Cutlass: $cutlass_time ms | $cutlass_gflops GFLOPS"
            echo "cuBLAS:  $cublas_time ms | $cublas_gflops GFLOPS"
            echo ""
            
            # 写入CSV文件（处理可能的空值）
            echo "$m,$n,$k,$cutlass_time,$cutlass_gflops,$cublas_time,$cublas_gflops" >> $result_file
        done
    done
done

# 输出总结
echo "========================================"
echo "测试完成！结果已保存至 $result_file"
echo "推荐分析："
echo "- cuBLAS适合大尺寸（如≥2048）且shape规整的矩阵"
echo "- Cutlass在中小尺寸或非规整shape下表现更优（需结合实际数据）"
