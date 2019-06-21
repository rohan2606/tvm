rm logs/error logs/*p3*.out


python3 scripts/gpu_test.py
echo "running llvm unfused"
python3 op_profiling_mxnet.py 'llvm' 'unfused' > logs/p3_CPU_op_unfused.out 2>>logs/error
echo "running llvm fused"
python3 op_profiling_mxnet.py 'llvm' 'fused' > logs/p3_CPU_op_fused.out 2>>logs/error
echo "running cuda unfused"
python3 op_profiling_mxnet.py 'cuda' 'unfused' > logs/p3_GPU_op_unfused.out 2>>logs/error
echo "running cuda fused"
python3 op_profiling_mxnet.py 'cuda' 'fused' > logs/p3_GPU_op_fused.out 2>>logs/error
