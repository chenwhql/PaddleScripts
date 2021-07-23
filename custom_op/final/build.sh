include_dir=$( python3.7 -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python3.7 -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

# exec obj: failed
# g++ relu_op.cc -o relu2_op -std=c++11 -O3 \
#   -I ${include_dir} \
#   -I ${include_dir}/third_party \
#   -L /usr/local/cuda/lib64 \
#   -L ${lib_dir} -lpaddle_framework -lcudart

# g++ relu_op.cc -o relu2_op.so -shared -fPIC -std=c++17 -O3 \
#   -I ${include_dir} \
#   -I ${include_dir}/third_party \
#   -L /usr/local/cuda/lib64 \
#   -L ${lib_dir} -lpaddle_framework -lcudart

# PaddlePaddel >=1.6.1, 仅需要include ${include_dir} 和 ${include_dir}/third_party
nvcc relu_op.cu -c -o relu_op.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \

g++ relu_op.cc relu_op.cu.o -o relu2_op.so -shared -fPIC -std=c++11 -O3 \
  -I ${include_dir} \
  -I ${include_dir}/third_party \
  -L /usr/local/cuda/lib64 \
  -L ${lib_dir} -lpaddle_framework -lcudart