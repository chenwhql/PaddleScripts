# GPU_settting
# cuda path

USE_CUDA=true
export FLAGS_fraction_of_gpu_memory_to_use=0.01
export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_fast_eager_deletion_mode=0
export FLAGS_benchmark=true
export CUDA_VISIBLE_DEVICES=4,5,6,7   # which GPU to use
#export GLOG_v=10

echo "the python your use is `which python`"
python -u pyreader_test.py