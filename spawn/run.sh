# export GLOG_vmodule=buffered_reader=2,gpu_info=2

python -m paddle.distributed.launch --gpus=0,1,2,3 launch_dataloader.py 
# python -m paddle.distributed.launch --selected_gpus=2,3 launch_old_dataloader.py 

# python spawn_dataloader.py