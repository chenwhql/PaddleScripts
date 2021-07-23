import numpy as np
import paddle.fluid as fluid

t = fluid.create_lod_tensor(np.ndarray([5, 30]), [[2, 3]], fluid.CUDAPlace(0))
p = t._place()
new_place = fluid.CUDAPlace(p.gpu_device_id())