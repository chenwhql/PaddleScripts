import numpy as np
import paddle
import paddle.fluid as fluid
from custom_relu import relu2

paddle.enable_static()

data = fluid.layers.data(name='data', shape=[32], dtype='float32')
relu = relu2(data)
use_gpu = True # or False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

x = np.random.uniform(-1, 1, [4, 32]).astype('float32')
out, = exe.run(feed={'data': x}, fetch_list=[relu])
np.allclose(out, np.maximum(x,0.))

paddle.disable_static()
t = paddle.to_tensor(x)
out = relu2(t)