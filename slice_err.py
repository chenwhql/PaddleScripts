import paddle.fluid as fluid
import numpy as np
import six

with fluid.dygraph.guard():
    data = np.random.random((2, 80, 16128)).astype("float32")
    print(data.shape)
    var = fluid.dygraph.to_variable(data)
    print(var.shape)
    if six.PY3:
        sliced = var[:,:,:var.shape[2]]
    elif six.PY2:
        sliced = var[:,:,:16000L]   
    print(sliced.shape)