import os
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import OpProtoHolder

# export LD_LIBRARY_PATH=/work/scripts/custom_op/new:$( python3.7 -c 'import paddle; print(paddle.sysconfig.get_lib())'):$LD_LIBRARY_PATH

paddle.disable_static()

fluid.core.load_custom_op('relu2_op.so')

OpProtoHolder.instance().update_op_proto()
op_proto = OpProtoHolder.instance().get_op_proto("relu2")
print(op_proto)


def relu2(x, name=None):
    # relu2的type和在OP中定义的type相同
    helper = LayerHelper("relu2", **locals())
    # 创建输出Variable
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="relu2", inputs={"X0": [x]}, outputs={"Out": out})
    return out


paddle.disable_static()

paddle.set_device('cpu')
x = np.random.uniform(-1, 1, [4, 32]).astype('float32')
t = paddle.to_tensor(x)
out = relu2(t)

print(out)