import os

import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper

# 调用load_op_library加载动态库
# os.environ["FLAGS_op_dir"] = "/work/scripts/custom_op"
fluid.load_op_library('relu2_op.so')

def relu2(x, name=None):
    # relu2的type和在OP中定义的type相同
    helper = LayerHelper("relu2", **locals())
    # 创建输出Variable
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="relu2", inputs={"X": [x]}, outputs={"Y": out})
    return out
