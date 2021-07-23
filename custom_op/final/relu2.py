import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import OpProtoHolder

# export LD_LIBRARY_PATH=/work/scripts/custom_op/final:$( python3.7 -c 'import paddle; print(paddle.sysconfig.get_lib())'):$LD_LIBRARY_PATH

def load_custom_op(so_name):
    fluid.core.load_custom_op(so_name)
    OpProtoHolder.instance().update_op_proto()
    # op_proto = OpProtoHolder.instance().get_op_proto("relu2")
    # print(op_proto)

def relu2(x, name=None):
    helper = LayerHelper("relu2", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="relu2", inputs={"X0": [x]}, outputs={"Out": [out]})
    return out