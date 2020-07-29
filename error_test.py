import paddle.fluid as fluid
import numpy as np
layer_norm = fluid.LayerNorm([32, 32])
x2 = fluid.layers.data(name='x2', shape=[3, 32, 32], dtype="int32")
layer_norm(x2)