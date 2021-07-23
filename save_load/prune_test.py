import paddle
from paddle.static import InputSpec
from paddle.fluid.dygraph import Linear

import numpy as np

class LinerNetWithLabel(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super(LinerNetWithLabel, self).__init__()
        self._linear = Linear(in_size, out_size)

    @paddle.jit.to_static(input_spec=[
        InputSpec(
            shape=[None, 784], dtype='float32', name="image"), InputSpec(
                shape=[None, 1], dtype='int64', name="label")
    ])
    def forward(self, x, label):
        out = self._linear(x)
        loss = fluid.layers.cross_entropy(out, label)
        avg_loss = fluid.layers.mean(loss)
        return out

layer = LinerNetWithLabel(784, 1)

model_path = "test_prune_to_static_no_train/model"
# TODO: no train, cannot get output_spec var here
# now only can use index
# output_spec = layer.forward.outputs[:1]
paddle.jit.save(
    layer,
    model_path,
    input_spec=[
        InputSpec(
            shape=[None, 784], dtype='float32', name="image")
    ])