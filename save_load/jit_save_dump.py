import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import declarative

import os
import pickle
import six

# import sys
# sys.setrecursionlimit(10000)

class SimpleNet(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(SimpleNet, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        loss = fluid.layers.mean(z)
        return z, loss

with fluid.dygraph.guard(fluid.CPUPlace()):
    net = SimpleNet(8, 8)
    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
    x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
    for i in range(10):
        loss, out = net(x)
        loss.backward()
        adam.minimize(loss)
        net.clear_gradients()

# Save inference model.
model_path = "./dy2stat_infer_model_with_layer"
fluid.dygraph.jit.save(net, model_path, [x], [loss])

layer_path = os.path.join(model_path, "__layer__")
with open(layer_path, 'wb') as f:
    pickle.dump(net, f, protocol=2)

with fluid.dygraph.guard(fluid.CPUPlace()):
    with open(layer_path, 'rb') as f:
        layer = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')

    print(layer)

    # x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
    # loss, out = layer(x)
    # print(loss)
    # print(out)
