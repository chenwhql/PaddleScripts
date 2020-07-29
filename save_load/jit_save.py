import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import declarative

class SimpleNet(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(SimpleNet, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        return z

with fluid.dygraph.guard(fluid.CPUPlace()):
    net = SimpleNet(8, 8)
    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
    x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
    for i in range(10):
        out = net(x)
        loss = fluid.layers.mean(out)
        loss.backward()
        adam.minimize(loss)
        net.clear_gradients()

# Save inference model.
model_path = "./dy2stat_infer_model"
fluid.dygraph.jit.save(net, model_path, [x])
