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

# enable dygraph mode
fluid.enable_dygraph() 

# train model
net = SimpleNet(8, 8)
adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
for i in range(10):
    out = net(x)
    loss = fluid.layers.mean(out)
    loss.backward()
    adam.minimize(loss)
    net.clear_gradients()

model_path = "simplenet.example.model.model_filename"
configs = fluid.dygraph.jit.SaveLoadConfig()
configs.model_filename = "__simplenet__"

# saving with configs.model_filename
fluid.dygraph.jit.save(
    layer=net,
    model_path=model_path,
    input_spec=[x],
    configs=configs)

# loading with configs.model_filename
infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
pred = infer_net(x)