import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import declarative
import paddle
from paddle.fluid.dygraph.jit import SaveLoadConfig

class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()

    @declarative
    def forward(self, x):
        return x 

# enable dygraph mode
fluid.enable_dygraph() 
# 1. train & save model.
# create network
net = Model()
x =  fluid.dygraph.to_variable(np.array([1]).astype('float32'), name='x')
out1 = net(x)
model_path = "linear.example.model"

fluid.dygraph.jit.save(
    layer=net,
    model_path=model_path,
    input_spec=[x])

# 2. load model & inference
# load model
infer_net = fluid.dygraph.jit.load(model_path)

# inference
x = fluid.dygraph.to_variable(np.random.random((1)).astype('float32'))
pred = infer_net(x)
print(pred)
