import paddle
import paddle.nn as nn
import paddle.optimizer as opt

class SimpleNet(nn.Layer):
    def __init__(self, in_size, out_size):
        super(SimpleNet, self).__init__()
        self._linear = nn.Linear(in_size, out_size)

    @paddle.jit.to_static
    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        return z

# enable dygraph mode
paddle.disable_static() 

# train model
net = SimpleNet(8, 8)
adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
x = paddle.randn([4, 8], 'float32')
for i in range(10):
    out = net(x)
    loss = paddle.tensor.mean(out)
    loss.backward()
    adam.step()
    adam.clear_grad()

model_path = "simplenet.example.model.separate_params"
config = paddle.jit.SaveLoadConfig()
config.separate_params = True

# saving with configs.separate_params
paddle.jit.save(
    layer=net,
    model_path=model_path,
    config=config)
# [result] the saved model directory contains:
# linear_0.b_0  linear_0.w_0  __model__  __variables.info__

# loading with configs.params_filename
infer_net = paddle.jit.load(model_path, config=config)
x = paddle.randn([4, 8], 'float32')
pred = infer_net(x)