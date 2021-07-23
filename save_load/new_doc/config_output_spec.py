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
        loss = paddle.tensor.mean(z)
        return z, loss

# enable dygraph mode
paddle.disable_static() 

# train model
net = SimpleNet(8, 8)
adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
x = paddle.randn([4, 8], 'float32')
for i in range(10):
    out, loss = net(x)
    loss.backward()
    adam.step()
    adam.clear_grad()

# use SaveLoadconfig.output_spec
model_path = "simplenet.example.model.output_spec"
config = paddle.SaveLoadConfig()
config.output_spec = [out]
paddle.jit.save(
    layer=net,
    model_path=model_path,
    config=config)

infer_net = paddle.jit.load(model_path)
x = paddle.randn([4, 8], 'float32')
pred = infer_net(x)
