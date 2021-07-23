import paddle
import paddle.nn as nn

IMAGE_SIZE = 784
CLASS_NUM = 10

class LoadingLinearNet(nn.Layer):
    def __init__(self, path):
        super(LoadingLinearNet, self).__init__(path)
        self._load_linear = paddle.jit.load(path)

    @paddle.jit.to_static
    def forward(self, x):
        return self._load_linear(x)

# create network
path = "example.model/linear"
layer = LoadingLinearNet(path)

# inference
layer.eval()
x = paddle.randn([1, IMAGE_SIZE], 'float32')
pred = layer(x)
print(pred)