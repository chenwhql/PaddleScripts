import paddle
import paddle.nn as nn
from paddle.static import InputSpec

IMAGE_SIZE = 784
CLASS_NUM = 10

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
    def forward(self, x):
        return self._linear(x)

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
    def another_forward(self, x):
        tmp = self._linear(x)
        return self._linear(tmp)

# create network
layer = LinearNet()

# save
path = "example.model/linear"
paddle.jit.save(layer, path)
