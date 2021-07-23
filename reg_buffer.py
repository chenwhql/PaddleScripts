import paddle
import paddle.nn as nn

class LinearNet(nn.Layer):
    def __init__(self, mean=paddle.to_tensor([1.0])):
        super(LinearNet, self).__init__()
        self.register_buffer("mean", mean, persistable=True)
        self._linear = nn.Linear(1, 1)

    def forward(self, x):
        return self._linear(x)


mean = paddle.to_tensor([0.1234])
linear = LinearNet(mean)
paddle.save(linear.state_dict(), 'model/model.pdparams')
print(linear.mean)

linear1 = LinearNet()
linear1.set_state_dict(paddle.load('model/model.pdparams'))
print(linear1.mean)