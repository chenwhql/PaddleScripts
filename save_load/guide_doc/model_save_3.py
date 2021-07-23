import paddle
import paddle.nn as nn

IMAGE_SIZE = 784
CLASS_NUM = 10

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static
    def forward(self, x, label=None):
        out = self._linear(x)
        if label:
            loss = nn.functional.cross_entropy(out, label)
            avg_loss = nn.functional.mean(loss)
            return out, avg_loss
        else:
            return out