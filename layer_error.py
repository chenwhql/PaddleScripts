import paddle.nn as nn
import paddle

class Detect(nn.Layer):
    def __init__(self):
        super(Detect, self).__init__()
        self.stride = None

# class Detect(nn.Layer):
#     stride = None

d = Detect()
print(d.__dict__)

d.stride = paddle.zeros([1])
print(d.stride)

