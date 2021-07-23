from paddle.fluid.initializer import Constant
import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import declarative

class GPT2LMHeadModel(fluid.dygraph.Layer):
    def __init__(self):
        super(GPT2LMHeadModel, self).__init__()
        self.embedding0 = paddle.nn.Embedding(50257, 768, padding_idx=-1, sparse=False)
        self.embedding1 = paddle.nn.Embedding(1024, 768, padding_idx=-1, sparse=False)
        self.lm_head_weight = paddle.to_tensor(np.random.rand(2, 3).astype('float32'))

    @declarative
    def forward(self, x1):
        x1 = fluid.layers.reshape(x1, shape=[-1, 6])
        x27 = -1
        x59 = fluid.layers.shape(x1)[0]
        x60 = fluid.layers.shape(x1)[1]
        x61 = x60
        x62 = [-1, 6]
        x63 = fluid.layers.reshape(x=x1, shape=x62)
        x101, x102, x103 = fluid.layers.split(input=x63, dim=1, num_or_sections=3)
        x110 = fluid.layers.shape(x101)
        return x101
    
input_ids = np.array([[15496,    11,   616,  3290,   318, 13779]]).astype("int64")

place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    paddle.jit.set_verbosity(3)
    model = GPT2LMHeadModel()
    model.eval()
    input_ids = paddle.to_tensor(input_ids)
    out = model(input_ids)
    print(out)