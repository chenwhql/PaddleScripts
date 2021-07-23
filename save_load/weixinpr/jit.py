
class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static(
        input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
    def forward(self, x):
        return self._linear(x)

...

# save
path = "example.model/linear"
paddle.jit.save(layer, path)


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static
    def forward(self, x):
        return self._linear(x)

...

# save
path = "example.model/linear"
paddle.jit.save(layer, path,
    input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
        self._linear_2 = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static(
        input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
    def forward(self, x):
        return self._linear(x)

    @paddle.jit.to_static(
        input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
    def another_forward(self, x):
        return self._linear_2(x)

# define layer
layer = LinearNet()

# save
path = "example.model/linear"
paddle.jit.save(layer, path)
