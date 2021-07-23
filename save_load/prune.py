import paddle
from paddle.static import InputSpec

class TestLayer(paddle.nn.Layer):
    def __init__(self):
        super(TestLayer, self).__init__()

    @paddle.jit.to_static(
        input_spec=[
            InputSpec(shape=[1], dtype='float32', name='x'), 
            InputSpec(shape=[1], dtype='float32', name='y'), 
            InputSpec(shape=[1], dtype='float32', name='z')])
    def forward(self, x, y, z):
        result = x + y
        return result

layer = TestLayer()
paddle.jit.save(layer, "dy2stat_prune_save",
    input_spec=[
        InputSpec(shape=[1], dtype='float32', name='x'), 
        InputSpec(shape=[1], dtype='float32', name='y')])