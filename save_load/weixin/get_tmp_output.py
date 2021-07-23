import paddle
import paddle.nn as nn

IMAGE_SIZE = 784

# load
path = "example.model/linear"
loaded_layer = paddle.jit.load(path)

print(loaded_layer.program())

# inference
loaded_layer.eval()
x = paddle.randn([1, IMAGE_SIZE], 'float32')
pred = loaded_layer(x)
print(pred)