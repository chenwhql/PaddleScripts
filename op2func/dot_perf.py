import paddle
import numpy as np
import yep

paddle.set_device("cpu")
x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
y_data = np.random.uniform(1, 3, [10]).astype(np.float32)

x = paddle.to_tensor(x_data, stop_gradient=True)
y = paddle.to_tensor(y_data, stop_gradient=True)

for i in range(100):
    z = paddle.dot(x, y)

yep.start("dot_dy.prof")
for i in range(100000):
    z = paddle.dot(x, y)
yep.stop()