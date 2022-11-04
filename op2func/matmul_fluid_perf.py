import paddle
import numpy as np
import yep

paddle.set_device("gpu")
x_data = np.random.random([1]).astype(np.float32)
y_data = np.random.random([1]).astype(np.float32)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)

for i in range(100):
    z = paddle.fluid.layers.matmul(x, y)

yep.start("prof/matmul_fluid_2.2.prof")
for i in range(1000000):
    z = paddle.fluid.layers.matmul(x, y)
yep.stop()

print("Success")