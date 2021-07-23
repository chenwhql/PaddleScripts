import paddle
import numpy as np

x = np.random.random([1]).astype("float32")
y = paddle.to_tensor(x)
print("x: ", x)
print("y: ", y)

z = x + y
print("z: ", z)