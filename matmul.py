import paddle
import numpy as np

# vector * vector
x_data = np.random.random([10]).astype(np.float64)
y_data = np.random.random([10]).astype(np.float64)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
z = paddle.matmul(x, y)
print(z)