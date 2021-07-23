import paddle
import numpy as np

x = np.random.random([2, 3]).astype(np.complex64) + 1j * np.random.random(
    [2, 3]).astype(np.complex64)

y = np.random.random([2, 3]).astype(np.complex64) + 1j * np.random.random(
    [2, 3]).astype(np.complex64)

x = paddle.to_tensor(x)
x.stop_gradient = False
y = paddle.to_tensor(y)
y.stop_gradient = False

print("x:", x)
print("y:", y)

out = paddle.multiply(x, y)
print("out:", out)

out.backward()

print(x.grad)