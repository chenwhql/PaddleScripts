import paddle
import numpy as np

paddle.set_device('gpu')

a = paddle.to_tensor(np.random.random((4, 4)).astype(np.double))

c = paddle.sum(a)

c.backward()