import paddle
import numpy as np

paddle.enable_static()

x = paddle.static.data()
x = np.array([[1,2,3], [4,5,6]]).astype('float32')
x = paddle.to_tensor(x)
y_train = paddle.nn.functional.dropout(x, 0.5)