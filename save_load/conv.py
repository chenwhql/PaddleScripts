import paddle
import paddle.nn.functional as F
import paddle.static as static

x_var = paddle.randn((2, 3, 8, 8), dtype='float32')
w_var = paddle.randn((6, 3, 3, 3), dtype='float32')

y_var = F.conv2d(x_var, w_var)
y_np = y_var.numpy()

print(y_np.shape)
# (2, 6, 6, 6)