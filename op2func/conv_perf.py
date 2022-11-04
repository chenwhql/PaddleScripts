import os
os.environ["FLAGS_enable_eager_mode"] = "1"

import paddle
import paddle.nn.functional as F
import yep

paddle.set_device("gpu")
x_var = paddle.randn((2, 3, 8, 8), dtype='float32')
w_var = paddle.randn((6, 3, 3, 3), dtype='float32')

yep.start("conv_final_state.prof")
for i in range(10000):
  y_var = F.conv2d(x_var, w_var)
  y_np = y_var.numpy()
  # print(y_np.shape)
yep.stop()
