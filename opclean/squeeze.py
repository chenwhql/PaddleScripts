import numpy as np
import paddle
import paddle.static as static
import paddle.nn.functional as F

paddle.enable_static()

x = static.data(name="x", shape=[None, 1, 10], dtype="float32")
y = paddle.squeeze(x, axis=1)

exe = static.Executor()
exe.run(static.default_startup_program())

x_var = np.random.random((5, 1, 10)).astype("float32")
y_val = exe.run(feed={"x": x_var}, fetch_list=[y.name])

print(static.default_main_program())
# print(y_val)


