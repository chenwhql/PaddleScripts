import numpy as np
import paddle
import paddle.static as static
import paddle.nn.functional as F

paddle.enable_static()

x = static.data(name="x", shape=[None, 3, 8, 8], dtype="float32")
w = static.data(name="w", shape=[None, 3, 3, 3], dtype="float32")
y = F.conv2d(x, w)

exe = static.Executor()
exe.run(static.default_startup_program())

x_var = np.random.random((2, 3, 8, 8)).astype("float32")
w_var = np.random.random((2, 3, 3, 3)).astype("float32")
y_val = exe.run(feed={"x": x_var, "w": w_var}, fetch_list=[y.name])

print(static.default_main_program())
clipped_program = static.default_main_program()._remove_training_info(clip_extra=True)
print(clipped_program)



