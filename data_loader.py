import sys
import os

os.environ["CPU_NUM"] = "1"

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddleslim as slim

places = fluid.cpu_places()
place = places[0]
exe = fluid.Executor(place)

train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    with fluid.unique_name.guard():
        x = layers.data("x", shape=[-1, 10], dtype="float32")
        label = layers.data("y", shape=[-1, 1], dtype="float32")
        loader = fluid.io.DataLoader.from_generator([x, label], 1)
        y = layers.fc(x, size=1, param_attr="fc.w_0", bias_attr="fc.b_0")
        loss = layers.mse_loss(y, label)
        avg_loss = layers.mean(loss)
        opt = fluid.optimizer.Adam()
        opt.minimize(avg_loss)
exe.run(startup_program)

def data_generator():
    x_np = np.random.rand(10, 10).astype("float32")
    y_np = np.random.rand(10, 1).astype("float32")
    def __generator__():
        print("haha")
        for i in range(0, 10, 2):
            print(i)
            yield x_np[i:i+2], y_np[i:i+2]
        print("finish")
    return __generator__

loader.set_batch_generator(data_generator(), places=places)
# train_program = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=avg_loss.name)
print("begin")
for data in loader():
    print(data)
    out = exe.run(train_program, feed=data, fetch_list=[avg_loss])
    print(out)
print("over")