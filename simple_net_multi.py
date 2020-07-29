import numpy
import os
import paddle.fluid as fluid

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
parallel_places = [fluid.CUDAPlace(0), fluid.CUDAPlace(1)] if use_cuda else [fluid.CPUPlace()] * 2

# 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
# 否则fluid会把逻辑核的所有数目设为CPU_NUM，
# 在这种情况下，输入的batch size应大于CPU_NUM，
# 否则程序会异常中断。
if not use_cuda:
    os.environ['CPU_NUM'] = str(2)

exe = fluid.Executor(place)

data = fluid.data(name='X', shape=[None, 1], dtype='float32')
hidden = fluid.layers.fc(input=data, size=10)
loss = fluid.layers.mean(hidden)

test_program = fluid.default_main_program().clone(for_test=True)
fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(fluid.default_startup_program())
compiled_train_prog = fluid.CompiledProgram(
    fluid.default_main_program()).with_data_parallel(
            loss_name=loss.name, places=parallel_places)
# 注意：如果此处不设置share_vars_from=compiled_train_prog，
# 测试过程中用的参数与训练使用的参数是不一致
compiled_test_prog = fluid.CompiledProgram(
    test_program).with_data_parallel(
            share_vars_from=compiled_train_prog,
            places=parallel_places)

train_data = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(compiled_train_prog,
                  feed={"X": train_data},
                  fetch_list=[loss.name])
test_data = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(compiled_test_prog,
                  feed={"X": test_data},
                  fetch_list=[loss.name])